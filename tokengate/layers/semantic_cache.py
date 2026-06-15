from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayResponse

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

_HAVE_EMBEDDER: bool = _HAVE_NUMPY and _HAVE_ST

_model = None
_embed_fn = None  # injectable for tests
_index: OrderedDict = OrderedDict()  # cache_key → (emb: np.ndarray, body_json: str)


def set_embedder(fn) -> None:
    """Tests inject a fake embedding function here. Pass None to reset."""
    global _embed_fn
    _embed_fn = fn


def can_embed() -> bool:
    return _HAVE_EMBEDDER or _embed_fn is not None


def _embed(text: str):
    if _embed_fn is not None:
        return _embed_fn(text)
    if not _HAVE_EMBEDDER:
        raise RuntimeError("sentence-transformers not installed and no embed_fn injected")
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode([text], normalize_embeddings=True)[0]


def _query_text(req) -> str:
    """System prompt fingerprint (first 100 chars) + last user message."""
    system_fp = ""
    last_user = ""
    for msg in req.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_fp = (content if isinstance(content, str) else str(content))[:100]
        elif role == "user":
            if isinstance(content, list):
                last_user = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                last_user = str(content)
    return f"{system_fp}\n{last_user}".strip()


def _is_blocked(text: str, patterns: list) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


async def verify_hit(query: str, cached_answer: str) -> bool:
    """Phase 2 stub — always True. Phase 4 replaces with cheap-model micro-prompt."""
    return True


def load_index(db_path: Path, max_entries: int) -> None:
    """Rebuild in-memory index from SQLite after server restart. Requires numpy."""
    if not _HAVE_NUMPY:
        return
    con = sqlite3.connect(db_path)
    rows = con.execute(
        # ASC so oldest entry is at front of OrderedDict — required for LRU eviction
        "SELECT cache_key, embedding, body_json FROM cache_semantic ORDER BY ts ASC"
    ).fetchall()
    con.close()
    for key, emb_bytes, body_json in rows[-max_entries:]:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        _index[key] = (emb, body_json)


async def apply(ctx: LayerContext) -> LayerContext:
    if not can_embed():
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "no_embedder"}))
        return ctx

    if ctx.settings is None:
        return ctx

    req = ctx.request

    if req.stream:
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "streaming"}))
        return ctx

    if req.tools:
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "has_tools"}))
        return ctx

    settings = ctx.settings
    query_text = _query_text(req)

    if _is_blocked(query_text, settings.cache_blocklist):
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "blocklisted"}))
        return ctx

    query_emb = _embed(query_text)

    best_key: str | None = None
    best_score: float = -1.0
    for key, (emb, _) in _index.items():
        score = float(np.dot(query_emb, emb))
        if score > best_score:
            best_score = score
            best_key = key

    threshold = settings.cache_semantic_threshold

    if best_key is not None and best_score >= threshold:
        _, body_json = _index[best_key]
        body = json.loads(body_json)

        if best_score >= 0.97:
            ctx.response = GatewayResponse(**body)
            ctx.decisions.append(LayerDecision("semantic_cache", "hit", {
                "score": round(best_score, 4),
                "key": best_key,
                "verified": True,
            }))
            _index.move_to_end(best_key)
            return ctx

        # 0.93 ≤ score < 0.97 — unverified band
        verified = await verify_hit(query_text, body["content"])
        if settings.cache_serve_unverified and verified:
            ctx.response = GatewayResponse(**body)
            ctx.decisions.append(LayerDecision("semantic_cache", "hit", {
                "score": round(best_score, 4),
                "key": best_key,
                "verified": verified,
            }))
            _index.move_to_end(best_key)
            return ctx

        ctx.decisions.append(LayerDecision("semantic_cache", "miss", {
            "score": round(best_score, 4),
            "reason": "unverified_blocked",
        }))
    else:
        ctx.decisions.append(LayerDecision("semantic_cache", "miss", {
            "best_score": round(best_score, 4) if best_score >= 0 else None,
        }))

    db_path = settings.db_path
    max_entries = settings.cache_max_entries
    _qemb = query_emb

    async def _write(response: GatewayResponse) -> None:
        key = hashlib.sha256(query_text.encode()).hexdigest()[:32]
        body_json = json.dumps({
            "content": response.content,
            "model": response.model,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "stop_reason": response.stop_reason,
            "raw_body": response.raw_body,
        })
        emb_bytes = _qemb.astype(np.float32).tobytes()
        ts = time.time()

        _con = sqlite3.connect(db_path)
        _con.execute(
            "INSERT OR REPLACE INTO cache_semantic (cache_key, embedding, body_json, ts) VALUES (?,?,?,?)",
            (key, emb_bytes, body_json, ts),
        )
        _con.commit()
        _con.close()

        if key not in _index and len(_index) >= max_entries:
            _index.popitem(last=False)
        _index[key] = (_qemb, body_json)
        _index.move_to_end(key)

    ctx.cache_writers.append(_write)
    return ctx
