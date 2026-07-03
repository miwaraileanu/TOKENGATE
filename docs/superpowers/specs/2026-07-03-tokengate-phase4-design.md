# TokenGate Phase 4 — Cascade Router Design

## Overview

Phase 4 activates `layers/router.py`, which has been a no-op stub since Phase 1. The router implements **cascade model routing**: score each request's difficulty with fast heuristics, route easy requests to a cheap model, run a self-confidence check after the cheap response, and escalate to the strong model only when necessary.

The router becomes the **terminal pipeline layer** for non-streaming requests — it makes all upstream calls itself and sets `ctx.response`. Streaming requests skip the router (documented limitation; streaming bypass is clean, buffering defeats streaming's purpose).

---

## 1. Pipeline Reorder

**Old order:**
```
exact_cache → semantic_cache → distiller → compressor → router → budgeter
```

**New order (Phase 4):**
```
exact_cache → semantic_cache → distiller → compressor → budgeter → router
```

Budgeter **must run before router** because the router makes the actual upstream call. If budgeter ran after, its `max_tokens` injection would arrive too late to affect the call. Budgeter only mutates the request — it is safe anywhere before the first upstream call.

The server's short-circuit rule (`if ctx.response is not None: break`) still terminates the pipeline correctly after a cache hit, before either budgeter or router.

**Change required:** update `_PIPELINE` list in `proxy/server.py`.

---

## 2. Streaming Skip

Router's first check:

```python
if req.stream:
    ctx.decisions.append(LayerDecision("router", "skip", {"reason": "streaming"}))
    return ctx
```

The server's fallback `call_upstream` then handles the request as in passthrough mode. This is a documented Phase 4 limitation. Streaming routing (with buffer-then-re-stream) is deferred to Phase 5.

---

## 3. Configuration

Added to `Settings.__init__` from the `router:` block of `tokengate.yaml`:

```yaml
router:
  enabled: true
  cheap_model:
    anthropic: claude-haiku-4-5
    openai: gpt-4o-mini
  strong_model:
    anthropic: claude-sonnet-4-6
    openai: gpt-4o
  difficulty_threshold: 0.4       # ≥ this → strong tier directly
  escalation_enabled: true
  escalation_threshold: 3         # confidence ≤ this → escalate
  tools_tier: strong              # "cheap" | "strong" — default strong
```

**Provider routing:** only the pair matching `req.route` is used. Anthropic requests never touch OpenAI model names and vice versa.

**Settings attributes:**

```python
_r = raw.get("router", {})
self.router_enabled: bool = bool(_r.get("enabled", True))
self.router_difficulty_threshold: float = float(_r.get("difficulty_threshold", 0.4))
self.router_escalation_enabled: bool = bool(_r.get("escalation_enabled", True))
self.router_escalation_threshold: int = int(_r.get("escalation_threshold", 3))
self.router_tools_tier: str = _r.get("tools_tier", "strong")
_cm = _r.get("cheap_model", {})
self.router_cheap_model: dict[str, str] = {
    "anthropic": _cm.get("anthropic", "claude-haiku-4-5"),
    "openai": _cm.get("openai", "gpt-4o-mini"),
}
_sm = _r.get("strong_model", {})
self.router_strong_model: dict[str, str] = {
    "anthropic": _sm.get("anthropic", "claude-sonnet-4-6"),
    "openai": _sm.get("openai", "gpt-4o"),
}
```

**Disabled router (`router_enabled=False`):** log `LayerDecision("router", "skip", {"reason": "disabled"})` and return immediately. The server's fallback handles the call.

---

## 4. Difficulty Scoring

Pure heuristics in Phase 4. Score is a `float` in `[0.0, 1.0]`:

| Feature | Max contribution | Rationale |
|---|---|---|
| Prompt token count | `min(tokens / 2000, 0.40)` | Long prompts tend harder |
| `req.tools` non-empty | `+0.25` | Tool-calling is structurally complex |
| Fenced code block in user messages | `+0.15` | Code tasks often require precision |
| Math symbols (`∑∫√±×÷=∂` or `\b(sin\|cos\|integral\|derivative)\b`) | `+0.10` | Math → strong reasoning needed |
| Multi-step markers (`step \d`, `^\d+\.` numbered lists, or `\bthen\b` adjacent to imperative verbs) | `+0.10` | Sequential reasoning harder |
| Conversation depth (non-system turns) | `min(turns / 20, 0.10)` | Deep context increases complexity |

**Sum clamped to `[0.0, 1.0]`.**

`_score_difficulty(req, settings) -> tuple[float, dict]` — returns `(total_score, feature_dict)` where `feature_dict` maps feature names to their per-feature contribution values. Both are logged in `LayerDecision.detail`.

---

## 5. Tier Selection

Evaluated in order — first match wins:

```
1. router_enabled is False        → skip("disabled")
2. req.stream                     → skip("streaming")
3. req.tools AND tools_tier==strong → strong, reason="tools_forced_strong"
4. header x-tokengate-tier=strong → strong, reason="client_override"
5. difficulty >= threshold        → strong, reason="above_threshold"
6. otherwise                      → cheap tier (self-check may escalate)
```

For paths 3–5 (strong directly): call strong model, no self-check (escalation is **one-directional only** — never run a confidence check on a strong-tier response).

---

## 6. Upstream Calls

### 6.1 Same function, no parallel implementation

The router calls `provider.call_upstream(synth_req, settings, transport=_transport)` — the identical function used by the server. `synth_req` is the original `ctx.request` with only `model` replaced by the selected tier's model name; all prior layer mutations (distiller's compressed messages, budgeter's `max_tokens`, etc.) are preserved.

**Module-level transport override for tests:**
```python
_transport: httpx.AsyncBaseTransport | None = None

def set_transport(t) -> None:
    global _transport
    _transport = t
```

### 6.2 Error relay (cheap call)

If the cheap-tier `call_upstream` raises `UpstreamError`:
- Do not catch it in the router.
- Let it propagate to the server's existing `UpstreamError` handler, which relays the body to the client and logs `status="upstream_error"`.
- This is identical behaviour to the no-router passthrough path.

### 6.3 Error relay (strong-call-after-escalation)

If the strong-tier call fails **after** a successful cheap call:
- **Serve the cheap response** — a below-par answer beats a 500 when we have an answer.
- Set `ctx.response = cheap_resp`.
- Log `escalation_reason="escalation_failed_served_cheap"`.
- `est_saved_usd` reflects only the cheap + check overhead (no strong cost, since it never completed).

If **both** the cheap call and the strong call fail, relay the strong-call `UpstreamError` (we have nothing to serve).

---

## 7. Self-Check (Escalation Check)

Only runs when: cheap tier was selected AND `router_escalation_enabled=True`.

### 7.1 Micro-prompt

```
"Does this response fully and correctly answer the question below?
Rate your confidence 1 (not at all) to 5 (completely).
Reply with exactly one digit and nothing else.

Question: {last_user_message[:500]}
Response: {cheap_response[:1000]}"
```

**Call parameters:**
- Model: `cheap_model[req.route]` (same cheap model, same provider)
- `max_tokens=5`, `temperature=0` (one digit; do not burn tokens or get creative)
- `stream=False`

### 7.2 Parsing

Strip whitespace. Valid response: single character in `{'1', '2', '3', '4', '5'}`.

### 7.3 Fail-safe (fail toward quality)

| Outcome | Action | `escalation_reason` |
|---|---|---|
| Call raises `UpstreamError` | Escalate | `"check_call_failed"` |
| Response not parseable | Escalate | `"check_parse_failed"` |
| Score ≤ `escalation_threshold` (default 3) | Escalate | `"low_confidence"` |
| Score > threshold | Serve cheap response | — (no escalation) |

A broken or unavailable self-check **always escalates**. Never serve a cheap answer when the check itself failed — the missing verification is itself a quality risk.

### 7.4 No escalation loops

The self-check only ever runs on cheap-tier responses. Strong-tier responses are served directly, regardless of their quality. Escalation is strictly one-directional.

---

## 8. Cost Accounting (Honest Numbers)

### 8.1 Baseline cost — counterfactual estimate

`baseline_cost_usd` = what the strong model **alone** would have cost for this request. Since the strong model never ran on non-escalated requests, its output token count is unknown. The formula uses the **served response's output tokens** as a proxy:

```python
baseline_cost_usd = compute_cost(
    strong_model_name,
    tokens_in=served_resp.tokens_in,       # actual input tokens sent
    tokens_out=served_resp.tokens_out,     # output tokens of SERVED response
    settings=settings,
)
```

**⚠ This is an estimate.** The strong model might have produced a longer or shorter answer. This assumption is documented in `LayerDecision.detail["baseline_is_estimate"] = True` and in the dashboard tooltip: *"Baseline cost assumes strong model would produce same output length — actual saving may differ."*

### 8.2 `est_saved_usd` formula per scenario

| Scenario | `est_cost_usd` | `est_saved_usd` |
|---|---|---|
| Strong directly (above threshold / override) | `strong_cost` | `0.0` (no alternative cheaper) |
| Cheap, no check | `cheap_cost` | `baseline - cheap_cost` (**positive**) |
| Cheap + check, no escalation | `cheap_cost + check_cost` | `baseline - (cheap + check)` |
| Cheap + check + escalation | `cheap_cost + check_cost + strong_cost` | `baseline - (cheap + check + strong)` = `-(cheap + check)` (**negative — routing overhead**) |
| Escalation failed (strong 500) | `cheap_cost + check_cost` | `baseline - (cheap + check)` |

`est_saved_usd` is written to `requests.est_saved_usd` (already supports negative). The stats endpoint's `SUM(est_saved_usd)` will go negative if escalation rate is high enough — this surfaces routing overhead in the dashboard without any special handling.

### 8.3 Server integration

The server reads `est_cost_usd` and `est_saved_usd` from the router's `LayerDecision.detail` for `write_row`. It also sets `escalated=1` on the DB row when `decision.detail["escalated"]` is True. The `ctx.response is not None` branch in `_non_streaming_response` is extended to check for a router decision:

```python
router_decision = next(
    (d for d in ctx.decisions if d.layer == "router" and d.action == "applied"),
    None,
)
if router_decision:
    est_cost  = router_decision.detail["est_cost_usd"]
    est_saved = router_decision.detail["est_saved_usd"]
    escalated = int(router_decision.detail["escalated"])
else:
    # cache hit path (unchanged)
    est_cost  = 0.0
    est_saved = compute_cost(model, tokens_in, tokens_out, s) or 0.0
    escalated = 0
```

---

## 9. LayerDecision Schema

```python
LayerDecision("router", "applied", {
    "difficulty":            float,         # 0.0–1.0 total score
    "features": {                           # per-feature contributions
        "length":            float,
        "tools":             float,
        "code":              float,
        "math":              float,
        "multi_step":        float,
        "depth":             float,
    },
    "tier":                  "cheap"|"strong",
    "model":                 str,           # model that produced ctx.response
    "reason":                str,           # tier selection reason
    "escalated":             bool,
    "confidence_score":      int|None,      # 1–5; None if check not run
    "escalation_reason":     str|None,
    "est_cost_usd":          float,         # actual cost paid
    "baseline_cost_usd":     float,         # counterfactual (strong alone)
    "baseline_is_estimate":  True,          # always True — always a counterfactual
    "est_saved_usd":         float,         # may be negative
})
```

---

## 10. Training Data and `retrain_router.py`

### 10.1 What is logged

Every router `applied` decision stores `difficulty`, `features`, and `escalated` in `layers_applied` JSON on the `requests` row. This is the training set: `(feature_vector, did_escalate)`.

### 10.2 Selection bias — document clearly

**Only cheap-routed requests have escalation labels.** Requests routed to strong directly (above threshold, tool override, client override) have no `did_escalate` label — they never went through the self-check. `retrain_router.py` must filter to `WHERE tier='cheap'` and document: *"This model only learns the difficulty boundary within the cheap-routed region. It cannot predict outcomes for requests already above the hard threshold."* Future work: use the strong model's responses as pseudo-labels for unlabeled high-difficulty requests.

### 10.3 `retrain_router.py` (manual offline script)

```python
# Pseudocode — actual implementation in scripts/retrain_router.py
features, labels = load_from_db(db_path, tier='cheap')
model = LogisticRegression().fit(features, labels)
save_coefficients(model, data_dir / "router_model.pkl")
```

The router loads `router_model.pkl` at startup if it exists and uses its `predict_proba` output as the difficulty score instead of the heuristic sum. If the file is absent (default), heuristic scoring applies. **No self-training in the request path. Ever.**

---

## 11. DB Schema

No new columns needed. `escalated INTEGER NOT NULL DEFAULT 0` already exists in `requests` per the Phase 1 schema. The `write_row` call already accepts an `escalated` parameter — the server just needs to pass it from the router decision.

`baseline_cost_usd` is stored only in `layers_applied` JSON (inside the router `LayerDecision.detail`), not as a first-class column. It is a derived counterfactual estimate, not a measured fact.

---

## 12. Tests

### 12.1 `tests/test_router.py` — Unit

- Difficulty: each feature in isolation contributes the right delta
- Difficulty: combined score is correct and clamped to 1.0
- `req.stream` → skip decision, no upstream call
- `router_enabled=False` → skip decision
- `x-tokengate-tier: strong` → strong model called, no check
- `req.tools` + `tools_tier=strong` → strong model, reason=`tools_forced_strong`
- Cheap: confidence=4 → cheap served, `est_saved_usd > 0`
- Cheap: confidence=2 → escalated to strong, `est_saved_usd < 0`
- Self-check call returns 500 → escalate, reason=`check_call_failed`
- Self-check returns "excellent" → escalate, reason=`check_parse_failed`
- Self-check call params: `max_tokens=5`, `temperature=0`, cheap model
- Strong-not-checked: confidence check never runs on strong-direct responses
- Cheap upstream 500 → `UpstreamError` propagates (not caught by router)
- Strong fails after escalation → cheap response served, reason=`escalation_failed_served_cheap`
- Both cheap and strong fail → `UpstreamError` propagated (nothing to serve)

### 12.2 Integration — additions to `tests/test_proxy.py`

- Budgeter's injected `max_tokens` reaches the mock upstream when router is active (pipeline order test)
- Escalated request → `requests.escalated = 1` in DB
- Non-escalated cheap → `est_saved_usd > 0` in DB row
- Escalated request → `est_saved_usd < 0` in DB row

---

## 13. Limitations

- **Streaming:** router skips all streaming requests. Streaming clients get no routing benefit.
- **Baseline is an estimate:** `baseline_cost_usd` assumes the strong model produces the same output length as the served response. Dashboard labels this explicitly.
- **Selection bias in training data:** the logistic regression trained by `retrain_router.py` only covers the cheap-routed region.
- **One-directional escalation:** no downgrade path (strong → cheap). If `tools_tier=strong` is set, tool requests never benefit from cheap routing even for trivially easy tool calls.
- **Single escalation:** no double-escalation or multi-tier cascades. One cheap → one strong max.
- **No per-route overrides in Phase 4:** difficulty thresholds and tier config are global. Per-route policies deferred to Phase 5.
