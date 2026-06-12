# TOKENGATE — Intelligent Token-Saving Gateway for LLM Apps

A build specification for Claude Code. Work through phases in order. Each phase ships something usable.

---

## 0. The idea in one paragraph

TokenGate is a **drop-in proxy** between any application and an LLM API (Anthropic / OpenAI compatible). The app changes one line — the base URL — and TokenGate transparently applies five stacked savings layers: **semantic caching, cascade model routing, context distillation, prompt compression, and output budgeting**. Every request gets the cheapest path that still meets a quality bar. Target: **50–90% token cost reduction** on real workloads (chatbots, agents, RAG apps) with measurable quality control. None of the individual techniques is new — the breakthrough is stacking them in one transparent gateway with a feedback loop that *learns which layer to apply per request type*.

---

## 1. Architecture

```
Client app ──HTTP──▶ TokenGate Proxy ──▶ LLM Provider(s)
                        │
        ┌───────────────┼──────────────────────┐
        ▼               ▼                      ▼
   [L1 Exact Cache] [L2 Semantic Cache]  [Router/Cascade]
        │               │                      │
        ▼               ▼                      ▼
   [Context Distiller]──▶[Prompt Compressor]──▶[Output Budgeter]
                        │
                        ▼
              [SQLite analytics + admin dashboard]
```

**Request lifecycle:**
1. Request arrives (OpenAI/Anthropic-compatible `/v1/messages` or `/v1/chat/completions`).
2. **L1 exact cache** — hash of normalized request. Hit → return instantly (0 tokens).
3. **L2 semantic cache** — embed the user query, cosine-search recent answers. Above threshold (default 0.93) → return cached answer, marked `x-tokengate-cache: semantic`.
4. **Context distiller** — if conversation history exceeds N tokens, replace old turns with a rolling summary + retrieve only the K most relevant past turns for the current query.
5. **Prompt compressor** — optional lossy compression of long context blocks (docs, code) keeping high-information tokens.
6. **Cascade router** — classify the request (heuristics + tiny classifier): easy → cheap model (Haiku-class), hard → strong model. If the cheap model's answer fails a self-check, escalate automatically.
7. **Output budgeter** — inject `max_tokens` and concision instructions appropriate for the request type; enforce structured output when the client asked for JSON.
8. Log everything: tokens in/out, layer decisions, latency, estimated $ saved.

---

## 2. Tech stack

- **Language:** Python 3.12, FastAPI, uvicorn (async throughout).
- **Storage:** SQLite (analytics + cache metadata), files or SQLite blobs for cached bodies.
- **Embeddings:** local `sentence-transformers` model (`all-MiniLM-L6-v2`) — zero API cost for the cache layer. Keep behind an interface so a hosted embeddings API can be swapped in.
- **Vector search:** `sqlite-vec` extension or plain numpy cosine over a bounded in-memory index (cap 50k entries, LRU eviction). No heavy vector DB.
- **Config:** single `tokengate.yaml` (thresholds, model tiers, budgets, per-route policies).
- **Dashboard:** FastAPI route serving one static HTML page (vanilla JS + a chart lib via CDN) — no frontend build step.
- **Tests:** pytest, with a **mock LLM provider** module so the entire pipeline is testable offline.

---

## 3. Modules to build

### 3.1 `proxy/server.py` — API-compatible gateway
- Endpoints: `POST /v1/chat/completions` (OpenAI shape) and `POST /v1/messages` (Anthropic shape). Internally normalize both to one `Request` dataclass.
- Streaming support (SSE pass-through) — cache layers store the assembled full text.
- Auth: forward the client's API key to the upstream provider; TokenGate itself protected by its own `TOKENGATE_KEY` header.
- Response headers expose decisions: `x-tokengate-cache`, `x-tokengate-model`, `x-tokengate-saved-tokens`.

### 3.2 `layers/exact_cache.py` (L1)
- Key = SHA-256 of (normalized messages + model + temperature bucket + tools).
- Normalization: strip whitespace runs, lowercase nothing (keep case — it matters), drop volatile fields (request id, timestamps).
- Only cache when `temperature <= 0.3` or the client opts in via header `x-tokengate-cache-write: always`.
- TTL configurable per route (default 24h).

### 3.3 `layers/semantic_cache.py` (L2)
- Embed the **last user message + a short fingerprint of system prompt**.
- Cosine similarity threshold from config (default 0.93). Below 0.97, prepend a tiny disclaimer-free re-ranking step: ask the cheap model "Does this cached answer fully answer the new question? yes/no" (costs ~50 tokens, saves thousands on hit).
- Never serve semantic hits for requests containing: time-sensitive words (today, now, latest, price), user-specific data, or tool calls. Maintain a blocklist regex in config.

### 3.4 `layers/distiller.py` — context distillation
- Trigger when history > `distill_threshold_tokens` (default 6000).
- Keep: system prompt, last `keep_recent_turns` (default 4) verbatim.
- Older turns → one rolling summary (generated by the cheap model, cached, updated incrementally — never re-summarize from scratch).
- Plus retrieval: embed all old turns once; for each new request, inject the top-K (default 3) most relevant old turns verbatim.
- Hard rule: distillation must never drop explicit user instructions/preferences — extract those into a persistent "pinned facts" block first.

### 3.5 `layers/compressor.py` — prompt compression
- For long document/code blocks inside the prompt (detected by fenced blocks or > 1500 token contiguous chunks):
  - Code: strip comments? NO — instead drop blank lines, collapse repeated boilerplate, and offer "signature mode" (keep signatures + docstrings, elide bodies of functions not mentioned in the query).
  - Prose: extractive compression — sentence scoring by embedding similarity to the user query, keep top sentences up to a budget.
- Compression is **opt-in per route** (`compress: true` in yaml) because it is lossy. Always log original vs compressed token counts.

### 3.6 `layers/router.py` — cascade routing
- Tiered models from config, e.g.:
  ```yaml
  tiers:
    - name: cheap   # e.g. claude-haiku-4-5
      max_difficulty: 0.4
    - name: strong  # e.g. claude-sonnet-4-6
      max_difficulty: 1.0
  ```
- Difficulty score (0–1) from fast heuristics: prompt length, presence of code, math symbols, multi-step instructions ("then", numbered lists), tool definitions, conversation depth. Start rule-based; Phase 4 trains a tiny logistic regression on logged escalations.
- **Escalation check:** after a cheap-tier response, run a self-verification micro-prompt on the same cheap model ("Rate 1-5 how confidently this answers the request; reply with one digit"). Score ≤ 3 → re-run on the strong tier. Log every escalation (this is the training data).
- Per-route override: clients can force a tier with `x-tokengate-tier: strong`.

### 3.7 `layers/budgeter.py` — output budgeting
- Request-type detection: chat / extraction / code / long-form.
- Inject sensible `max_tokens` if the client didn't set one (config table per type).
- For extraction-type requests, append a one-line system instruction: "Answer with the requested data only, no preamble." (Saves 10–30% output tokens on structured tasks.)

### 3.8 `cli/` — the `rait` command-line tool
The whole product is installed and operated through a single branded CLI: **`rait`**.

**Packaging:**
- Publishable Python package named `rait-tokengate` with a console-script entry point `rait` (`[project.scripts] rait = "tokengate.cli:main"` in `pyproject.toml`).
- Built with `typer` (or `click`) + `rich` for pretty terminal output.
- Also provide a one-line bootstrap installer for people without pip knowledge:
  ```bash
  curl -fsSL https://rait.ie/tokengate/install.sh | bash
  ```
  The script: checks Python ≥ 3.10 → creates an isolated venv in `~/.rait/` → pip-installs the package → symlinks `rait` into `~/.local/bin` (and prints instructions to add it to PATH if needed). Write this `install.sh` as part of the repo (`scripts/install.sh`).

**Commands:**

| Command | What it does |
|---|---|
| `rait install` | Interactive setup wizard (see below) |
| `rait start` / `rait stop` | Start/stop the gateway (daemon via `--detach`, logs to `~/.rait/logs/`) |
| `rait status` | Health check: running?, port, uptime, upstream reachability, cache size |
| `rait stats` | Savings summary in the terminal (tokens & $ saved by layer, hit rates) + link to web dashboard |
| `rait config` | Open `~/.rait/tokengate.yaml` in `$EDITOR`; `rait config set key value` for scripting |
| `rait cache clear` | Wipe caches (with `--semantic`, `--exact`, `--all` flags) |
| `rait test` | Send a sample request through the gateway and print which layers fired |
| `rait update` | Self-update the package inside its venv |
| `rait uninstall` | Clean removal: stop daemon, remove venv, ask before deleting data dir |

**`rait install` wizard flow:**
1. ASCII banner + version.
2. Ask: provider (Anthropic / OpenAI / both) and read API key (input hidden, stored only in `~/.rait/.env` with `chmod 600` — never in the yaml).
3. Ask: port (default 8787), cache mode (safe defaults pre-selected), model tiers (suggest sensible defaults per provider).
4. Write `~/.rait/tokengate.yaml` + `~/.rait/.env`, init SQLite DB.
5. Run a self-test request against the mock provider, print green checkmarks per layer.
6. Print the integration snippet, personalized with the chosen port:
   ```python
   client = Anthropic(base_url="http://localhost:8787")  # that's the only change
   ```
7. Offer to start now (`rait start --detach`) and to enable autostart (systemd user unit on Linux / launchd plist on macOS — generate the file, ask before installing it).

All wizard answers must also be settable via flags (`rait install --provider anthropic --port 8787 --yes`) for non-interactive/CI installs.

### 3.9 `analytics/` — measurement (this sells the product)
- SQLite tables: `requests(id, ts, route, layers_applied, tokens_in_raw, tokens_in_final, tokens_out, model_used, cache_kind, escalated, latency_ms, est_cost_usd, est_saved_usd)`.
- Dashboard page `/dashboard`: totals, savings by layer (stacked bar), cache hit rate over time, escalation rate, top expensive routes.
- `GET /stats` JSON endpoint for programmatic access.

---

## 4. Build phases

**Phase 1 — Skeleton + CLI + passthrough + analytics (ship first).**
FastAPI proxy, both API shapes, streaming passthrough, full request logging, dashboard with raw spend, **plus the `rait` CLI with `install`, `start`, `stop`, `status`** (wizard can be minimal at this stage). *Value even with zero optimization: visibility + one-command setup.*

**Phase 2 — Caching.** L1 exact, then L2 semantic with the local embedder. Mock-provider tests: identical request → 0 upstream calls; paraphrased request → semantic hit; "what time is it" → never cached.

**Phase 3 — Distiller + budgeter.** Rolling summary with pinned-facts extraction. Test: 50-turn synthetic conversation must keep a user preference stated in turn 2 visible to the model at turn 50.

**Phase 4 — Cascade router + escalation learning.** Heuristic router → log escalations → fit logistic regression weekly (`scripts/retrain_router.py`).

**Phase 5 — Compressor (opt-in) + hardening.** Rate limiting, request size caps, config hot-reload, Docker image, `scripts/install.sh` one-liner, `rait update` / `rait uninstall`, autostart units (systemd/launchd), README with one-line integration examples for the Anthropic and OpenAI SDKs.

---

## 5. Project structure

```
tokengate/
├── pyproject.toml          # package: rait-tokengate, entry point: rait
├── tokengate.yaml          # default config (copied to ~/.rait/ on install)
├── proxy/server.py
├── cli/{main.py,wizard.py,daemon.py}
├── layers/{exact_cache,semantic_cache,distiller,compressor,router,budgeter}.py
├── core/{normalize.py,tokens.py,provider.py,mock_provider.py}
├── analytics/{db.py,dashboard.html,stats.py}
├── scripts/{install.sh,retrain_router.py}
├── tests/
└── Dockerfile
```

---

## 6. Hard rules & safety

1. **Never degrade silently.** Every applied layer is visible in response headers and logs.
2. **Correctness beats savings.** Blocklisted patterns (time-sensitive, personal data, tool calls) bypass caches entirely.
3. **Lossy layers are opt-in** (compressor) or guarded (semantic cache re-rank check).
4. Cached bodies may contain user data → encrypt at rest if `TOKENGATE_ENCRYPT_KEY` is set; document retention; provide `DELETE /cache` admin endpoint (GDPR).
5. No external network calls except the configured upstream providers.

---

## 7. Acceptance criteria

- [ ] `pip install rait-tokengate` (or `pipx install`) exposes a working `rait` command
- [ ] `rait install --yes` completes non-interactively and `rait start` brings the gateway up; `rait status` reports healthy
- [ ] API key is stored only in `~/.rait/.env` with `0600` permissions, never echoed to the terminal or written into yaml
- [ ] `rait test` prints which layers fired on a sample request
- [ ] One-line integration works: changing `base_url` in the Anthropic SDK routes through TokenGate with identical responses on passthrough mode
- [ ] Streaming responses byte-identical to direct provider calls (passthrough)
- [ ] Exact-cache hit returns < 20 ms, zero upstream tokens
- [ ] Semantic cache: paraphrase benchmark (20 hand-written pairs in tests) ≥ 80% hit rate, 0 false hits on the time-sensitive blocklist set
- [ ] Distiller: pinned fact from turn 2 survives to turn 50 in the synthetic test
- [ ] Router: on the included 60-prompt benchmark (20 easy / 20 medium / 20 hard), ≤ 10% of hard prompts answered by cheap tier *without* escalation
- [ ] Dashboard shows per-layer savings; `est_saved_usd` math verified by a unit test
- [ ] Full test suite passes offline using the mock provider
- [ ] README quickstart gets a new user running in < 5 minutes

---

## 8. Honest framing (put this in the README)

Semantic caching, model cascading, and prompt compression each exist as separate research/products. TokenGate's contribution is the **transparent, self-measuring combination**: one proxy, five layers, per-request decisions, and an escalation log that improves routing over time. Publish real benchmark numbers, not marketing numbers.