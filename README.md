# TokenGate

**Cut your LLM API costs by 50–90% — by changing one line of code.**

TokenGate is a drop-in local proxy that sits between your application and the LLM API (Anthropic / OpenAI compatible). It transparently applies five stacked savings layers to every request and shows you exactly how much you saved — per request, per layer, in dollars.

```python
# Before
client = Anthropic()

# After — that's the only change
client = Anthropic(base_url="http://localhost:8787")
```

Built in Dublin by [RAIT](https://www.rait.ie).

---

## Why

LLM API bills grow quietly: repeated questions hit the API again and again, conversation histories balloon, simple requests go to expensive models, and verbose answers burn output tokens. Each of these problems has a known fix — but applying them means rewriting your app.

TokenGate applies all of them **outside your code**, with full transparency: every optimization decision is visible in response headers, logs, and a live dashboard. No silent quality degradation.

## The five layers

| Layer | What it does | Typical savings |
|---|---|---|
| **L1 Exact cache** | Identical request → instant answer, zero API tokens | 100% on repeats |
| **L2 Semantic cache** | Paraphrased question → finds the cached answer via local embeddings (no API cost) | 100% on near-repeats |
| **Context distiller** | Long chat history → rolling summary + only the relevant past turns. Pinned user facts are never lost | 40–80% of input |
| **Cascade router** | Easy request → cheap model. Hard request → strong model. Auto-escalates if the cheap answer fails a self-check | 60–90% on easy traffic |
| **Output budgeter** | Sensible `max_tokens` + concision hints per request type | 10–30% of output |

Safety rails: time-sensitive queries, personal data, and tool calls **bypass caches entirely**. Lossy compression is opt-in. Every applied layer is reported in `x-tokengate-*` response headers.

## Quickstart

### Install

```bash
pip install rait-tokengate
# or the one-liner (isolated venv, no pip knowledge needed):
curl -fsSL https://rait.ie/tokengate/install.sh | bash
```

### Set up & run

```bash
rait install     # interactive wizard: provider, API key, port
rait start       # launch the gateway
rait status      # health check
```

### Point your app at it

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8787")
```

Works with the OpenAI SDK the same way (`OpenAI(base_url="http://localhost:8787/v1")`).

### Watch the savings

```bash
rait stats       # terminal summary
```

Or open **http://localhost:8787/dashboard** — savings by layer, cache hit rates, escalation rate, your most expensive routes.

## CLI reference

| Command | Description |
|---|---|
| `rait install` | Interactive setup wizard (`--yes` for non-interactive) |
| `rait start` / `rait stop` | Start/stop the gateway (`--detach` for background) |
| `rait status` | Health: port, uptime, upstream reachability, cache size |
| `rait stats` | Tokens & dollars saved, by layer |
| `rait test` | Send a sample request, see which layers fired |
| `rait config` | Edit configuration (`rait config set key value`) |
| `rait cache clear` | Wipe caches (`--exact`, `--semantic`, `--all`) |
| `rait update` | Self-update |
| `rait uninstall` | Clean removal |

## Configuration

Everything lives in `~/.rait/tokengate.yaml` — similarity thresholds, model tiers, per-route policies, cache TTLs, blocklist patterns. Your API key is stored separately in `~/.rait/.env` with `0600` permissions and never appears in config or logs.

Example tier setup:

```yaml
tiers:
  - name: cheap     # claude-haiku-4-5
    max_difficulty: 0.4
  - name: strong    # claude-sonnet-4-6
    max_difficulty: 1.0
```

## Security

- Binds to **localhost only** by default. Exposing it externally requires explicitly setting a `TOKENGATE_KEY` auth header — never run it on `0.0.0.0` without one.
- Cached responses may contain user data: enable at-rest encryption with `TOKENGATE_ENCRYPT_KEY`, and use `rait cache clear` / the `DELETE /cache` admin endpoint for data removal (GDPR).
- No external network calls except your configured upstream providers.
- All requests still go through the official provider APIs with your key — TokenGate does not bypass any provider policies or moderation.

## How honest are the numbers?

Semantic caching, model cascading, and prompt compression each exist as separate research and products. TokenGate's contribution is the **transparent, self-measuring combination**: one proxy, five layers, per-request decisions, and an escalation log that improves routing over time.

Your actual savings depend entirely on your traffic: a FAQ chatbot with repetitive questions can exceed 90%; a stream of unique, hard, long-form tasks might see 15–30% (mostly from distillation and budgeting). Run it in passthrough mode first — Phase 1 gives you full spend visibility with zero optimization — then enable layers and compare. The dashboard never shows estimated marketing numbers, only measured ones.

## Architecture

```
Your app ──HTTP──▶ TokenGate ──▶ LLM Provider
                      │
   L1 exact cache → L2 semantic cache → distiller
        → compressor (opt-in) → cascade router → budgeter
                      │
            SQLite analytics + dashboard
```

Python 3.12 · FastAPI · local sentence-transformers embeddings · SQLite. No heavy vector DB, no frontend build step, fully testable offline with the bundled mock provider.

## Development

```bash
git clone https://github.com/YOUR_ORG/tokengate
cd tokengate
pip install -e ".[dev]"
pytest                  # full suite runs offline (mock provider)
```

See `TOKENGATE.md` for the complete build specification and acceptance criteria.

## License

MIT — see [LICENSE](LICENSE).

---

*Questions, audits, or managed setup for your business: [rait.ie](https://www.rait.ie) · misha@rait.ie*
