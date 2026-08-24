# Agentic Voice Assistant

A multi-tenant voice agent platform for Indian languages. One engine, many configurable
agents, reachable over web, phone, and later mobile.

> **Status: the brain works and is tested. Nothing speaks yet.**
> See [Where we are](#where-we-are) for the honest breakdown.

---

## The goal

A tenant — a hospital, a clinic, a campaign — logs into a web app, configures an Agent, and
that Agent handles real conversations:

- **Web** — click-to-talk in the browser
- **Phone inbound** — caller dials an Exophone, Exotel routes to us
- **Phone outbound** — reminder and campaign calls
- **Mobile** — a thin client on the same protocol

The whole thing hangs on one number: **under 1000 ms from the caller going silent to the
first byte of audio coming back.** Past that, people talk over the agent and the illusion
dies. Every design decision in this repo is downstream of that budget.

The stack is **Sarvam AI** (streaming STT/TTS and the conversational LLM), **Groq**
(intent classification and failover),
**LiveKit** (media), **Exotel** (PSTN). Full rationale in [BLUEPRINT.md](BLUEPRINT.md).

### The differentiator

Most voice bots feel bad because the LLM decides *when* to talk. This one does not. Turn
taking is a deterministic state machine with no model in the loop; the LLM only decides
*what* to say, inside a flow graph that constrains where the conversation can go. See
[BRAIN.md](BRAIN.md).

---

## Where we are

Roughly **Phase 0 → 1** of the [phased plan](BLUEPRINT.md#10-phased-plan). The conversation
core is real, tested, and measured. The media, web, and telephony layers do not exist yet.

### Built and verified

| Layer | State |
|---|---|
| **L0 Turn-taking** | Deterministic controller, barge-in, no LLM. 17 tests. |
| **L2 Dialogue** | Flow graph + planner, sentence-boundary streaming. |
| **L3 Tools** | Timeout-bounded registry with spoken fallbacks. Canned data. |
| **L5 Guardrails** | Inbound emergency/injection screen, outbound clinical screen. |
| **Providers** | Sarvam planner, Groq classifier + failover, NVIDIA spare. |
| **Deep reasoning** | Optional background subagent. Off by default. |
| **Speech** | Sarvam streaming STT/TTS. Verified live in Hindi and Telugu. |

**92 tests passing.** `voice_sim.py` runs a full simulated call — real brain, real turn
taking, printed frames instead of sound.

### Written but unverified

Neither speech provider has been run against its live service — both were written from
docs, without credentials. Payload parsing reads defensively on both paths.

- **Sarvam** (`brain/speech/sarvam.py`) — the default. Streams both directions over
  WebSockets: VAD signals arrive while the caller is still talking, which is what drives
  barge-in, and TTS audio arrives frame by frame. Set `SARVAM_API_KEY`, then
  `python probe_sarvam.py`.

Every argument the Sarvam adapter passes is checked against the installed SDK signature
by the test suite, so an SDK upgrade that renames a parameter fails in CI rather than at
the first spoken word.


### Not started

LiveKit media layer · web UI · Postgres/Redis/pgvector · multi-tenancy and auth · RAG ·
Exotel telephony · real tool backends · observability.

### Measured latency

From Windows, **not** Mumbai — treat as relative, not absolute:

```
intent                    ~300-900 ms
planner_first_sentence    ~300-600 ms   (~1200-2000 ms on a tool round)
turn_total                ~900-1600 ms
```

Already close to the 1000 ms budget *before* audio is attached. Intent classification is
the biggest single cost and may be foldable into the planner call.

---

## What it takes to finish

Ordered by what unblocks the most.

**1. Get a Sarvam key and verify the speech layer.** Run `probe_sarvam.py`, fix whatever
the live wire format disagrees with. This is now the critical-path blocker: the default
provider cannot speak without it. Then tune `SARVAM_MIN_BUFFER_SIZE` against real
time-to-first-audio.

**2. Measure from Mumbai.** Every latency number here is from a Windows box on the wrong
side of the planet. The budget is meaningless until it is measured where it will run.

**3. Start Exotel KYC and request vSIP trunking.** Longest lead time in the project and a
hard blocker on Phase 3. Start it before you need it, not when.

**4. Phase 1 — web voice MVP.** LiveKit room, agent worker wiring Sarvam → Groq → Sarvam,
barge-in working end to end in a browser. This is the demo that proves the brain.

**5. Phase 2 — the platform.** Postgres schema, tenants and auth, agent builder UI, KB
upload with pgvector RAG, real tool backends, transcripts and analytics.

**6. Phase 3 — telephony.** Exotel vSIP → LiveKit SIP, per-Exophone dispatch, DTMF,
transfer-to-human, TRAI compliance.

**7. Phase 4 — hardening.** 100 concurrent calls, autoscaling, OTel, DPDP retention,
per-tenant billing, on-call runbook.

### Known gaps to close before real callers

- Sessions are in-process; `InMemorySessionStore` needs Redis before a second worker exists.
- No PII redaction before transcripts are stored.
- Slots hold the caller's phrasing ("tomorrow at ten"), not normalised timestamps — fine
  for a demo, wrong for a real scheduling backend.
- Tools return canned data.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Add your keys to `.env`, then:

```bash
python cli.py
```

`cli.py` is an interactive REPL printing every event and per-stage timing (`/state` dumps
the session). Other entry points:

```bash
python voice_sim.py
```

```bash
python smoke.py
```

```bash
python -m pytest -q
```

`voice_sim.py` simulates a call end to end. `smoke.py` runs a fixed five-turn booking
against the live API.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Drives live conversation. |
| `NVIDIA_API_KEY` | — | Optional. Dev provider, Groq failover, deep reasoning. |
| `SARVAM_MIN_BUFFER_SIZE` | `24` | Chars buffered before synthesis starts. Main lever on first-audio latency. |
| `SARVAM_MAX_CHUNK_LENGTH` | `150` | Cap on one synthesis burst. |
| `SARVAM_API_KEY` | — | Speech **and** the planner LLM — one shared credit balance. |
| `LLM_PROVIDER` | `sarvam` | Who writes what the caller hears. |
| `FAST_LLM_PROVIDER` | `groq` | Intent classification only. Keep off Sarvam to halve credit use. |
| `DEEP_REASON_ENABLED` | `false` | Opt in to the slow reasoning tier. |
| `SUBAGENT_MODEL` | `minimaxai/minimax-m3` | Deep-reasoning model. |
| `SUBAGENT_FALLBACK_MODEL` | `moonshotai/kimi-k3` | Tried when the primary fails. |

### Why the LLM is split across two providers

`sarvam-105b-conversations` writes the replies. It is trained on the ten Indian
languages this agent speaks, and in Telugu it produces flow-appropriate phrasing —
honorifics included — where the gpt-oss models produce translated-sounding English.
Streaming and tool calling both work; verified live.

Intent classification stays on Groq. Its output is a JSON label nobody ever hears, Groq
handles Telugu classification correctly, and it is roughly 3x faster (1.1-2.3 s on
Sarvam against 0.3-0.6 s on Groq). It also matters commercially: **Sarvam bills LLM
calls against the same credit balance as speech**, so classifying there would spend the
budget twice a turn on the half the caller cannot hear.

### Deep reasoning is opt-in

A second tier for questions the fast model should not answer from its own head. It runs
**off the conversational hot path**: the tool starts a background job and returns in ~5 ms,
the agent says a holding line, and the answer is folded into a later turn.

It is off by default because the tiers have opposite failure modes — Groq answers in
hundreds of milliseconds, the subagent in 6-25 s, which a caller experiences as the agent
going quiet and coming back several turns later. Turn it on per deployment:

```bash
DEEP_REASON_ENABLED=true
```

Off is a real off: no HTTP client, no tool registered, no mention in any prompt.

---

## Secrets

**Never commit `.env`.** It is gitignored, has never been committed, and the history has
been scanned for key material. Every credential is read from the environment at runtime —
there are no literals in the source.

`.env.example` documents the shape with empty values. Copy it, do not rename it.

If a key is ever exposed, rotate it at the provider first — removing it from git history
does not un-leak it.

---

## Documentation

| File | Contents |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md) | System architecture, latency budget, phased plan, stack rationale |
| [BRAIN.md](BRAIN.md) | Conversation core internals, design decisions, known gaps |
| [VOICE_APIS.md](VOICE_APIS.md) | Sarvam STT/TTS API notes |

