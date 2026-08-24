# The Brain — implementation notes

Text-in, events-out conversation core. No audio anywhere: the voice layer (Sarvam STT/TTS
over LiveKit) attaches later by consuming the same event stream. See [BLUEPRINT.md](BLUEPRINT.md)
for the wider system.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python cli.py
```

```bash
python smoke.py
```

```bash
python -m pytest -q
```

`cli.py` is an interactive REPL showing every event and per-stage timing (`/state` dumps the
session). `smoke.py` runs a fixed five-turn booking against the live API.

## Layout

| File | Role |
|---|---|
| `brain/models.py` | Immutable domain types: `Session`, `Turn`, `IntentResult`, events |
| `brain/flow.py` | Flow graph + `next_node()` — pure, deterministic, no I/O |
| `brain/intent.py` | LLM intent + slot extraction on the fast model |
| `brain/planner.py` | Prompt building, tool loop, sentence-boundary streaming |
| `brain/tools.py` | Timeout-bounded tool registry with spoken fallbacks |
| `brain/guardrails.py` | Inbound emergency/injection screen, outbound clinical-advice screen |
| `brain/memory.py` | Session store behind a `Protocol` (swap for Redis/Postgres) |
| `brain/orchestrator.py` | `Brain.handle()` — one turn, start to finish |
| `brain/agents/hospital.py` | Seed agent, pure configuration |

## Turn pipeline

```
guardrails(in) -> intent+slots -> flow decision -> planner(+tools) -> guardrails(out) -> persist
```

`Brain.handle(session_id, text)` is an async generator yielding `SayEvent`, `ToolEvent`,
`IntentEvent`, `TransitionEvent`, `EscalateEvent`, `EndEvent`, `TimingEvent`.

`SayEvent`s arrive at **sentence boundaries**, not at completion — that is the hook speech
synthesis will use to start talking while the model is still writing.

## Providers

Groq and NVIDIA NIM are both OpenAI-compatible, so `brain/providers.py` makes the choice
configuration rather than code. `LLM_PROVIDER=groq|nvidia`; the unset provider becomes automatic
failover (`LLM_FALLBACK=0` disables). Both are verified end to end — a full booking completes on
either.

| | planner | intent | notes |
|---|---|---|---|
| `groq` (default) | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | `reasoning_effort=low` |
| `nvidia` (free dev) | `nvidia/nemotron-3-nano-30b-a3b` | same | thinking off, 3x timeout budget |

Measured time-to-first-token, three samples each, from Windows:

```
Groq   gpt-oss-20b               323 /  448 /  527 ms
Groq   gpt-oss-120b              512 /  536 /  558 ms
NVIDIA nemotron-3-nano-30b-a3b   631 /  714 /  835 ms
NVIDIA nemotron-nano-9b-v2      1477 / 1477 / 1477 ms
NVIDIA nemotron-3.5-lightning   1154 / 4029 / 9412 ms
```

**Use NVIDIA for development, Groq for live calls.** The median NVIDIA latency is workable; the
spread is not. A 9-second turn is a dead call, and free endpoints are shared so the spread is not
under your control. As a free provider for prompt iteration, flow testing, and burning zero Groq
credits, it is excellent.

Two NVIDIA-specific gotchas, both handled in `providers.py`:

- **Nemotron thinks out loud.** Reasoning is on by default and lands in `content`, not a separate
  field — the agent said *"We need to follow instructions: greet in one short line"* aloud. Fixed
  with `extra_body={"chat_template_kwargs": {"thinking": False}}`.
- **Timeouts are provider-scoped.** Budgets tuned for Groq made NVIDIA time out on every turn,
  degrade to intent `unknown`, and escalate the call. `Provider.timeout_scale` fixes it.

The Llama models in most Groq tutorials are **not** available on this key; check
`client.models.list()` before changing model names.

## The audio path

**Decision: Sarvam only for speech.** Bhashini, Gemini Live, ElevenLabs, Deepgram, AssemblyAI and
NVIDIA Riva were evaluated and rejected — see [VOICE_APIS.md](VOICE_APIS.md) for the comparison.
Sarvam wins on Indian-language accuracy, is the only candidate with proper streaming WebSockets
plus VAD signals in all four target languages, and keeps one vendor across STT and TTS.

`brain/speech/sarvam.py` implements it, written against `sarvamai` 0.1.30:

- **`Ear`** — streaming STT (`saaras:v3`) with `vad_signals=true`. Emits `speech_start` (the
  barge-in trigger), `speech_end`, and `transcript` on one stream.
- **`Mouth`** — streaming TTS (`bulbul:v3`). One socket per utterance, on purpose: Sarvam's TTS
  WebSocket has no cancel message, so closing the connection is the only way to stop generation on
  barge-in, and a shared long-lived socket could not be interrupted. Text is pushed in on a
  background task while audio streams out, so early frames play while later sentences are still
  being written.

**Status: unverified against the live service** — written without a key. Add `SARVAM_API_KEY` to
`.env` and run:

```bash
python probe_sarvam.py
```

It speaks a line with Bulbul, feeds the audio straight back into Saaras, and prints the transcript
plus first-frame latency for English, Hindi, Tamil, Telugu and Kannada. No microphone required.
Expect the parsing in `_parse_stt` / `_parse_tts_audio` to need adjusting — the wire format is
documented loosely, which is why both read defensively by attribute rather than assuming a schema.

### L0 — the turn-taking controller

`brain/voice/controller.py` binds `Ear` events to `Brain.handle()` and `SayEvent`s to `Mouth`.
It depends on three narrow protocols — `SpeechSource`, `Synthesizer`, `AudioSink` — so the whole
thing is exercised with fakes: **18 tests, no audio, no network, no API key.**

```bash
python voice_sim.py     # real brain, scripted caller, printed audio
```

**Barge-in order is load-bearing:** stop feeding the sink → `sink.clear()` the transport's buffer
→ cancel the turn (closing the TTS socket). Cancelling first would leave already-buffered audio
playing after generation stopped, so the agent keeps talking for a second after being interrupted.

Three bugs the simulation caught that unit tests alone would not have:

- **The source ending cancelled the reply in flight** — `run()` killed the turn task before it
  ever executed, so a single-utterance call produced no speech at all. A graceful end of stream
  now waits for the turn; a hangup cancels `run()` from the transport side.
- **`_ending` was checked after handling an event**, so a transcript arriving after the call had
  already escalated started a fresh turn on a dead session.
- **The filler played audio without claiming the floor** — it bypassed the state machine, so the
  agent was audibly saying "One moment." while the controller still believed it was idle and
  ignored a caller trying to interrupt. All audio now goes through one `_emit()` path.

Two tunables, both measured rather than guessed:

- `BARGE_IN_GRACE` (0.25 s) — ignore speech detected just as our own audio starts, or echo and
  the tail of our own greeting retrigger barge-in instantly. The cost is that a caller cannot
  interrupt inside that window, so keep it short.
- `FILLER_AFTER` (1.2 s) — at the 0.6 s the blueprint suggested, the filler fired on nearly every
  turn, which sounds worse than brief silence. At 1.2 s it fires only on tool rounds.

Still to build: transport adapters (`AudioSink` for LiveKit / Exotel / browser) and mic capture.

### NVIDIA pieces, for reference

Evaluated and not adopted, but worth knowing about.

### Maxine Studio Voice — adopt, as a pre-processor

Speech enhancement: denoise, dereverb, and lift a bad microphone toward studio quality. It sits
**before** STT and touches nothing else:

```
mic / phone audio -> Studio Voice -> Sarvam STT -> brain
```

For Indian deployments this is a real robustness win — hospital lobbies, street noise, cheap
handsets. Constraints to plan around:

- gRPC over NVCF (`grpc.nvcf.nvidia.com:443`), not the OpenAI-style REST endpoint.
- Streaming mode exists; use it plus the low-latency sub-mode. Transactional mode is for files.
- Sub-modes are 48 kHz and 16 kHz. Telephony is 8 kHz, so a resample is unavoidable, and the
  quality ceiling on a phone call is lower than the demos suggest.
- It adds a hop to the turn budget. Measure before committing; self-host the NIM container for
  production rather than depending on a shared free endpoint.

**Best immediate use, before any of that:** run recorded noisy audio through it offline and
measure how much word-error-rate Sarvam gains. That answers "is this worth a hop?" with a number,
and needs no live pipeline.

### Nemotron VoiceChat 11B — benchmark, do not adopt as the brain

An open **full-duplex speech-to-speech** model: ~448 ms turn-taking, listens while it speaks,
barge-in take-over 1.00 at 480 ms, tool calling, bidirectional WebSocket. Technically impressive
and genuinely faster than any cascade.

It is also **architecturally opposed to this system**. It collapses STT + LLM + TTS + turn-taking
into one model, which means it owns the brain. Everything in `brain/` that makes the agent
controllable — the flow graph, structural tool gating, the deterministic escalation rules, the
clinical-advice guardrail — degrades to asking a model nicely in a prompt. For a hospital or a
political campaign that is the wrong trade.

Three further blockers:

- **Licensing**: on build.nvidia.com it is early-access, evaluation purposes only. Not commercial.
- **Languages**: English-centric. Indian-language coverage is the entire reason Sarvam is in the
  stack.
- **Control**: no way to guarantee a booking tool actually ran before the agent says "confirmed" —
  the exact bug this codebase already had and fixed structurally.

Worth doing anyway: run it as a **latency benchmark** for what full-duplex feels like, and keep it
as a candidate for a low-stakes English web demo. Revisit if NVIDIA licenses it commercially and
adds Indian languages.

## Design decisions worth keeping

**The LLM never controls flow.** `next_node()` is a pure function over (agent, session, intent).
The planner writes words; the graph decides where the conversation is, when to escalate, and
when to stop.

**Tools are gated structurally, not by prompting.** A tool is offered only when its own required
arguments are already known slots. This is why `book_appointment` cannot fire before a slot is
chosen — the model is never given the option, so it cannot invent a confirmation. Gating on the
*node's* missing slots instead was tried and is wrong: at `offer_slots` the whole point of
`check_availability` is to produce the value that is missing.

**Booking happens at `close`, not `confirm`.** The confirm step only obtains a yes. If the tool
lived on `confirm`, the transition would fire on the intent and the booking would never run —
which is exactly the bug that produced a cheerful confirmation with a hallucinated reference
number.

**Language is sticky.** Detected per turn, but only switched on an utterance of 12+ characters
with confidence ≥0.8 in a language the agent speaks. Without this the agent flips language when a
caller says "haan", and the opening turn drifts into Hindi for an English caller.

**Speech output is defended three ways.** `_clean_for_speech()` drops punctuation-only chunks and
markdown artefacts; `MAX_SENTENCES_PER_TURN` abandons the stream when a model starts degenerating
(observed live: `**…**... … … …`); and a word-density floor rejects chunks that are mostly
punctuation or zero-width characters even when they contain a few real letters (observed live on
Nemotron: `An SMS ​​​​​​ on … ………………`). Density is judged *before* repeated punctuation is
collapsed — collapsing first turns a 40-character ellipsis flood into one "…" and it slips past.

**Everything degrades to a sayable line.** Intent failure → `unknown`. Tool failure/timeout →
the tool's `fallback_line`. Planner failure → an apology and a re-ask. A caller must never hear
silence.

## Known gaps (deliberate, not oversights)

- Sessions are in-process only — `InMemorySessionStore` needs replacing with Redis before more
  than one worker exists.
- No RAG yet. `search_kb` and pgvector retrieval are Phase 2.
- Tools return canned data. The registry contract does not change when they become HTTP calls.
- No PII redaction before transcripts are stored — required before real caller data.
- Slot values are stored as the caller's phrasing ("tomorrow at ten o'clock in the morning")
  rather than normalised timestamps. Fine for a demo, wrong for a real scheduling backend.
- Intent classification adds ~450-900 ms per turn. Worth revisiting once real latency is
  measured from Mumbai; it may be foldable into the planner call.

## Observed latency (from Windows, not Mumbai)

`intent` ~300-900 ms · `planner_first_sentence` ~300-600 ms (~1200-2000 ms on a tool round)
· `turn_total` ~900-1600 ms. A tool round costs an extra LLM round-trip — that is what the
filler-phrase trick in the blueprint is for.
