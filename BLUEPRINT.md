# Agentic Voice Assistant — Architecture Blueprint

Stack (locked): **Sarvam AI** (STT + TTS, Indian languages) · **Groq** (LLM brain / orchestration)
· **Exotel AgentStream / VoiceBot Applet** (PSTN calling) · **Web** as the first surface.

---

## 1. Product shape

Multi-tenant SaaS. One engine, many "Agents". A tenant (hospital, political campaign, clinic,
individual) logs into the web app, configures an Agent, and that Agent is reachable via:

- **Web** — click-to-talk widget / dashboard mic (Phase 1)
- **Phone inbound** — caller dials an Exophone → VoiceBot applet → our WSS
- **Phone outbound** — reminder/campaign calls via the Exotel Connect API
- **Mobile app** — Phase 5, a thin client on the same protocol

Key insight: **a surface is just a transport.** Everything above the media layer is identical.

---

## 2. High-level architecture

```
 ┌──────────┐   ┌──────────┐          Surfaces
 │ Browser  │   │  Exotel  │
 │ mic/spkr │   │ VoiceBot │
 └────┬─────┘   └────┬─────┘
      │ WebRTC       │ vSIP trunk → LiveKit SIP bridge
      └───────┬──────┘
              ▼
   ┌──────────────────────────┐
   │  LIVEKIT ROOM            │ one media layer for both surfaces;
   │  (Cloud, ap-south)       │ codecs, AEC, jitter, framing handled
   └───────────┬──────────────┘
               ▼
   ┌─────────────────────────────────────────────────────────┐
   │            SESSION ORCHESTRATOR  ("the brain")          │
   │                                                         │
   │  L0 Turn-taking  → deterministic, NO LLM                │
   │  L1 Perception   → Sarvam STT WS (partials + VAD)       │
   │  L2 Dialogue     → flow graph + Groq planner            │
   │  L3 Tools        → typed function-calling registry      │
   │  L4 Memory       → Redis working set + pgvector recall  │
   │  L5 Guardrails   → pre/post filters, escalation, safety │
   │  L6 Speech out   → Sarvam TTS WS, streamed+interruptible│
   └───────────┬─────────────────────────────────────────────┘
               ▼
   Postgres (+pgvector) · Redis · S3 (recordings) · OpenTelemetry
```

---

## 3. The brain — layered design (this is the differentiator)

Most voice bots feel bad because the LLM decides *when* to talk. Do not do that.

### L0 — Turn-taking controller (deterministic, ~0 ms)
Owns the conversational floor. A tiny state machine, pure code, no model:

`IDLE → LISTENING → THINKING → SPEAKING → (interrupted) → LISTENING`

- Uses Sarvam STT with `vad_signals=true`. `START_SPEECH` while `SPEAKING` = **barge-in**.
- Barge-in must do all three, in this order, inside ~150 ms:
  1. stop local playback and flush the outbound audio buffer
  2. send Exotel a `clear` event (drops audio already queued in their jitter buffer)
  3. close the current Sarvam TTS socket (there is no server-side cancel message), open a fresh one
- End-of-turn = VAD silence + STT finalisation. Tune per language and per vertical
  (`high_vad_sensitivity` for snappy turns; longer silence for elderly / hospital callers).
- On LiveKit, this layer is **configured, not built**: use its semantic turn-detector plugin plus
  Sarvam VAD signals, and let the framework handle playback cancellation. The state machine above
  is still yours — it decides escalation, timeouts and node transitions.
- **Filler audio**: if the planner takes >600 ms, play a pre-synthesised "mm-hmm" / "ek second"
  cached clip. This one trick removes most of the "is it dead?" feeling.

### L1 — Perception
Sarvam streaming STT over WebSocket. `saaras:v3-realtime` for true partials and millisecond VAD
tuning; `saarika:v2.5` / `saaras:v2.5` otherwise. Streaming only — **never** the batch HTTP endpoint
during a live call. Formats: WAV / raw PCM (`pcm_s16le`) only.
`saaras` auto-detects language and can translate — that is how you handle Hinglish and code-switching.

Emits to L2: `partial_transcript`, `final_transcript`, `detected_language`, `speech_start`,
`speech_end`, plus non-speech signals — DTMF digits from Exotel, silence timers, call metadata.

### L2 — Dialogue policy (hybrid, not pure LLM)
A **flow graph** per Agent: `greet → identify → intent → task → confirm → close`.
Each node declares allowed tools, required slots, exit conditions, max turns, fallback.
Inside a node Groq reasons freely; the graph keeps control.

- Model: `openai/gpt-oss-120b` on Groq (measured TTFT ~510-560 ms). The Llama models in Groq
  tutorials are not available on every account — check `models.list()` first.
- Stream tokens and **start TTS at the first sentence boundary**, not at completion.
- Run a small fast model (`openai/gpt-oss-20b`, ~320-530 ms) for cheap jobs: intent
  classification, PII redaction, "did the user finish?" scoring, safety triage.
- Enable Groq prompt caching on the static system prompt (50% cheaper input).
- Hard rules live in code, never in the prompt: max call duration, transfer conditions,
  business hours, retry counts.

### L3 — Tools / actions
A typed per-tenant registry, exposed to Groq as function schemas: `book_appointment`,
`check_slot`, `lookup_patient`, `send_sms`, `transfer_to_human`, `log_grievance`,
`search_kb`, `end_call`. Rules:
- every tool has a timeout (≤1.5 s) and a spoken fallback line
- side-effecting tools (booking, payment, SMS) require an explicit spoken confirmation turn
- tool latency is always masked by an L0 filler phrase

### L4 — Memory
- **Working** — last N turns + slot store in Redis, keyed by `session_id`
- **Episodic** — full transcript + outcome written to Postgres at call end
- **Semantic** — tenant KB chunked into pgvector; retrieve *before* the LLM call, triggered by the
  fast model, top-k = 4, hard token cap. Never dump the whole KB into context.
- **Caller profile** — phone number → past calls, so the agent can say "welcome back"

### L5 — Guardrails
- Domain allowlist per vertical. Hospital agent: **no diagnosis, no dosage, no triage advice** —
  hard refusal + offer transfer. That is a legal boundary, not a style choice.
- Political agent: mandatory automated-call disclosure in the first seconds, honour opt-out
  immediately, no claims outside the tenant-supplied fact sheet.
- Prompt-injection defence: KB content and caller speech are **data, never instructions**.
- Escalation triggers: two consecutive no-match, explicit "talk to a human", anger detection,
  any medical/legal emergency keyword → Exotel transfer to a live agent.

### L6 — Speech out
Sarvam TTS WebSocket — `bulbul:v3` (24 kHz, 20+ voices) or `bulbul:v2` if you need pitch/loudness
control. One socket per utterance; push text chunk-by-chunk as the LLM streams.
Telephony output `mulaw` / `linear16` at 8 kHz; web at 22.05/24 kHz.
Pre-cache TTS for fixed phrases (greeting, hold, disclaimer, prompts) — free latency.

---

## 4. Latency budget (target: <1000 ms from user silence to first audio byte)

| Stage | Budget |
|---|---|
| VAD end-of-speech detection | 150–250 ms |
| STT finalisation | 100–200 ms |
| RAG retrieval (parallel with prompt build) | ≤80 ms |
| Groq TTFT | 120–250 ms |
| First sentence assembled | 100–150 ms |
| Sarvam TTS first chunk | 150–250 ms |
| Network + jitter buffer | 50–100 ms |
| **Total** | **~750–1100 ms** |

Rules that protect the budget: host everything in **ap-south-1 (Mumbai)**; keep STT/TTS/LLM sockets
warm and pooled; never block the turn loop on a DB write (queue it); overlap retrieval, generation
and synthesis instead of running them in sequence.

---

## 5. Telephony integration (Exotel)

**Primary path — Virtual SIP Trunking into LiveKit** (see §7a):
1. Exotel account on the **Mumbai instance**, KYC complete, vSIP trunk provisioned, Exophone bought.
2. LiveKit console → Telephony → Configuration → copy the SIP FQDN.
3. Point the Exotel inbound trunk at that FQDN. Authenticate by static-IP allowlist (`/32` entries
   only — Exotel does not accept CIDR ranges) or by digest auth.
4. Create a LiveKit **dispatch rule** mapping the inbound trunk to the right agent room, so the
   called Exophone selects the tenant's Agent.
5. Outbound: LiveKit SIP API creates a SIP participant on the Exotel outbound trunk.

With this path you never touch audio framing — the SIP bridge handles it.

**Fallback path — AgentStream VoiceBot Applet (WebSocket)**, if vSIP is not yet provisioned:

- Account on the **Mumbai instance**, KYC complete, AgentStream enabled (IP–PSTN intermixing is
  Mumbai-only). Buy an Exophone, build the Call Flow in App Bazaar.
- Flow: `Greeting (optional) → VoiceBot Applet → Passthru (logging) → fallback applet`.
- Point the VoiceBot applet at an **https** endpoint that returns a fresh `wss` URL per call.
  That lets you shard sessions and pin a caller to a warm worker. Do not hardcode one wss URL.
- Protocol — Exotel → you: `connected`, `start`, `media` (base64 PCM), `dtmf`, `stop`.
  You → Exotel: `media`, `clear`, `mark`.
- **Audio framing: 8 kHz, 16-bit, mono. 100 ms = 3200 bytes. Every chunk must be a multiple of
  320 bytes.** Wrong framing is the number-one cause of choppy/robotic audio. Keep messages well
  under 100 KB.
- Ending a call: just close the WebSocket — Exotel emits `stop` internally and advances the flow.
  Put a terminating applet after the VoiceBot applet.
- Outbound: Exotel Connect / Campaign API into the same flow. TRAI rules apply — DND scrubbing,
  calling-window limits, identity disclosure, per-call opt-out.

---

## 6. Web interface (Phase 1 — build this first)

- **Frontend**: Next.js (App Router) + TypeScript + Tailwind. Auth via email OTP or Google → JWT.
- **Voice**: LiveKit JS SDK (`@livekit/components-react`). The browser joins a LiveKit room; the
  gateway mints the room token after validating the user's JWT. WebRTC gives you acoustic echo
  cancellation, noise suppression, jitter handling and reconnection for free — do not rebuild those
  on a raw WebSocket. Without AEC the mic hears your own TTS and barge-in misfires constantly.
- **Playback and interruption** are handled inside the room by the agent, not by hand-rolled buffers.
- **Dashboard**: Agent builder (persona, languages, voice, flow nodes, KB upload, tools),
  Live calls, Transcript + audio player, Analytics (containment rate, p95 latency, escalation rate),
  Numbers & campaigns, API keys, Usage/billing.
- Same session protocol as telephony, so the mobile app later is trivial.

---

## 7. Backend & data

- **Language**: Python 3.12 + FastAPI + uvloop. Async throughout; one asyncio task group per session.
- **Framework: LiveKit Agents (Python).** See §7a for why, and for the Pipecat alternative.
  Do not hand-roll the media pipeline.
- **Postgres 16 + pgvector**: tenants, users, agents, flows, kb_documents, kb_chunks, calls, turns,
  tool_invocations, phone_numbers, campaigns, api_keys, usage_events. Every table carries
  `tenant_id`, enforced with Row Level Security.
- **Redis**: live session state, rate limits, pub/sub for the dashboard's live view.
- **Object storage**: recordings + transcripts, lifecycle-expired per tenant retention policy.
- **Observability**: an OpenTelemetry span per turn with sub-spans (vad / stt / llm / tts) so latency
  regressions are visible per tenant and per language. Log every prompt and response, redacted.

---

## 7a. Framework choice: LiveKit vs Pipecat

**Decision: LiveKit Agents (Python), with Exotel Virtual SIP Trunking (vSIP) for PSTN.**

Exotel publishes an official LiveKit integration — trunk setup, dispatch rules, reference
architecture, and a sample repo (`github.com/exotel/agentstream-voiceaiecosystem`). This is a
supported path, not a hack. Exotel holds the Indian Unified Licence (VNO) that Twilio and other
global providers cannot offer, so it stays the carrier either way.

### Why LiveKit wins for this product

| | LiveKit Agents | Pipecat |
|---|---|---|
| Web browser surface | **WebRTC: echo cancellation, jitter buffer, packet-loss concealment, auto-reconnect — free** | Raw WS/PCM or a Daily dependency; you own the audio problems |
| Mobile (Phase 5) | **iOS / Android / RN / Flutter SDKs, same room protocol** | Build it yourself |
| Exotel PSTN | **vSIP trunk → LiveKit SIP bridge (officially documented)** | AgentStream WebSocket applet (also documented) |
| Audio framing | **Handled by the SIP bridge** | You own 8 kHz / 320-byte chunking |
| Turn detection | **Ships a semantic end-of-turn model + interruption handling** | VAD-based, more manual |
| Sarvam + Groq | Plugins for both | Services for both |
| Scale / ops | LiveKit Cloud dispatch, or self-host SFU | You build worker orchestration |

The decisive point is **acoustic echo cancellation on the web**. Without WebRTC AEC, the browser
mic hears your own TTS through the laptop speakers and your barge-in detector fires constantly.
Solving that by hand is weeks of misery. LiveKit gives it to you on day one — and the same room
abstraction becomes the mobile app later.

### The trade-offs (be honest about these)

- LiveKit Cloud is a paid dependency. Self-hosting means running an SFU + Redis + the SIP service.
- vSIP onboarding is more telecom work than pasting a VoiceBot applet URL: trunk provisioning,
  static-IP allowlisting (`/32` entries only — no CIDR ranges) or digest auth, dispatch rules.
- LiveKit is opinionated. Your flow-graph brain still lives in your own code; LiveKit only owns
  transport, turn detection and the agent worker lifecycle.

### Required LiveKit configuration for India
- Deploy agents in **LiveKit Cloud Mumbai (ap-south)**.
- **Ask LiveKit to enable region pinning** so inbound and outbound calls stay in-region. Without
  this, calls can hairpin through another region and you silently lose 200–400 ms.
- Inbound: Exotel PSTN → LiveKit SIP FQDN (`sip:xxxx.sip.livekit.cloud`) → room → agent joins.
- Outbound: dispatch an agent with the destination number in metadata → LiveKit SIP API creates a
  SIP participant on the Exotel outbound trunk.

### When to pick Pipecat instead
Choose Pipecat only if Exotel vSIP provisioning is blocked or slow for your account and you need
a demo on a real phone number this week — the AgentStream WebSocket applet needs no trunk, just a
URL. If you go that route, §5 applies in full (including the 320-byte rule). **Do not run both
frameworks.**

### NVIDIA NIM as a free development resource

Verified working alongside this stack (see [BRAIN.md](BRAIN.md) for measurements):

- **Nemotron text models** — the brain runs on Groq *or* NVIDIA by config, with the unused one as
  automatic failover. NVIDIA is free and fine for development; its latency spread (up to 9 s on a
  shared endpoint) rules it out for live calls.
- **Maxine Studio Voice** — speech enhancement placed *before* STT. Worth adopting for noisy Indian
  call conditions. gRPC over NVCF, 16/48 kHz only, adds a hop; self-host the NIM for production.
- **Nemotron VoiceChat 11B** — full-duplex speech-to-speech, ~448 ms turn-taking. Deliberately not
  adopted: it owns turn-taking and dialogue policy, which is precisely the control this
  architecture keeps in code. Evaluation-only licence and English-centric besides. Keep as a
  benchmark.

### Options deliberately rejected
- **Vapi / Retell / Bland** — managed voice-agent platforms that own the orchestration layer.
  Your brain is the product; renting theirs removes the differentiator, adds a latency hop, and
  their Indian-language and Indian-PSTN support is weak.
- **Twilio ConversationRelay** — no Indian PSTN licence. Non-starter here.
- **Plivo** — a legitimate second Indian carrier, also LiveKit-integrated. Worth adding later
  purely as carrier redundancy, not as the primary.

---

### Repo layout
```
apps/web            Next.js dashboard + LiveKit JS SDK voice widget
services/gateway    FastAPI: auth, CRUD, LiveKit token issuer, Exotel + LiveKit webhooks
services/voice      LiveKit Agent workers (the brain)
packages/core       flow engine, tool registry, prompts, guardrails (shared)
infra               docker-compose (dev), terraform (prod), migrations
```

### Environment variables
```
SARVAM_API_KEY, GROQ_API_KEY
LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_SIP_TRUNK_ID
EXOTEL_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_SUBDOMAIN, EXOPHONE
DATABASE_URL, REDIS_URL, S3_*, JWT_SECRET
```
Never in code, never committed. Validate presence at boot and fail fast.

---

## 8. Multi-tenancy, security, India compliance

- Tenant isolation at the DB row level + per-tenant API keys + per-tenant rate limits.
- RBAC: owner / admin / operator / viewer.
- PII: redact phone / Aadhaar / card patterns (regex + fast model) before storing a transcript or
  sending it to the LLM. Encrypt at rest. Configurable retention, default 90 days.
- **DPDP Act 2023**: consent notice at call start, purpose limitation, deletion on request, breach
  notification. Keep data in India — Mumbai region, and Sarvam is an Indian provider.
- Hospitals: treat as sensitive health data, sign a DPA, audit-log every access.
- Recording consent must be spoken and logged per call.

---

## 9. What NOT to do

1. **Don't let the LLM own turn-taking, interruption, or timing.** Code owns the floor.
2. **Don't use batch/HTTP STT or TTS in a live call.** WebSockets only.
3. **Don't skip the `clear` event on barge-in** — the caller will hear the bot talk over itself.
4. **Don't violate Exotel's 320-byte chunk rule** on the AgentStream fallback path (the vSIP path
   removes this hazard entirely). Don't send oversized messages.
5. **Don't build one giant system prompt.** Per-node prompts + retrieved context, static part cached.
6. **Don't stuff the whole knowledge base into context** — latency and hallucination surface.
7. **Don't let the hospital agent give medical advice.** Refuse and transfer. Same for legal/financial.
8. **Don't run political outbound calls without DND scrubbing, disclosure, and opt-out.**
9. **Don't start with the mobile app, a self-hosted SFU, or Kubernetes.** LiveKit Cloud first.
10. **Don't hand-roll the media pipeline**, and don't run LiveKit and Pipecat side by side.
10a. **Don't forget to request LiveKit region pinning for India** — without it calls hairpin out of
    region and you lose 200–400 ms you will never find in profiling.
11. **Don't host outside India** — every 100 ms of extra RTT is audible in conversation.
12. **Don't use one model for everything.** Big model reasons; tiny fast model classifies.
13. **Don't store raw audio by default.** Opt-in per tenant.
14. **Don't judge success on "it worked once".** Track containment rate, barge-in correctness,
    p95 turn latency, escalation rate.

---

## 10. Phased plan

**Phase 0 — Accounts & spikes (week 1)**
Sarvam key, Groq key, Exotel account + KYC + Exophone. Three throwaway scripts: mic → Sarvam STT WS,
text → Sarvam TTS WS, Groq streaming with a tool call. Measure real latency from a Mumbai VM.

**Phase 1 — Web voice MVP (weeks 2–4)**
Next.js login → LiveKit room → LiveKit Agent worker (Sarvam STT → Groq → Sarvam TTS).
Barge-in working. One hardcoded persona. This is the demo that proves the brain.

**Phase 2 — Agent platform (weeks 5–7)**
Postgres schema, tenants/auth, agent builder UI, flow nodes, KB upload + pgvector RAG, tool registry
with three real tools, transcripts and analytics.

**Phase 3 — Telephony (weeks 8–10)**
Exotel vSIP trunk → LiveKit SIP FQDN, dispatch rules per Exophone, region pinning, DTMF,
transfer-to-human, inbound then outbound campaigns, TRAI compliance layer.

**Phase 4 — Hardening & scale (weeks 11–14)**
Load test to 100 concurrent calls, autoscaling workers, OTel dashboards, DPDP/retention tooling,
per-tenant billing on `usage_events`, on-call runbook.

**Phase 5 — Mobile app.** Thin client over the same WS protocol.

---

## 11. First three things to do

1. Create Sarvam + Groq + LiveKit Cloud accounts (LiveKit project in **Mumbai / ap-south**) and run
   the three latency spikes from a Mumbai VM.
2. Start Exotel KYC **and ask for vSIP trunking** — longest lead time, and it blocks Phase 3.
   In the same week, email LiveKit to request India region pinning.
3. `pip install "livekit-agents[sarvam,groq,silero,turn-detector]"` and get the Sarvam ↔ Groq ↔
   Sarvam loop talking in a browser room.

---

### Reference links
- Sarvam docs: https://docs.sarvam.ai — streaming STT: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/streaming-api — streaming TTS: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/streaming-api/web-socket
- Exotel VoiceBot applet: https://docs.exotel.com/exotel-agentstream/voicebot-applet — bidirectional streaming: https://docs.exotel.com/exotel-agentstream/bidirectional-streaming — dev guide: https://developer.exotel.com/docs/agentstream/developer-guide
- Exotel × LiveKit SIP trunking: https://docs.exotel.com/dynamic-sip-trunking/connect-exotel-sip-trunk-to-livekit — dev guide: https://developer.exotel.com/docs/agentstream/livekit-integration — reference architecture: https://exotel.com/blog/production-architecture-low-latency-voice-ai-exotel-livekit/ — sample repo: https://github.com/exotel/agentstream-voiceaiecosystem
- LiveKit voice agents in India: https://livekit.com/blog/building-performant-voice-agents-india — Sarvam plugin: https://github.com/livekit/agents-js/blob/main/plugins/sarvam/README.md
- Pipecat (fallback path): Exotel transport https://docs.pipecat.ai/deployment/pipecat-cloud/guides/telephony/exotel-websocket — Sarvam STT https://docs.pipecat.ai/api-reference/server/services/stt/sarvam
