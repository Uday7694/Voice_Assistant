# Free voice APIs for Indian languages — options and verdict

> **Decision (2026-08-22): Sarvam only for STT and TTS.** The survey below is kept as the
> record of what was compared and why. Groq Whisper remains a verified free fallback for
> STT if Sarvam credits run out mid-development.

API-only, no self-hosting, no GPU. Target languages: **Hindi, Tamil, Telugu, Kannada, English**,
with broad Indian coverage beyond that.

First, the distinction that matters most:

- **Free forever** — a recurring monthly quota that resets. You can build on it indefinitely.
- **Free credits** — a one-time grant. Excellent for development, runs out.

---

## Verified on your existing keys

### Groq Whisper — free STT, working right now ✅

Tested against `api.groq.com` with the key already in `.env`. All returned native script:

| language | latency | result |
|---|---|---|
| Hindi `hi` | 1641 ms | `प्राप्ति` |
| Tamil `ta` | 240 ms | `சரி.` |
| Telugu `te` | 266 ms | `మిల్లిక్కులు` |
| Kannada `kn` | 1058 ms | `ಮಾಗಾಯಿದಿದಿದಿ` |

Also accepted: `ml, mr, bn, gu, pa, ur, en`. Models: `whisper-large-v3`,
`whisper-large-v3-turbo`. (The text above is hallucinated — the probe sent a 440 Hz tone, not
speech. It proves the endpoint, the language parameter, and script output, not accuracy.)

```python
client.audio.transcriptions.create(
    model="whisper-large-v3", file=("a.wav", audio_bytes, "audio/wav"), language="hi"
)
```

**Two real limitations.** It is a **batch/file** endpoint, not a streaming WebSocket — for live
conversation you must chunk audio yourself with local VAD, which costs latency and loses partial
transcripts. And Whisper's Telugu and Kannada accuracy is materially worse than Indic-specialist
models; Hindi is decent, Tamil acceptable.

### Groq TTS — not usable ❌

The only speech-out models on the account are `canopylabs/orpheus-v1-english` and
`orpheus-arabic-saudi`. **No Indian language TTS on Groq.** This is the gap you have to fill
elsewhere.

### NVIDIA — partial

`build.nvidia.com` hosts Riva/Nemotron speech free, but for Indian languages the coverage is thin:
**Magpie TTS Multilingual added Hindi (hi-IN)** and that is essentially it — no Tamil, Telugu or
Kannada. Parakeet multilingual ASR covers 25 languages, weighted towards European ones. Everything
is **gRPC** (`pip install nvidia-riva-client`), not REST.

---

## The strongest free options

### 1. Bhashini — Government of India ⭐ best truly-free choice

MeitY's national language platform, built with the IITs. **Free for low volume**, REST APIs, and
the broadest Indian coverage anywhere: **22 languages** including all four of yours, plus Bengali,
Marathi, Gujarati, Punjabi, Malayalam, Odia, Assamese, Urdu and more.

Does **ASR, TTS and translation**, and can chain them as one `ASR → NMT → TTS` pipeline.

Getting started: register on the **ULCA portal**, verify email, generate keys from the profile
section — you need `userId`, `ulcaApiKey` and `InferenceApiKey`. Then a *Pipeline Search* call
returns the Pipeline ID and Service IDs for the task you want, which you pass to the compute call.

Trade-offs to expect: the two-step pipeline/service discovery is clumsy compared to a modern SDK,
latency and uptime are not commercial-grade, and "free for low volume" is not a published number.
Fine for development and low-traffic production; verify before scaling.

- Docs: https://bhashini.gitbook.io/bhashini-apis
- Service IDs: https://dibd-bhashini.gitbook.io/bhashini-apis/available-models-for-usage
- Python helper: https://github.com/dteklavya/bhashini_translator

### 2. Sarvam AI — ₹1,000 free credits, best Indian accuracy

Credits, not a recurring free tier — but they **never expire**, need no card, and cover every
Sarvam API (STT, TTS, translation, transliteration, chat).

Why it still wins on quality: Saaras v3 reports **19.31% WER on IndicVoices** across 10 languages,
ahead of Deepgram Nova-3, GPT-4o Transcribe and Gemini on *Indian* speech. Bulbul v3 TTS covers 11
Indian languages with 35+ voices and sub-250 ms streaming, and beat ElevenLabs and Cartesia in a
20,000-vote blind evaluation. It also has real **streaming WebSockets** with VAD signals, which
Groq Whisper does not.

This remains the right production choice for your stack. Use the free credits for development.

### 3. Google Gemini — free tier, no credit card

A genuine free tier (rate-limited: roughly 10–30 RPM, daily quotas, no card required). The **Live
API** is full speech-to-speech with 70+ languages including Hindi, Tamil and Telugu, plus
**barge-in, tool calling, and input/output transcripts** — and it auto-detects and switches
language mid-conversation without configuration.

For a free end-to-end voice demo this is the fastest path by a wide margin. Same architectural
caveat as Nemotron VoiceChat: a speech-to-speech model owns turn-taking and dialogue policy, so it
bypasses the flow graph and guardrails in `brain/`. Use it to prototype the *feel*, not as the
production brain.

### 4. ElevenLabs — small but genuinely recurring

**10,000 TTS characters/month free forever**, plus 4.5 h batch and 2.5 h realtime STT monthly.
Scribe v2 Realtime does ~150 ms first-partial across 90+ languages. Indian language quality is
moderate — behind Sarvam — but the monthly reset makes it useful as a permanent fallback.

### 5. One-time credits worth taking

- **Deepgram** — $200 free (~690 hours of Nova-3 batch). Strong realtime; limited Indian focus.
- **AssemblyAI** — $100 free, no credit card. Universal-3.5 Pro leads code-switching benchmarks
  (7.69 avg WER), which matters for Hinglish.

---

## Comparison

| Provider | Free model | hi | ta | te | kn | STT | TTS | Streaming | Protocol |
|---|---|---|---|---|---|---|---|---|---|
| **Bhashini** | free, low volume | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | limited | REST |
| **Sarvam** | ₹1,000 credits | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ WS | REST + WS |
| **Groq Whisper** | free tier | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | ❌ batch | REST |
| **Gemini Live** | free tier, no card | ✅ | ✅ | ✅ | ~ | ✅ | ✅ | ✅ WS | WS |
| **ElevenLabs** | 10k chars/mo | ~ | ~ | ~ | ~ | ✅ | ✅ | ✅ | REST + WS |
| **NVIDIA Riva** | free | TTS only | ❌ | ❌ | ❌ | ~ | hi only | ✅ | gRPC |
| **Deepgram** | $200 once | ~ | ~ | ~ | ~ | ✅ | ❌ | ✅ | REST + WS |
| **AssemblyAI** | $100 once | ~ | ~ | ~ | ~ | ✅ | ❌ | ✅ | REST + WS |

⚠️ = supported but weak accuracy · ~ = supported, not specialised · WS = WebSocket

---

## Recommended free development stack

Nothing here needs a card or a GPU:

```
STT   Groq Whisper        already working on your key, zero setup
TTS   Bhashini            free, covers all four languages + 18 more
LLM   Groq / NVIDIA       already wired into brain/providers.py
```

Then, when you want to hear how good it *can* sound, spend the Sarvam ₹1,000 on streaming
Saaras + Bulbul and compare. That is also the moment to measure whether Whisper's Telugu and
Kannada accuracy is good enough for your users, or whether Sarvam is doing real work for the money.

**Fastest possible demo, separate track:** Gemini Live free tier gives you a talking agent in
Hindi/Tamil/Telugu in an afternoon — useful for showing stakeholders, not for the production
architecture.

**Production, unchanged:** Sarvam for STT/TTS, Groq for the brain, Exotel for telephony.

---

## Sources

- [Sarvam STT](https://www.sarvam.ai/apis/speech-to-text) · [Sarvam TTS](https://www.sarvam.ai/apis/text-to-speech) · [free credits](https://aicredits.dev/submissions/180-sarvam-ai-1-000-free-api-credits-22-indian-languages-stt-tts-translation)
- [Bhashini API docs](https://bhashini.gitbook.io/bhashini-apis) · [models & service IDs](https://dibd-bhashini.gitbook.io/bhashini-apis/available-models-for-usage) · [bhashini.ai](https://www.bhashini.ai/)
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api) · [capabilities](https://ai.google.dev/gemini-api/docs/live-api/capabilities)
- [NVIDIA Magpie TTS Multilingual](https://build.nvidia.com/nvidia/magpie-tts-multilingual/api) · [Speech NIM release notes](https://docs.nvidia.com/nim/speech/latest/about/release-notes.html)
- [AssemblyAI: free STT comparison](https://www.assemblyai.com/blog/the-top-free-speech-to-text-apis-and-open-source-engines)
