"""The console's shape: what is on screen and what is wired to what.

Two columns, unequal on purpose. The left is the call — the transcript, the microphone,
the text box — and it is where your attention lives. The right is the flow's reasoning,
which you glance at when something goes wrong and ignore when it does not.

Kept apart from `console.py` so the wiring can be read in one screen without the turn
loop, and the turn loop without the wiring.
"""

from __future__ import annotations

import gradio as gr

from brain.agents.hospital import HOSPITAL_AGENT
from ui import player, trace
from ui.console import FFMPEG_READY, STYLES, Console

LANGUAGE_NAMES = {
    "en-IN": "English",
    "hi-IN": "हिन्दी  Hindi",
    "te-IN": "తెలుగు  Telugu",
}

MASTHEAD = """
<div class="vd-head">
  <h1>Voice Desk</h1>
  <span class="vd-sub">{agent} · appointment desk</span>
  <span class="vd-spacer"></span>
  <span class="vd-pill live">voice to voice</span>
  <span class="vd-pill">bulbul&nbsp;v3 · ishita</span>
  <span class="vd-pill">saaras&nbsp;v3</span>
  <span class="vd-pill">{audio}</span>
</div>
"""


def build() -> gr.Blocks:
    console = Console()

    with gr.Blocks(title="Voice Desk · agent console", fill_height=True) as app:
        gr.HTML(
            MASTHEAD.format(
                agent=HOSPITAL_AGENT.name,
                # Said out loud on the masthead rather than left to be discovered: the
                # difference between the two modes is audible, and someone judging the
                # agent's latency needs to know which one they are hearing.
                audio="ffmpeg ready" if FFMPEG_READY else "no ffmpeg",
            )
        )

        session_id = gr.State("")

        with gr.Row(equal_height=False):
            # --- the call -------------------------------------------------
            with gr.Column(scale=3):
                transcript = gr.Chatbot(
                    label=None,
                    show_label=False,
                    height=470,
                    elem_classes="vd-panel",
                    placeholder="Pick a language and press Start call.",
                    avatar_images=(None, None),
                )

                # Our own player, not gr.Audio. See ui/player.py: the component plays
                # the first clip of a page and silently ignores every one after it.
                reply_audio = gr.HTML(value=player.SILENT, elem_classes="vd-voice")

                with gr.Row():
                    language = gr.Dropdown(
                        choices=[(name, tag) for tag, name in LANGUAGE_NAMES.items()
                                 if tag in HOSPITAL_AGENT.languages],
                        value=HOSPITAL_AGENT.languages[0],
                        label="Language",
                        scale=1,
                    )
                    start = gr.Button("Start call", variant="primary", scale=1)
                    hang_up = gr.Button("Hang up", scale=1)

                with gr.Row():
                    # A recorded clip rather than a live stream: the browser's own
                    # silence detection is what decides a turn is over, which keeps the
                    # console honest about what the ear receives on a call.
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Speak",
                        show_label=False,
                        scale=3,
                        elem_classes="vd-panel",
                    )
                    with gr.Column(scale=2):
                        typed = gr.Textbox(
                            placeholder="…or type a turn",
                            show_label=False,
                            lines=1,
                            submit_btn=True,
                        )
                        # A button as well as the enter key. Enter alone is fine for a
                        # person and useless for anything driving the console from
                        # outside it, which is how these get regression-tested.
                        send = gr.Button("Send turn", variant="primary")

            # --- what the flow decided ------------------------------------
            with gr.Column(scale=2):
                gr.HTML('<p class="vd-label">This turn</p>')
                panel = gr.HTML(
                    value=trace.render({}), elem_classes="vd-panel", padding=True
                )

        # --- wiring ------------------------------------------------------

        turn_outputs = [transcript, panel, typed, reply_audio]

        start.click(
            console.open_call,
            inputs=[language],
            outputs=[session_id, transcript, panel, reply_audio],
        )

        def hang_up_call():
            return "", [], trace.render({}), player.SILENT

        hang_up.click(
            hang_up_call, outputs=[session_id, transcript, panel, reply_audio]
        )

        for trigger in (typed.submit, send.click):
            trigger(
                console.take_turn,
                inputs=[session_id, language, typed, transcript],
                outputs=turn_outputs,
            )

        # Recording stops, the clip is transcribed, and the transcript becomes the turn.
        # Two steps rather than one so a misheard word is visible in the text box before
        # it becomes an answer — which is most of what you are here to check.
        mic.stop_recording(
            console.heard, inputs=[mic, language], outputs=[typed]
        ).then(
            console.take_turn,
            inputs=[session_id, language, typed, transcript],
            outputs=turn_outputs,
        ).then(
            lambda: None, outputs=[mic]
        )

    return app


def launch(**kwargs) -> None:
    # Gradio 6 moved theme and css off Blocks() and onto launch().
    build().launch(css=STYLES, theme=gr.themes.Base(), **kwargs)
