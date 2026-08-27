"""What the console shows about a turn, beside what was said.

The point of the console is not the conversation — `talk.py` already holds one. It is
everything the flow decided while holding it: which intent was read, which values were
captured, which step the call moved to, which tool answered and how long each stage took.
Those come out of the brain as events on every turn and are otherwise invisible, and they
are what tells you whether a wrong answer came from mishearing, misclassifying, or a tool.

Rendering is plain HTML on purpose. A dataframe would need a schema for facts that are
genuinely of different kinds, and would lose the one thing worth having here: a slow
stage should look slow at a glance, without reading the number.
"""

from __future__ import annotations

from html import escape

# Above this, a stage is worth noticing. Voice turns live or die on the first second:
# 800 ms of planning is a pause the caller hears as the line going dead.
SLOW_MS = 800.0

# Widest bar in the stage breakdown, in pixels. Bars are relative to the slowest stage of
# the turn, not to a fixed ceiling, so the shape of a fast turn is still readable.
BAR_MAX = 120


def _row(key: str, value: str, tone: str = "") -> str:
    return (
        f'<div class="row"><span class="k">{escape(key)}</span>'
        f'<span class="v {tone}">{value}</span></div>'
    )


def _slots(slots: dict[str, str]) -> str:
    if not slots:
        return '<span class="empty">nothing captured yet</span>'
    chips = "".join(
        f'<span class="chip"><b>{escape(k.replace("_", " "))}</b>{escape(str(v))}</span>'
        for k, v in slots.items()
    )
    return f'<div class="vd-slots">{chips}</div>'


def _stages(stage_ms: dict[str, float]) -> str:
    # The total is the sum of the parts and is already on its own row above. Left in, it
    # is always the longest bar and every stage that actually matters is squashed against
    # the left margin — the breakdown stops showing a breakdown.
    stage_ms = {k: v for k, v in stage_ms.items() if k not in ("turn_total", "total")}
    if not stage_ms:
        return '<span class="empty">no timings this turn</span>'
    slowest = max(stage_ms.values()) or 1.0
    steps = []
    for name, ms in sorted(stage_ms.items(), key=lambda kv: -kv[1]):
        width = max(2, int(BAR_MAX * ms / slowest))
        slow = " slow" if ms >= SLOW_MS else ""
        steps.append(
            f'<div class="step{slow}"><span class="bar" style="width:{width}px"></span>'
            f'<span class="ms">{ms:.0f} ms</span>'
            f'<span>{escape(name.replace("_", " "))}</span></div>'
        )
    return f'<div class="vd-steps">{"".join(steps)}</div>'


def render(turn: dict) -> str:
    """One turn's decisions, as the console's right-hand panel."""
    if not turn:
        return (
            '<div class="vd-trace"><span class="empty">No turn yet. '
            "Speak or type to start the call.</span></div>"
        )

    intent = turn.get("intent") or "—"
    confidence = turn.get("confidence")
    # The confidence gate is the flow's, not a display choice: below it a transition does
    # not fire, so a turn can be classified correctly and still go nowhere. Colouring it
    # is what makes that visible instead of puzzling.
    tone = "live" if (confidence or 0) >= 0.55 else "warn"
    shown = f"{intent} · {confidence:.2f}" if confidence is not None else intent

    rows = [
        _row("heard as", escape(str(turn.get("heard") or "—"))),
        _row("intent", escape(shown), tone),
        _row("step", escape(str(turn.get("node") or "—"))),
    ]
    if turn.get("moved"):
        rows.append(_row("moved", escape(turn["moved"]), "live"))
    for tool in turn.get("tools", []):
        state = "live" if tool.get("ok") else "alert"
        rows.append(
            _row(f"tool · {tool['name']}", f"{tool.get('ms', 0):.0f} ms", state)
        )
    if turn.get("total_ms"):
        rows.append(
            _row(
                "turn total",
                f"{turn['total_ms']:.0f} ms",
                "warn" if turn["total_ms"] >= 2 * SLOW_MS else "",
            )
        )

    blocks = [
        f'<div class="vd-trace">{"".join(rows)}</div>',
        '<p class="vd-label" style="margin-top:18px">Captured</p>',
        _slots(turn.get("slots") or {}),
        '<p class="vd-label" style="margin-top:18px">Where the time went</p>',
        _stages(turn.get("stage_ms") or {}),
    ]
    if turn.get("note"):
        blocks.append(f'<p class="vd-note alert">{escape(turn["note"])}</p>')
    return "".join(blocks)
