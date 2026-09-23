"""What the model is shown of the conversation so far.

`SessionState.messages` keeps every message of the session -- the step log
and the traceability panel read it -- but the model does not need all of it.
Measured 22 Sep 2026 on the scenario's three questions: the third turn's five
model calls read 59.5k prompt tokens between them, most of it the first two
turns' search candidates, table previews and intermediate notes. Every token
of that is latency on a slow MIA day, and latency is what timed turns out.

So earlier turns are shown as what they amount to: the question and the final
answer. Their tables are not lost -- the system prompt lists every table and
its columns (`build_tables_summary`), which is what a follow-up such as "add a
column without breaking the table" actually needs. The current turn is shown
in full, because the model is still working through it.
"""

from __future__ import annotations

from typing import Any

from lakehouse_agent.agent.state import SessionState

#: Stands in for the answer of an earlier turn that produced none, so the
#: conversation does not read as two questions in a row with nothing between.
UNANSWERED = "(Bu soru cevaplanamadi; o turda kurulan tablolar oturumda duruyor.)"


def model_messages(state: SessionState) -> list[dict[str, Any]]:
    """The conversation as the model sees it: earlier turns reduced to their
    question and final answer, the current turn in full.

    Only whole turns are dropped, so an assistant message asking for a tool
    is never left without the tool message that answers it -- that would make
    the request a protocol error. The converse is guarded too: a tool message
    whose call is not in the view is left out, since a reply to a call the
    request does not contain is the same error from the other side (the loop
    repairs a dead turn's calls inside that turn, so this is a second line).
    `state.messages` is not modified.
    """
    messages = state.messages
    starts = [s for s in state.turn_starts if 0 <= s < len(messages)]
    if len(starts) <= 1:
        return _without_stray_replies(list(messages))

    answers = set(state.answer_indices)
    view: list[dict[str, Any]] = list(messages[: starts[0]])
    for begin, end in zip(starts, starts[1:], strict=False):
        view.append(messages[begin])
        answered = [i for i in range(begin, end) if i in answers]
        if answered:
            final = messages[answered[-1]]
            view.append({"role": "assistant", "content": final.get("content") or ""})
        else:
            view.append({"role": "assistant", "content": UNANSWERED})
    view.extend(messages[starts[-1] :])
    return _without_stray_replies(view)


def _without_stray_replies(view: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool messages that answer no call made earlier in `view`."""
    asked: set[str] = set()
    kept = []
    for message in view:
        for call in message.get("tool_calls") or []:
            if call.get("id"):
                asked.add(call["id"])
        if message.get("role") == "tool" and message.get("tool_call_id") not in asked:
            continue
        kept.append(message)
    return kept
