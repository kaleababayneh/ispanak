"""Is this text an answer, or the model saying what it is about to do?

Measured 23 Sep on the browser demo: the answer step returned "En düşük ve
en yüksek aylardaki faiz düzeylerini tablodan çekiyorum." -- a sentence of
intent, in the voice of the planning steps -- while the real answer sat in
the last observe note, which the answer call deliberately never sees
(`loop._answer_messages`). The verifier passed it: no numbers, nothing to
check. The empty-answer retry did not fire: the text was not empty.

The check is narrow on purpose. A real answer that happens to contain one
of these verbs also contains a number, a date, or is longer than a sentence
or two; measured against the thirteen eval answers of 23 Sep, none matched.
"""

from __future__ import annotations

import re

#: First-person present-continuous verbs of *doing*, the vocabulary of a plan
#: step. "öneriyorum" or "düşünüyorum" are left out: an answer can say those.
_DOING = re.compile(
    r"\b(çek|ar|hesapl|bak|bul|getir|çiz|işaretl|kur|topl|den|başl|al|incel|ok|"
    r"karşılaştır|çevir|ekl|hazırl|listel|say|ölç|test ed|kontrol ed)"
    r"[ıiuü]yorum\b"
)

#: Longer than this and it is telling the person something, whatever verbs
#: it uses. The measured non-answer was 70 characters.
MAX_CHARS = 300


def draft_hint(messages: list[dict], *, min_chars: int = 120) -> str | None:
    """The answer the model may already have written in its last note.

    `loop._answer_messages` drops trailing assistant notes on purpose (a
    request ending on the model's own words makes some templates continue
    them). Measured 23 Sep: the dropped note *was* the answer, five
    sentences of it, and the answer call then produced one line of intent.
    On a narration retry that note goes into the system prompt as a draft
    to rewrite -- never as a trailing assistant message.
    """
    for message in reversed(messages):
        if message.get("role") != "assistant":
            break
        if message.get("tool_calls"):
            break
        text = str(message.get("content") or "").strip()
        if len(text) >= min_chars and not looks_like_narration(text):
            return text
    return None


def looks_like_narration(text: str) -> bool:
    """True for "şimdi şunu yapıyorum" written where an answer should be."""
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_CHARS:
        return False
    if re.search(r"\d", stripped):
        return False
    return _DOING.search(stripped) is not None
