"""`looks_like_narration`: the sentence of intent that is not an answer."""

from __future__ import annotations

import json
from pathlib import Path

from lakehouse_agent.agent.narration import draft_hint, looks_like_narration

MEASURED = "En düşük ve en yüksek aylardaki faiz düzeylerini tablodan çekiyorum."


def test_the_measured_non_answer_is_caught():
    assert looks_like_narration(MEASURED)
    assert looks_like_narration("Şimdi payı hesaplıyorum ve grafiği çiziyorum.\n\n")
    assert looks_like_narration("İpotekli konut satışları serilerini arıyorum.")


def test_answers_are_not_narration():
    assert not looks_like_narration("")
    assert not looks_like_narration("Evet, veri var; tablo ekranda.")
    assert not looks_like_narration(
        "Bu veriyle gösterilemiyor; başka bir seri öneriyorum."
    )
    assert not looks_like_narration(
        "Faiz 2023-07'de %32,2'ye çıktı; kırılma orada."
    )  # a number
    assert not looks_like_narration("Model bu tur icin bir cevap uretmedi.")
    long = (
        "Payın en düşük olduğu ay aralık, en yüksek olduğu ay haziran; " * 6
        + "bakıyorum."
    )
    assert not looks_like_narration(long)  # too long to be a plan line


def test_none_of_the_shipped_eval_answers_is_narration():
    """The thirteen real answers of 23 Sep are the negative set."""
    path = (
        Path(__file__).resolve().parents[1] / "eval/results/answer_eval_2026-09-23.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    flagged = [
        q["id"] for q in report["sorular"] if looks_like_narration(q.get("cevap") or "")
    ]
    assert flagged == []


def test_the_draft_is_the_last_long_assistant_note_and_nothing_else():
    draft = (
        "Veriler hazır. " + "Payın en düşük olduğu ay 2023-12, en yüksek 2022-06. " * 3
    )
    messages = [
        {"role": "user", "content": "soru"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "sonuc"},
        {"role": "assistant", "content": draft},
    ]
    assert draft_hint(messages) == draft.strip()

    assert draft_hint(messages[:-1]) is None  # ends on a tool result: no note
    assert (
        draft_hint([*messages[:-1], {"role": "assistant", "content": "Veriler hazır."}])
        is None
    )
    assert (
        draft_hint([*messages[:-1], {"role": "assistant", "content": MEASURED * 3}])
        is None
    )
    assert draft_hint([]) is None
