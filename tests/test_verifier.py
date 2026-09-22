"""Tests for the answer verifier (agent/verifier.py, plan 5.3).

Plan 5.3's definition of done is "a deliberately made-up number is caught in a
test"; the other half matters as much for a demo: a correct answer must pass
clean, because every false flag is printed under it. So besides small cases,
the three answers the real model gave on 22 Sep are replayed against the same
tables they were written from. One number in them was wrong ("%27,85 puan" for
-0.2785) and one range was loose ("14-20 bin" for monthly rises of 16.6-20.2
bin): a range check cannot call the second one wrong, and does not.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from lakehouse_agent.agent.state import initial_state
from lakehouse_agent.agent.verifier import AnswerVerifier
from lakehouse_agent.contracts import Source

VERIFY = AnswerVerifier().verify
FIXTURE = Path(__file__).parent / "fixtures" / "live_answers_2026-09-22.json"


def _state(question="Soru?", tool_text="", sources=(), **columns):
    """A session with one monthly table, 2021-01..2025-12."""
    state = initial_state("s1")
    periods = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    frame = pd.DataFrame({"period": periods})
    for name, values in columns.items():
        frame[name] = list(values) + [float("nan")] * (len(periods) - len(values))
    state.tables["t01_get_series"] = frame
    state.turn_starts.append(0)
    state.messages += [
        {"role": "user", "content": question},
        {"role": "tool", "tool_call_id": "c1", "content": tool_text},
    ]
    state.sources = [
        Source(series_id=sid, fetched_at=datetime(2026, 9, 22, tzinfo=UTC))
        for sid in sources
    ]
    return state


def _flagged(result):
    return [flag.text for flag in result.flags]


# --- numbers ------------------------------------------------------------------


def test_numbers_taken_from_the_table_pass():
    state = _state(kredi=[276740.0, 678912.0])

    result = VERIFY("Bakiye 276.740'tan 678.912'ye cikti.", state)

    assert result.passed, _flagged(result)


def test_a_made_up_number_is_caught():
    """Plan 5.3's definition of done."""
    state = _state(kredi=[276740.0, 678912.0])

    result = VERIFY("Bakiye 512.345 milyon TL oldu.", state)

    assert not result.passed
    assert result.flags[0].kind == "number_not_found"
    assert "512.345" in result.flags[0].text


def test_the_decimal_slip_from_the_live_run_is_caught():
    """22 Sep: -0.2785 points became "%27,85 puan"."""
    state = _state(faiz_diff=[float("nan"), -0.2785])

    assert not VERIFY("Faiz %27,85 puan dustu.", state).passed
    assert VERIFY("Faiz 0,28 puan dustu.", state).passed


def test_rounding_follows_the_precision_the_answer_writes():
    state = _state(faiz=[18.3325])

    assert VERIFY("Faiz %18 idi.", state).passed
    assert VERIFY("Faiz %18,3 idi.", state).passed
    assert not VERIFY("Faiz %18,4 idi.", state).passed


def test_approximate_wording_widens_the_window():
    state = _state(faiz=[43.39])

    assert VERIFY("Faiz %40'un uzerine cikti.", state).passed
    assert VERIFY("Faiz yaklasik %42 oldu.", state).passed
    assert not VERIFY("Faiz %40 oldu.", state).passed


def test_the_english_decimal_style_is_accepted_too():
    """Answer.md asks for Turkish format; a slip into "18.33" is not a wrong
    number, and flagging it would be a false alarm under a correct answer."""
    state = _state(faiz=[18.3325])

    assert VERIFY("Faiz 18.33 idi.", state).passed


def test_scale_words_are_read_both_as_unit_and_as_multiplier():
    state = _state(kredi_diff=[20201.0], kredi=[615029.2])

    assert VERIFY("Her ay 20 bin milyon TL artti.", state).passed
    assert not VERIFY("Her ay 25 bin milyon TL artti.", state).passed
    assert VERIFY("Reel bakiye 615.029 milyon TL.", state).passed
    assert VERIFY("Reel bakiye yaklasik 615 milyar TL.", state).passed


def test_numbers_the_tools_or_the_question_stated_count_as_known():
    state = _state(
        question="60 aylik veriyi goster.",
        tool_text="4 ay kosullarin hepsini sagliyor: 2021-02. 288 hafta okundu.",
        kredi=[1.0],
    )

    assert VERIFY("60 aylik tabloda 288 haftalik veri var.", state).passed


def test_counts_years_dates_and_index_bases_are_not_values():
    state = _state(kredi=[1.0])

    answer = (
        "2021-2025 arasi 4 ay oyle gecti; 2025 yilinda TUFE 2025=100 bazinda. "
        "2021-02 ve 2024-01 aylari. Tuketici Kredileri (2+3+4) ve 1.1.1.Konut."
    )

    assert VERIFY(answer, state).passed, _flagged(VERIFY(answer, state))


def test_identifiers_and_code_spans_are_not_numbers():
    state = _state(sources=["evds:TP.KTF12"], kredi=[1.0])

    answer = "`TP_KTF12` sutunu, TP.KTF12 serisi ve `t01_get_series` tablosu."

    assert VERIFY(answer, state).passed, _flagged(VERIFY(answer, state))


# --- sources and dates --------------------------------------------------------


def test_a_number_cut_short_rather_than_rounded_passes():
    """Fourth 22 Sep run: 1716202.69 written "1.716.202" was printed under a
    correct answer as unverifiable. A digit more is still a wrong number."""
    state = _state(reel=[1716202.688])

    assert VERIFY("Reel karsiligi 1.716.202 milyon TL.", state).passed
    assert VERIFY("Reel karsiligi 1.716.203 milyon TL.", state).passed
    assert not VERIFY("Reel karsiligi 1.716.204 milyon TL.", state).passed


def test_a_series_that_is_not_among_the_sources_is_flagged():
    state = _state(sources=["evds:TP.KTF12"], kredi=[1.0])

    result = VERIFY("Kaynak: evds:TP.KTF12 ve evds:TP.YOK.", state)

    assert [flag.kind for flag in result.flags] == ["source_not_listed"]
    assert "evds:TP.YOK" in result.flags[0].text


def test_a_month_outside_the_tables_is_flagged_unless_the_question_asked():
    state = _state(kredi=[1.0])

    assert not VERIFY("2026-09 ayinda yukseldi.", state).passed
    asked = _state(question="2026-09 icin veri var mi?", kredi=[1.0])
    assert VERIFY("2026-09 icin havuzda veri yok.", asked).passed


# --- through the loop ---------------------------------------------------------


def test_the_loop_prints_the_flags_under_the_answer():
    from lakehouse_agent.agent.events import ListSink
    from lakehouse_agent.agent.loop import AgentDeps, run_turn
    from lakehouse_agent.config import AgentSettings
    from lakehouse_agent.testing.fake_llm import FakeLLM
    from lakehouse_agent.testing.fake_tools import make_fake_registry

    sink = ListSink()
    deps = AgentDeps(
        llm=FakeLLM(
            [
                FakeLLM.tool_call(
                    "get_series",
                    {
                        "series_ids": ["evds:TP.KTF12"],
                        "start": "2021-01",
                        "end": "2021-03",
                    },
                ),
                FakeLLM.text("Hazir."),
                FakeLLM.text("Faiz 2021-03'te 987,65 oldu."),
            ]
        ),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        sink=sink,
    )
    state = initial_state("s1")

    result = asyncio.run(run_turn(state, deps, "Faiz nasil?"))

    flag_lines = result.answer.split("**Dogrulanamayan ifadeler**")[1]
    assert "'987,65' tablolarda" in flag_lines
    payload = sink.last("verification").payload
    assert payload["passed"] is False
    assert payload["flags"][0]["kind"] == "number_not_found"


# --- the three real answers ---------------------------------------------------


@pytest.fixture(scope="module")
def live_session():
    """The live run's three turns, rebuilt offline: the same tool calls against
    the same committed gold data, and the answers the real model wrote."""
    from lakehouse_agent.lake import gold

    if not gold.OBSERVATIONS_PATH.exists():
        pytest.skip("gerçek gold verisi yok (data/gold/*.parquet)")

    from lakehouse_agent.agent.loop import AgentDeps, run_turn
    from lakehouse_agent.config import AgentSettings
    from lakehouse_agent.testing.fake_llm import FakeLLM
    from lakehouse_agent.tools.lake_adapter import LakeAdapter
    from lakehouse_agent.tools.toolset import make_registry

    answers = json.loads(FIXTURE.read_text(encoding="utf-8"))["answers"]
    credit, rate = "aylik_tuketici_konut_TL_sektor", "TP_KTF12"

    def call(name, args, n):
        return FakeLLM.tool_call(name, args, call_id=f"c{n}")

    script = [
        call(
            "get_series",
            {
                "series_ids": ["bddk:aylik:tuketici:konut:TL:sektor", "evds:TP.KTF12"],
                "start": "2021-01",
                "end": "2025-12",
            },
            1,
        ),
        call("transform_table", {"op": "diff", "column": credit}, 2),
        call("transform_table", {"op": "diff", "column": rate}, 3),
        call(
            "transform_table",
            {
                "op": "mark",
                "name": "faiz_dustu_kredi_artmadi",
                "conditions": [
                    {"column": f"{rate}_diff", "op": "<", "value": 0},
                    {"column": f"{credit}_diff", "op": "<=", "value": 0},
                ],
            },
            4,
        ),
        FakeLLM.text("Veriler hazir."),
        FakeLLM.text(answers[0]),
        call("transform_table", {"op": "deflate", "column": credit}, 5),
        FakeLLM.text("Veriler hazir."),
        FakeLLM.text(answers[1]),
        call(
            "get_series",
            {
                "series_ids": ["evds:TP.KFE.TR"],
                "start": "2021-01",
                "end": "2025-12",
                "into": "active",
            },
            6,
        ),
        call("transform_table", {"op": "diff", "column": "TP_KFE_TR"}, 7),
        FakeLLM.text("Veriler hazir."),
        FakeLLM.text(answers[2]),
    ]
    offline = lambda *_: (_ for _ in ()).throw(RuntimeError("ag yok"))  # noqa: E731
    deps = AgentDeps(
        llm=FakeLLM(script),
        registry=make_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        lake=LakeAdapter(embed_documents=offline, embed_query=offline),
    )
    state = initial_state("canli")
    verdicts = []
    for question in ("Soru 1: 60 aylik veri.", "Soru 2.", "Soru 3."):
        asyncio.run(run_turn(state, deps, question))
        verdicts.append(state.verification)
    return verdicts


def test_of_the_live_answers_only_the_wrong_number_is_flagged(live_session):
    first, second, third = live_session

    assert [flag.kind for flag in first.flags] == ["number_not_found"]
    assert "27,85" in first.flags[0].text
    assert second.passed, _flagged(second)
    assert third.passed, _flagged(third)


# --- regressions from the 22 Sep review ---------------------------------------


def test_a_month_a_tool_named_may_be_repeated():
    """get_series itself writes "Istenen bitis 2026-12; havuzdaki veri 2026-06
    ayinda bitiyor", and the answer is told to pass that on."""
    state = _state(
        tool_text="Istenen bitis 2026-12; havuzdaki veri 2025-12 ayinda bitiyor.",
        kredi=[1.0],
    )

    assert VERIFY("2026-12 icin veri yok; veri 2025-12'de bitiyor.", state).passed


@pytest.mark.parametrize(
    "sentence",
    [
        "KFE 200'ü aştı.",
        "KFE 200'ün üzerine çıktı.",
        "KFE 200’ün üstüne çıktı.",
        "Bakiye 800 bin milyon TL'yi aştı.",
        "Bakiye 800 bin milyon TL'nin üzerine çıktı.",
        "Bakiye yaklaşık olarak 800 bin milyon TL.",
        "Bakiye kabaca 800 bin milyon TL civarında.",
    ],
)
def test_bounds_and_round_figures_are_read_as_such(sentence):
    state = _state(kfe=[204.04], kredi=[801376.0])

    assert VERIFY(sentence, state).passed, _flagged(VERIFY(sentence, state))


def test_a_bound_is_still_checked():
    state = _state(kfe=[204.04])

    assert not VERIFY("KFE 400'ü aştı.", state).passed


@pytest.mark.parametrize(
    "sentence",
    [
        "KFE 2025 boyunca 165 ile 205 arasında seyretti.",
        "KFE 2025'te 165-205 bandında.",
        "Bakiye 520-680 bin milyon TL aralığında.",
        "Faiz %17-%45 aralığında.",
    ],
)
def test_a_range_is_checked_as_one_interval(sentence):
    state = _state(kfe=[165.88, 204.36], kredi=[523401.0, 678912.0], faiz=[17.1, 44.9])

    assert VERIFY(sentence, state).passed, _flagged(VERIFY(sentence, state))


def test_a_range_the_tables_do_not_reach_is_flagged():
    state = _state(kfe=[165.88, 204.36])

    assert not VERIFY("KFE 300 ile 400 arasında seyretti.", state).passed


@pytest.mark.parametrize(
    "sentence",
    [
        "Bakiye 5 trilyon TL'ye ulaştı.",
        "Bakiye her ay 25 milyar TL arttı.",
        "Faiz 9 puan arttı.",
    ],
)
def test_a_small_number_with_a_scale_or_unit_is_an_amount(sentence):
    """Only bare small numbers are counts; "5 trilyon TL" is a claim."""
    state = _state(kredi=[801376.0], faiz_diff=[0.5])

    assert not VERIFY(sentence, state).passed


def test_a_bin_tl_series_can_be_restated_in_milyar():
    """TP.HPBITABLO6.3 is in bin TL: 803.644.379 bin TL is about 804 milyar TL."""
    state = _state(kredi=[803644379.0])

    assert VERIFY("Bakiye yaklaşık 804 milyar TL.", state).passed
    assert VERIFY("Bakiye 803,6 milyar TL.", state).passed


def test_english_thousands_are_read_as_thousands_too():
    state = _state(kredi=[276740.0, 678912.0])

    assert VERIFY("Bakiye 276,740'tan 678,912'ye cikti.", state).passed
    assert not VERIFY("Bakiye 1,234,567 milyon TL oldu.", state).passed


def test_month_dot_year_is_a_date_not_a_number():
    state = _state(kredi=[678912.0])

    assert VERIFY("12.2025 itibarıyla bakiye 678.912 milyon TL.", state).passed


def test_basis_points_are_hundredths_of_a_point():
    state = _state(faiz_diff=[float("nan"), -0.2785])

    assert VERIFY("Faiz 27,85 baz puan düştü.", state).passed
    assert not VERIFY("Faiz 27,85 puan düştü.", state).passed


def test_digits_inside_ids_and_dates_in_tool_text_are_not_values():
    """'TP_KTF12' and '2021-12-01' in a tool message must not make 12 known."""
    state = _state(
        tool_text='{"period": "2021-12-01", "TP_KTF12": 18.33} t01_get_series',
        faiz=[18.33],
    )

    assert not VERIFY("Faiz %12'ye indi.", state).passed
