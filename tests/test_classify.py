"""Kind and unit rules for pool series (lake/classify.py).

The expected values are the hand mapping the scenario series had before the
rules existed (lake/gold.py `SCENARIO_SERIES`, docs/data_dictionary.md), and
real catalog names for each rule's corner.
"""

import pytest

from lakehouse_agent.lake.classify import evds_kind, evds_unit


@pytest.mark.parametrize(
    ("unit", "name", "agg", "kind"),
    [
        # The scenario series, exactly as the EVDS catalog lists them.
        ("Ağırlıklı ortalama", "Konut Kredisi (TL, Akım, %)", "avg", "rate"),
        (None, "Konut Fiyat Endeksi (KFE)", "avg", "index"),
        (None, "Genel Endeks", "avg", "index"),
        ("Adet", "Türkiye_Konut_Toplam Satışlar", "sum", "flow"),
        ("bin TL", "1.1.1.Konut (TL + YP)", "last", "stock"),
    ],
)
def test_the_rules_reproduce_the_hand_mapping(unit, name, agg, kind):
    assert evds_kind(unit, name, agg)[0] == kind


def test_interest_income_is_an_amount_not_a_rate():
    """A bare "faiz" in the name made budget lines ("Faiz Gelirleri", summed
    by EVDS) rates; they are TL flows."""
    assert evds_kind("bin TL", "1.1.4.1. Faiz Gelirleri", "sum")[0] == "flow"


def test_a_rate_published_as_an_end_of_period_value_is_still_a_rate():
    assert evds_kind(None, "Politika Faiz Oranı (%)", "last")[0] == "rate"


def test_the_unit_outranks_the_aggregation():
    assert evds_kind("2003=100", "Tüketici Fiyatları", "last")[0] == "index"
    assert evds_kind("Yüzde", "Kapasite Kullanımı", "last")[0] == "rate"


def test_a_series_no_rule_covers_gets_no_kind():
    """An average of counts or a TL/m2 price is neither a stock nor a rate;
    a guessed kind would steer deflate and diff without anyone seeing it."""
    kind, reason = evds_kind("TL/m2", "Konut Birim Fiyatları", "avg")

    assert kind is None
    assert reason.startswith("belirsiz")


@pytest.mark.parametrize(
    ("birimi", "name", "unit", "source"),
    [
        ("bin TL", "Mevduat", "bin TL", "BIRIMI"),
        ("", "Enflasyon Beklentisi (%)", "%", "ad"),
        (None, "Sanayi Üretimi (2010=100)", "2010=100", "ad"),
        (None, "Tüketici Fiyatları (İTO 2023=100)", "İTO 2023=100", "ad"),
        (None, "Kredi Kartı Harcamaları (Arşiv)", None, "yok"),
        (None, "Vade Yapısı (24D2)", None, "yok"),
        (None, "Konut Fiyat Endeksi (KFE)", None, "yok"),
    ],
)
def test_a_unit_is_read_from_the_name_only_when_it_is_one(birimi, name, unit, source):
    assert evds_unit(birimi, name) == (unit, source)


def test_a_missing_catalog_cell_is_read_as_empty():
    """pandas hands a missing cell over as NaN, not None."""
    nan = float("nan")

    assert evds_unit(nan, nan) == (None, "yok")
    assert evds_kind(nan, "Genel Endeks", nan)[0] == "index"


def test_the_review_lists_the_series_without_a_kind_by_datagroup():
    import pandas as pd

    from lakehouse_agent.lake.classify import render_kind_review

    classification = pd.DataFrame(
        {
            "series_id": ["evds:A", "evds:B"],
            "datagroup": ["bie_fiyat", "bie_faiz"],
            "kind": [None, "rate"],
            "kind_reason": ["belirsiz: ...", "birim oran/yuzde"],
            "unit": ["TL/m2", "Yüzde"],
        }
    )
    catalog = pd.DataFrame(
        {
            "series_id": ["evds:A", "evds:B"],
            "name_tr": ["Konut Birim Fiyatı", "Politika Faizi"],
            "source": ["EVDS", "EVDS"],
            "kind": [None, "rate"],
        }
    )

    text = render_kind_review(classification, catalog)

    assert "türü boş kalan 1 seri" in text
    assert "| bie_fiyat | 1 | TL/m2 | Konut Birim Fiyatı |" in text
    assert "| EVDS | 0 | 0 | 1 | 0 | 1 |" in text


def test_a_percentage_in_the_name_beats_the_datagroups_unit():
    """BIRIMI is per datagroup: the labour-force group gives its
    unemployment rate the unit of its levels."""
    assert evds_unit("Bin kişi", "8.İşsizlik oranı (%)") == (
        "%",
        "ad (grup birimi yerine)",
    )
    assert evds_kind("%", "8.İşsizlik oranı (%)", "avg")[0] == "rate"
    # The scale stays with BIRIMI where the name only says the currency.
    assert evds_unit("bin TL", "Mevduat (TL)") == ("bin TL", "BIRIMI")


def test_a_bare_percent_unit_is_a_rate():
    assert evds_kind("%", "Kapasite Kullanımı", "last") == ("rate", "birim oran/yuzde")
