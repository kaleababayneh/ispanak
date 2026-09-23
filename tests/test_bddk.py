"""BDDK Aylık Bülten, bronze table files -> silver (lake/bddk.py).

Synthetic tables in the shape the bulletin returns (measured 22 Sep 2026),
with hand-computed expectations.
"""

from datetime import date

import pandas as pd
import pytest

from lakehouse_agent.lake import bddk

KONUT = "bddk:aylik:tuketici:konut:TL:sektor"


def _consumer(write, year, month, konut):
    write(
        4,
        year,
        month,
        f"Tüketici Kredileri (milyon TL), Dönem:{year}/{month}",
        ["TP", "YP", "Toplam"],
        [
            ("Tüketici Kredileri (2+3+4)", "bold", konut + 10, 1, konut + 11),
            ("Tüketici Kredileri - Konut", "", konut, 1, konut + 1),
        ],
    )


def test_every_row_and_column_becomes_a_series_the_scenarios_ids_kept(bddk_bronze):
    _consumer(bddk_bronze, 2024, 1, 100)

    observations, catalog, _ = bddk.build_monthly()

    ids = set(catalog["series_id"])
    assert KONUT in ids  # the alias, not a generated id, for (4, Konut, TP)
    assert "bddk:aylik:tuketici:tuketici_kredileri_konut:yp:sektor" in ids
    assert "bddk:aylik:tuketici:tuketici_kredileri:toplam:sektor" in ids
    assert len(ids) == 6
    row = observations.set_index("series_id").loc[KONUT]
    assert (row["value"], row["unit"], row["freq"], row["obs_date"]) == (
        100.0,
        "milyon TL",
        "M",
        date(2024, 1, 1),
    )


def test_a_renumbered_total_stays_one_series(bddk_bronze):
    """BDDK renumbers the rows a total adds up when it adds a row: one item,
    two labels. The formula is not part of the series' identity."""
    for month, formula in ((1, "(10+27+28)"), (2, "(10+30+31)")):
        bddk_bronze(
            12,
            2024,
            month,
            f"Sermaye Yeterliliği (milyon TL), Dönem:2024/{month}",
            ["Toplam"],
            [(f"Risk Ağırlıklı Kalemler Toplamı {formula}", "", 500 + month)],
        )

    observations, catalog, _ = bddk.build_monthly()

    assert list(catalog["series_id"]) == [
        "bddk:aylik:sermaye:risk_agirlikli_kalemler_toplami:toplam:sektor"
    ]
    assert list(observations["value"]) == [501.0, 502.0]
    assert catalog["name_tr"].iloc[0].endswith("(10+30+31) (Toplam)")  # latest label


def test_a_repeated_label_is_qualified_with_its_parent(bddk_bronze):
    bddk_bronze(
        9,
        2024,
        1,
        "Mevduat Türler İtibarıyla (milyon TL), Dönem:2024/1",
        ["Toplam"],
        [
            ("TP Mevduat - Yurt İçi Yerleşik", "bold", 10),
            ("a) Gerçek Kişiler", "italic", 7),
            ("Döviz Tevdiat Hesabı - Yurt İçi Yerleşik", "bold", 20),
            ("a) Gerçek Kişiler", "italic", 15),
        ],
    )

    _, catalog, _ = bddk.build_monthly()

    names = set(catalog["name_tr"])
    assert "TP Mevduat - Yurt İçi Yerleşik - a) Gerçek Kişiler (Toplam)" in names
    assert (
        "Döviz Tevdiat Hesabı - Yurt İçi Yerleşik - a) Gerçek Kişiler (Toplam)" in names
    )
    assert catalog["series_id"].is_unique


@pytest.mark.parametrize(
    ("tablo_no", "caption", "column", "label", "unit", "kind"),
    [
        (
            15,
            "Rasyolar",
            "Rasyo",
            "Takipteki Alacaklar / Toplam Nakdi Krediler (%)",
            "%",
            "rate",
        ),
        (
            15,
            "Rasyolar",
            "Rasyo",
            "Toplam Mevduat / Ortalama Şube Sayısı (Bin TL)",
            "Bin TL",
            None,
        ),
        (
            12,
            "Sermaye Yeterliliği (milyon TL)",
            "Toplam",
            "Standart Rasyo (YÜZDE)",
            "%",
            "rate",
        ),
        (16, "Diğer Bilgiler", "Adet", "Banka Sayısı", "Adet", "stock"),
        (
            9,
            "Mevduat Türler İtibarıyla (milyon TL)",
            "Toplam",
            "Mudi Sayısı - Yurt İçi*",
            "Adet",
            "stock",
        ),
        (
            17,
            "Yurt Dışı Şube Rasyoları",
            "Rasyo",
            "Yurt Dışı Şubeler Kredileri / Toplam Krediler",
            None,
            "rate",
        ),
        (
            5,
            "Sektörel Kredi Dağılımı (bin TL)",
            "Nakdi Krediler",
            "İnşaat",
            "bin TL",
            "stock",
        ),
    ],
)
def test_a_value_carries_the_unit_its_row_column_or_table_states(
    bddk_bronze, tablo_no, caption, column, label, unit, kind
):
    """A ratio in "Bin TL" is not a percentage, and a ratio table that states
    no unit gets none: nothing is guessed."""
    bddk_bronze(
        tablo_no, 2024, 1, f"{caption}, Dönem:2024/1", [column], [(label, "", 1.5)]
    )

    _, catalog, _ = bddk.build_monthly()

    row = catalog.iloc[0]
    assert (None if pd.isna(row["unit"]) else row["unit"]) == unit
    assert (None if pd.isna(row["kind"]) else row["kind"]) == kind


def test_profit_and_loss_lines_become_monthly_flows(bddk_bronze):
    """The bulletin publishes them year to date: 100, 250, 450 and then 80
    in the new January. Monthly: 100, 150, 200, 80."""
    for (year, month), cumulative in {
        (2024, 1): 100,
        (2024, 2): 250,
        (2024, 3): 450,
        (2025, 1): 80,
    }.items():
        bddk_bronze(
            2,
            year,
            month,
            f"Kar Zarar (milyon TL), Dönem:{year}/{month}",
            ["Toplam"],
            [("DÖNEM NET KARI (ZARARI) (51-52)", "bold", cumulative)],
        )

    observations, catalog, _ = bddk.build_monthly()

    assert list(observations.sort_values("period")["value"]) == [
        100.0,
        150.0,
        200.0,
        80.0,
    ]
    row = catalog.iloc[0]
    assert (row["kind"], row["agg_method"], row["cum_type"]) == ("flow", "sum", "ytd")
    assert (
        row["series_id"] == "bddk:aylik:kar_zarar:donem_net_kari_zarari:toplam:sektor"
    )


def test_two_rows_on_one_id_are_kept_apart(bddk_bronze):
    """Deduplication would silently drop one of them."""
    bddk_bronze(
        1,
        2024,
        1,
        "Bilanço (milyon TL), Dönem:2024/1",
        ["Toplam"],
        [("Krediler (1+2)", "", 1), ("Krediler (3+4)", "", 2)],
    )

    observations, _, _ = bddk.build_monthly()

    assert sorted(observations["value"]) == [1.0, 2.0]
    assert observations["series_id"].is_unique


# --- labels -----------------------------------------------------------------


def test_nested_repeated_items_stay_apart():
    """The custody table lists "a) Kamu Borçlanma Senetleri" under the banks
    of two different sections: four rows, four series."""
    rows = [
        (1, "Yurtiçi Yerleşikler (2+5)", True),
        (2, "Bankalar (3+4)", False),
        (3, "a) Kamu Borçlanma Senetleri", False),
        (4, "b) Eurobondlar", False),
        (5, "Yurtdışı Yerleşikler (6)", True),
        (6, "Bankalar (7)", False),
        (7, "a) Kamu Borçlanma Senetleri", False),
    ]

    labels = bddk.qualified_labels(rows)

    assert labels[2] == "Yurtiçi Yerleşikler - Bankalar - a) Kamu Borçlanma Senetleri"
    assert labels[6] == "Yurtdışı Yerleşikler - Bankalar - a) Kamu Borçlanma Senetleri"
    assert labels[3] == "Yurtiçi Yerleşikler - Bankalar - b) Eurobondlar"
    assert labels[0] == "Yurtiçi Yerleşikler (2+5)"  # unambiguous, left alone


def test_a_list_item_is_named_with_its_parent():
    """ "a) Konut" alone says nothing a search could match."""
    rows = [
        (3, "Tüketici Kredileri (4+5)", False),
        (4, "a) Konut", False),
        (5, "b) Taşıt", False),
    ]

    assert bddk.qualified_labels(rows)[1] == "Tüketici Kredileri - a) Konut"


def test_a_list_item_hangs_from_the_row_above_its_list_not_the_section():
    """Profit and loss, June 2026: the fee rows have no formula to point at
    their parent, and the nearest bold row is the net interest income above
    them. The item belongs to the row its list starts under."""
    rows = [
        (26, "PROV.SONRASI NET FAİZ GELİRİ (24-25)", True),
        (27, "Kredilerden Alınan Ücret ve Komisyonlar", False),
        (28, "a) Nakdi Kredilerden Alınan Ücret ve Komisyonlar", False),
        (29, "b) Gayrinakdi Kredilerden Alınan Ücret ve Komisyonlar", False),
        (30, "Alınan Kar Payları", False),
    ]

    labels = bddk.qualified_labels(rows)

    assert labels[2] == (
        "Kredilerden Alınan Ücret ve Komisyonlar - "
        "a) Nakdi Kredilerden Alınan Ücret ve Komisyonlar"
    )
    assert labels[3].startswith("Kredilerden Alınan Ücret ve Komisyonlar - b)")
    assert labels[4] == "Alınan Kar Payları"


# --- the weekly bulletin ---------------------------------------------------------


def _weekly_html(day_text, rows):
    body = "".join(
        f'<tr class="satir" style="{"font-weight:bold;" if bold else ""}">'
        f"<td>{sira}</td><td>{label}</td>"
        + "".join(f"<td> {value}</td>" for value in values)
        + "</tr>"
        for sira, label, bold, values in rows
    )
    return (
        '<table class="table" id="TabloExcel2"><thead><tr><th></th>'
        f"<th>Sekt&#246;r / Krediler ( {day_text} <text>) (Milyon TL) </text></th>"
        "<th>TP</th><th>YP</th><th>TOPLAM</th></tr></thead>"
        f"{body}</table>"
    )


def _write_weekly(root, day, html):
    import json

    path = root / "t289" / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"html": html, "fetched_at": "2026-09-22T00:00:00+00:00"}),
        encoding="utf-8",
    )


def test_a_weekly_value_is_filed_under_its_monday_and_observed_on_its_day(tmp_path):
    """A holiday week's bulletin is dated Thursday 22 April 2021: that is the
    observation day, and the week is still Monday 19 April's."""
    rows = [
        (3, "Tüketici Kredileri (4)", True, ["675.735,98", "79,607", "675.815,587"]),
        (4, "a) Konut", False, ["277.825,065", "0", "277.825,065"]),
    ]
    _write_weekly(tmp_path, "2021-04-22", _weekly_html("22 Nisan 2021 Perşembe", rows))

    observations, catalog, _ = bddk.build_weekly(tmp_path)

    konut = observations[
        observations["series_id"]
        == "bddk:haftalik:krediler:tuketici_kredileri_a_konut:tp:sektor"
    ].iloc[0]
    assert (konut["period"], konut["obs_date"], konut["freq"]) == (
        date(2021, 4, 19),
        date(2021, 4, 22),
        "W",
    )
    assert konut["value"] == 277825.065  # Turkish number format read
    names = set(catalog["name_tr"])
    assert "Tüketici Kredileri - a) Konut (TP)" in names
    assert set(catalog["unit"]) == {"Milyon TL"}


# --- FinTürk ---------------------------------------------------------------------


def _write_finturk(root, tablo_no, table, columns, cells):
    import json

    record = {
        "params": {"tabloNo": tablo_no},
        "table": table,
        "fetched_at": "2026-09-22T00:00:00+00:00",
        "payload": {
            "Json": {
                "colNames": ["Eftodu", "Yıl", "Ay", "Şehir", "Grup", *columns],
                "data": {"rows": [{"cell": cell} for cell in cells]},
            }
        },
    }
    path = root / f"t{tablo_no:02d}" / "2026-6.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_finturk_is_one_quarterly_series_per_province_and_column(tmp_path):
    _write_finturk(
        tmp_path,
        3,
        "Bireysel Bankacılık (Bin TL)",
        ["Konut Kredisi"],
        [
            [10001, 2026, 6, "İSTANBUL", "SEKTÖR", 1500],
            [10001, 2026, 6, "ADANA", "SEKTÖR", 90],
        ],
    )

    observations, catalog, _ = bddk.build_finturk(tmp_path)

    row = observations.set_index("series_id").loc[
        "bddk:finturk:bireysel:istanbul:konut_kredisi:sektor"
    ]
    assert (row["period"], row["freq"], row["value"], row["unit"]) == (
        date(2026, 4, 1),  # the quarter ending in June
        "Q",
        1500.0,
        "Bin TL",
    )
    assert set(catalog["topic"]) == {"FinTürk: Bireysel Bankacılık"}


@pytest.mark.parametrize(
    ("column", "unit", "kind"),
    [
        ("Yurtiçi Şube Sayısı", "Adet", "stock"),
        ("Şubeye Düşen Nüfus", "Kişi", None),
        ("Kişi Başı Nakdi Kredi", "TL", None),
    ],
)
def test_the_branches_table_says_what_each_column_is(column, unit, kind):
    """A count of branches is a stock; people per branch and credit per head
    are ratios, and get no kind rather than a wrong one."""
    assert bddk.finturk_unit(column, "TL") == (unit, kind)


def test_finturk_ratios_are_rates():
    assert bddk.finturk_unit("Konut Kredisi Performans Oranı", "%") == ("%", "rate")
