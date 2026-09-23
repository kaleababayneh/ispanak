"""Column labels come from lineage, and from nowhere else.

Every expected string below is either a catalog name the fixture wrote into
`label_tr`, a unit it wrote, or a phrase from `OP_PHRASES`. Nothing is
shortened or paraphrased: a label the lineage cannot justify is the raw
column name.
"""

from __future__ import annotations

from lakehouse_agent.agent.labels import OP_PHRASES, column_label, table_labels
from lakehouse_agent.contracts import ColumnInfo

REF = "t01_get_series"


def _lineage(*infos: ColumnInfo) -> dict[str, ColumnInfo]:
    return {f"{REF}.{info.name}": info for info in infos}


FAIZ = ColumnInfo(
    name="TP_KTF12",
    label_tr="Konut Kredisi (TL, Akım, %) Ağırlıklı ortalama",
    unit="%",
    tool="get_series",
)
BAKIYE = ColumnInfo(
    name="aylik_tuketici_tuketici_kredileri_konut_toplam_sektor",
    label_tr="Tüketici Kredileri - Konut (Toplam)",
    unit="milyon TL",
    tool="get_series",
)


def test_a_fetched_column_is_labelled_with_its_catalog_name_and_unit():
    lineage = _lineage(FAIZ, BAKIYE)

    assert (
        column_label(REF, BAKIYE.name, lineage)
        == "Tüketici Kredileri - Konut (Toplam) (milyon TL)"
    )


def test_a_unit_already_in_the_name_is_not_repeated():
    """`TP.KTF12`'s catalog name carries its `%` itself."""
    assert (
        column_label(REF, "TP_KTF12", _lineage(FAIZ))
        == "Konut Kredisi (TL, Akım, %) Ağırlıklı ortalama"
    )


def test_a_derived_column_is_its_inputs_label_plus_what_was_done():
    diff = ColumnInfo(
        name="TP_KTF12_diff",
        op="diff",
        inputs=["TP_KTF12"],
        unit="%",
        tool="transform_table",
    )
    lineage = _lineage(FAIZ, diff)

    assert (
        column_label(REF, "TP_KTF12_diff", lineage)
        == "Konut Kredisi (TL, Akım, %) Ağırlıklı ortalama — fark"
    )


def test_a_chain_of_derivations_is_followed_to_the_catalog():
    diff = ColumnInfo(name="TP_KTF12_diff", op="diff", inputs=["TP_KTF12"])
    pct = ColumnInfo(
        name="TP_KTF12_diff_pct", op="pct_change", inputs=["TP_KTF12_diff"], unit="%"
    )
    lineage = _lineage(FAIZ, diff, pct)

    label = column_label(REF, "TP_KTF12_diff_pct", lineage)

    assert label.startswith(
        "Konut Kredisi (TL, Akım, %) Ağırlıklı ortalama — fark — % değişim"
    )


def test_a_base_period_is_read_back_from_the_column_name():
    """`rebase` and `change_since` put the base in the name they generate."""
    rebased = ColumnInfo(
        name="TP_KFE_TR_endeks_202101", op="rebase", inputs=["TP_KFE_TR"], unit="endeks"
    )
    since = ColumnInfo(
        name="TP_KFE_TR_degisim_202101",
        op="change_since",
        inputs=["TP_KFE_TR"],
        unit="%",
    )
    kfe = ColumnInfo(name="TP_KFE_TR", label_tr="Konut Fiyat Endeksi (KFE)")
    lineage = _lineage(kfe, rebased, since)

    assert (
        column_label(REF, rebased.name, lineage)
        == "Konut Fiyat Endeksi (KFE) — endeks (2021-01 = 100)"
    )
    # `%` is already in the phrase, so the unit is not repeated after it.
    assert (
        column_label(REF, since.name, lineage)
        == "Konut Fiyat Endeksi (KFE) — % değişim (2021-01 bazlı)"
    )


def test_a_derived_column_drops_its_inputs_unit_and_carries_its_own():
    """Measured 23 Sep on gold data: a rebased loan balance read
    "... (milyon TL) — endeks (2021-01 = 100)". An index has no milyon TL."""
    bakiye = ColumnInfo(
        name="bakiye", label_tr="Konut Kredisi Bakiyesi", unit="milyon TL"
    )
    rebased = ColumnInfo(
        name="bakiye_endeks_202101", op="rebase", inputs=["bakiye"], unit="endeks"
    )
    diff = ColumnInfo(
        name="bakiye_diff", op="diff", inputs=["bakiye"], unit="milyon TL"
    )
    lineage = _lineage(bakiye, rebased, diff)

    assert (
        column_label(REF, rebased.name, lineage)
        == "Konut Kredisi Bakiyesi — endeks (2021-01 = 100)"
    )
    assert (
        column_label(REF, diff.name, lineage)
        == "Konut Kredisi Bakiyesi — fark (milyon TL)"
    )


def test_a_ratio_names_both_of_its_inputs():
    top = ColumnInfo(
        name="TP_AKONUTSAT2_KTRTOPLAM",
        label_tr="Türkiye_Konut_İpotekli Satışlar",
        unit="Adet",
    )
    bottom = ColumnInfo(
        name="TP_AKONUTSAT1_KTRTOPLAM",
        label_tr="Türkiye_Konut_Toplam Satışlar",
        unit="Adet",
    )
    ratio = ColumnInfo(
        name="TP_AKONUTSAT2_KTRTOPLAM_pay",
        op="ratio",
        inputs=["TP_AKONUTSAT2_KTRTOPLAM", "TP_AKONUTSAT1_KTRTOPLAM"],
        unit="%",
    )

    label = column_label(REF, ratio.name, _lineage(top, bottom, ratio))

    # The inputs' own unit is dropped: a share of two counts is not
    # "(Adet) / (Adet)"; the ratio's unit is the one that applies.
    # Catalog underscores are machine glue and become spaces.
    assert label == (
        "Türkiye Konut İpotekli Satışlar / Türkiye Konut Toplam Satışlar payı (%)"
    )


def test_unknown_columns_keep_their_own_name_and_fixed_ones_have_theirs():
    lineage = _lineage(FAIZ)

    assert column_label(REF, "period", lineage) == "Dönem"
    assert column_label(REF, "bilinmeyen", lineage) == "bilinmeyen"
    assert column_label("t09_baska", "TP_KTF12", lineage) == "TP_KTF12"  # other table


def test_an_operation_with_no_phrase_falls_back_to_the_tidied_name():
    """A new op added to the tool without a phrase here must not crash a
    chart; it shows the column name, underscores as spaces, until someone
    writes the phrase. Analysis tables (`ortalama_öncesi`) read this way."""
    odd = ColumnInfo(name="ortalama_öncesi", op="yeni_islem", inputs=["TP_KTF12"])

    assert (
        column_label(REF, "ortalama_öncesi", _lineage(FAIZ, odd)) == "ortalama öncesi"
    )
    assert "yeni_islem" not in OP_PHRASES


def test_the_models_label_flows_into_every_column_derived_from_it():
    """The model called `TP_KTF12` "Konut kredisi faiz oranı" on the chart.
    Its diff, and the diff's percent change, must start from those words,
    not from the catalog's technical name -- otherwise one table mixes two
    vocabularies for the same series."""
    diff = ColumnInfo(
        name="TP_KTF12_diff", op="diff", inputs=["TP_KTF12"], unit="yuzde puan"
    )
    pct = ColumnInfo(
        name="TP_KTF12_diff_pct", op="pct_change", inputs=["TP_KTF12_diff"], unit="%"
    )
    lineage = _lineage(FAIZ, diff, pct)
    said = {"TP_KTF12": "Konut kredisi faiz oranı (%)"}

    labels = table_labels(
        REF, ["TP_KTF12", "TP_KTF12_diff", "TP_KTF12_diff_pct"], lineage, said
    )

    assert labels == {
        "TP_KTF12": "Konut kredisi faiz oranı (%)",  # as written, unit and all
        "TP_KTF12_diff": "Konut kredisi faiz oranı (%) — fark (yuzde puan)",
        "TP_KTF12_diff_pct": "Konut kredisi faiz oranı (%) — fark — % değişim",
    }
    assert (
        column_label(REF, "TP_KTF12", lineage, {"TP_KTF12": "   "}) == FAIZ.label_tr
    )  # blank: no


def test_a_mark_column_is_named_by_the_name_the_model_gave_it():
    """Its inputs are the conditions, not the thing it measures; the label
    that started from the first input read "...faiz oranı — koşul işareti"."""
    mark = ColumnInfo(
        name="faiz_dustu_kredi_artmadi",
        op="mark",
        inputs=["TP_KTF12_diff", "bakiye_diff"],
    )

    assert (
        column_label(REF, mark.name, _lineage(FAIZ, mark))
        == "faiz dustu kredi artmadi (koşul)"
    )


def test_a_cycle_in_a_hand_built_lineage_stops():
    a = ColumnInfo(name="a", op="diff", inputs=["b"])
    b = ColumnInfo(name="b", op="diff", inputs=["a"])

    label = column_label(REF, "a", _lineage(a, b))

    assert label  # terminated; whatever it says, it said it finitely


def test_table_labels_is_identity_for_what_it_does_not_know():
    labels = table_labels(REF, ["period", "TP_KTF12", "x"], _lineage(FAIZ))

    assert labels == {
        "period": "Dönem",
        "TP_KTF12": "Konut Kredisi (TL, Akım, %) Ağırlıklı ortalama",
        "x": "x",
    }


def test_lineage_entries_may_be_plain_dicts():
    lineage = {f"{REF}.v": {"label_tr": "Bir Şey", "unit": "adet"}}

    assert column_label(REF, "v", lineage) == "Bir Şey (adet)"


def test_column_units_come_from_lineage_and_are_empty_when_unknown():
    from lakehouse_agent.agent.labels import column_units

    units = column_units(REF, ["period", "TP_KTF12", "yok"], _lineage(FAIZ))

    assert units == {"period": "", "TP_KTF12": "%", "yok": ""}


def test_a_deflated_column_names_its_price_base_once_through_the_unit():
    """The tool's unit already says "2021-01 fiyatlariyla"; the header read
    "— reel (2021-01 fiyatlarıyla) (milyon TL, 2021-01 fiyatlariyla)"."""
    bakiye = ColumnInfo(
        name="bakiye", label_tr="Konut Kredisi Bakiyesi", unit="milyon TL"
    )
    reel = ColumnInfo(
        name="reel_bakiye_202101",
        op="deflate",
        inputs=["bakiye"],
        unit="milyon TL, 2021-01 fiyatlariyla",
    )

    label = column_label(REF, reel.name, _lineage(bakiye, reel))

    assert label == "Konut Kredisi Bakiyesi — reel (milyon TL, 2021-01 fiyatlariyla)"
    assert label.count("2021-01") == 1
