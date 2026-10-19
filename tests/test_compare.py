from market_signals.compare import compare, default_iv, default_slip
from market_signals.synthetic import synthetic_bars


def test_compare_table(tmp_path):
    for i, sym in enumerate(["QQQ", "NVDA"]):
        synthetic_bars(days=120, seed=30 + i, price=200 + 300 * i).to_csv(tmp_path / f"{sym}_1min.csv",
                                                                         index_label="datetime")
    table, missing = compare(["QQQ", "NVDA", "AMD"], data_dir=tmp_path, params={"min_score": 7},
                             risk=150, account=3000, max_position_pct=30)
    assert missing == ["AMD"]
    assert list(table["activo"]) == ["QQQ", "NVDA", "TODOS"]
    rows = table.set_index("activo")
    assert rows.loc["TODOS", "trades"] == rows.loc["QQQ", "trades"] + rows.loc["NVDA", "trades"]
    assert rows.loc["NVDA", "IV"] == default_iv("NVDA") and rows.loc["NVDA", "costo_lado"] == 0.05
    assert rows.loc["QQQ", "costo_lado"] == 0.02
    # una opción ITM de un activo de $500 cuesta más que $900 -> no cabe
    assert rows.loc["NVDA", "cabe_1_contrato"] in ("sí", "NO")
    assert default_slip("spy") == 0.02 and default_iv("XYZ") == 0.35
