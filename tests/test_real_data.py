from lease_lab.real_data import benchmark_cases, source_text


def test_external_html_conversion_excludes_markup_and_keeps_paragraphs():
    assert source_text("<html><body><p>Tenant: Example LLC</p><p>Rent: 100</p></body></html>") == [
        "Tenant: Example LLC",
        "Rent: 100",
    ]


def test_reviewed_benchmark_does_not_infer_currency_from_yen_symbol():
    case = next(c for c in benchmark_cases() if c["id"] == "aleasebert-73")
    assert case["expected"]["base_rent"] == {"amount": 3600.0, "currency": None, "period": "month"}
    assert case["expected"]["commencement_date"] == "2011-03-01"
