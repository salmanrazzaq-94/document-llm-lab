"""A separate, noncommercial evaluation set. Never used for adapter training."""

import hashlib
import io
import json
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from .dataset import render_pdf, write_json, write_jsonl
from .pdf import read_pdf
from .schema import Lease

SOURCE = "https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/35058109"
ARCHIVE_SHA256 = "eefd69346fda26223ade30c1d4eb2f8ad7088b03ea2660d56d6b6582da909250"
ATTRIBUTION = "Spyretta Leivaditi, Julien Rossi, Evangelos Kanoulas. A Benchmark for Lease Contract Review (2020); ALeaseBert dataset (2022). DOI: 10.21942/uva.19732993.v1"


def source_text(html: str) -> list[str]:
    class Reader(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paragraphs = []

        def handle_data(self, data):
            if data.strip():
                self.paragraphs.append(data.strip())

    parser = Reader()
    parser.feed(html)
    # Normalize typographic punctuation to the PDF's built-in font encoding.
    replacements = str.maketrans(
        {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
    )
    return [p.translate(replacements) for p in parser.paragraphs]


def benchmark_cases() -> list[dict]:
    return [
        {
            "id": "aleasebert-237",
            "archive_path": "pool/ad9zo.dzAh1VkbRe3UFE6_pwBpL0-lease_contract_237.plain.html",
            "expected": {
                "tenant": "Big Tree International Co., Ltd.",
                "landlord": "Shantou Youbang International Express Supervise Center Co., Ltd",
                "premises_address": "Waisha Shanfen Road, Taixinglong Yard, Building 7,8 and 9, Longhu District",
                "commencement_date": "2012-10-01",
                "expiration_date": "2017-12-31",
                "base_rent": {"amount": 120000.0, "currency": "CNY", "period": "month"},
            },
            "annotation_notes": "Parties at heading; address in Article 1; explicit dates in Article 2. Article 3 states RMB 120,000 PER MONTH although paid quarterly. RMB is the named renminbi currency, mapped to ISO CNY. Decoration-period deposit is not additional rent.",
            "evidence": {
                "dates": "effective from October 1, 2012 and ending on December 31, 2017",
                "rent": "The rent per month is RMB 120,000, the payment shall be paid quarterly.",
            },
        },
        {
            "id": "aleasebert-73",
            "archive_path": "pool/avJvIqAKYDzjh1GH0ZUNkwEpH4.G-lease_contract_73.plain.html",
            "expected": {
                "tenant": "Xingguo General Fruit Industry Development Co., Ltd.",
                "landlord": "XU Shiyong",
                "premises_address": "128 Pingchuan Boulevard, Xingguo County",
                "commencement_date": "2011-03-01",
                "expiration_date": "2014-02-28",
                "base_rent": {"amount": 3600.0, "currency": None, "period": "month"},
            },
            "annotation_notes": "Parties at heading; address in clause 1. Clause 2 states dates, first-year rent and a separate deposit. Yen/yuan symbol alone is ambiguous; currency is null. Later 10% increases are outside the initial-rent field.",
            "evidence": {
                "dates": "from March 1, 2011 to February 28, 2014",
                "rent": "The rent for the first year of the lease is ¥3,600 per month",
            },
        },
    ]


def fresh_cases() -> list[dict]:
    """Prospective challenge: labels fixed before any v3 model predictions."""
    return [
        {
            "id": "aleasebert-84",
            "archive_path": "pool/a.7gwt0jast7_yYkZGNMh7zbFZZ8-lease_contract_84.plain.html",
            "expected": {
                "tenant": "Shantou Big Tree Toys Co., Ltd",
                "landlord": "Shantou Yunjia Fashion Clothing Co., Ltd.",
                "premises_address": "South Part 1, Nanshe Area, Pengnan Industrial Park on North Yingbinbei Road in Waisha Town of Longhu District in Shantou City",
                "commencement_date": "2011-01-01",
                "expiration_date": "2021-12-31",
                "base_rent": {"amount": 72000.0, "currency": "CNY", "period": "year"},
            },
            "annotation_notes": "Use the explicit whole-property annual rent in RMB, not the monthly per-square-meter rate or parenthetical USD conversion. Keep the explicit dates even though the stated ten-year duration disagrees. This source shares a tenant corporate family with development case 237; not an independent-population sample.",
            "evidence": {
                "dates": "on January 1, 2011 and take back on December 31, 2021",
                "rent": "The annual rent is RMB 72,000 (US $ 11,340)",
            },
        },
        {
            "id": "aleasebert-166",
            "archive_path": "pool/azVMqwVBF_D9jfBlqAg989G2P4F4-lease_contract_166.plain.html",
            "expected": {
                "tenant": "CEEG (Nanjing) PV-Tech Co., Ltd.",
                "landlord": None,
                "premises_address": "No. 6 West Chengfou Road",
                "commencement_date": "2007-04-01",
                "expiration_date": "2009-04-01",
                "base_rent": {"amount": 207360.0, "currency": "CNY", "period": "year"},
            },
            "annotation_notes": "Landlord is ambiguous: heading names China Electric Equipment Group Co., Ltd., signature names CEEG (Nanjing) Special Transformers Co., Ltd.; no precedence rule resolves this. Return null under the unchanged ambiguity policy. Rental is distinct from safety insurance and utilities. Do not calculate an aggregate monthly rate from the per-unit parenthesis.",
            "evidence": {
                "dates": "from April 1, 2007 to April 1, 2009",
                "rent": "Rental is RMB 207,360 per year",
                "landlord_conflict": "Lessor: China Electric Equipment Group Co., Ltd. / Lessor (seal): CEEG (Nanjing) Special Transformers Co., Ltd.",
            },
        },
        {
            "id": "aleasebert-201",
            "archive_path": "pool/a0Xts4kMzdTGCoSkzyR4Ag3s93q0-lease_contract_201.plain.html",
            "expected": {
                "tenant": "Anhui Xuelingxian Pharmaceutical Co., Ltd.",
                "landlord": "Bozhou Fengyi Chinese Medicine Development and Research Institute",
                "premises_address": "East Liuge Village, Weiwu Road, Bozhou",
                "commencement_date": "2008-08-01",
                "expiration_date": "2023-07-31",
                "base_rent": {"amount": 100000.0, "currency": "CNY", "period": "month"},
            },
            "annotation_notes": "Article 3 explicitly identifies 100000 Yuan as monthly rent; the preceding 1.2 million has no explicit period. Yuan names the currency; CNY is not inferred from location. Article 1 gives the property, not the representatives' names.",
            "evidence": {
                "dates": "from Aug 1, 2008 to July 31, 2023",
                "rent": "pay 100000 Yuan to Party A before 15 of each month as the monthly rent",
            },
        },
    ]


def build_real_dataset(root: Path, archive: Path | None = None, suite: str = "original") -> dict:
    if suite not in {"original", "fresh"}:
        raise ValueError("Real suite must be original or fresh")
    if suite == "original":
        cases = benchmark_cases()
        scope = (
            "Two real lease texts rendered as new PDFs. Tiny qualitative challenge set, not a "
            "representative benchmark. Evaluation only; excluded from all training."
        )
    else:
        cases = fresh_cases()
        scope = (
            "Three prospectively labeled real leases, evaluation only. Selected for property-lease "
            "scope, explicit fixed rent and context fit before model predictions. Tiny convenience "
            "sample from the same corpus, not a representative or contamination-free benchmark. "
            "Original two cases are development/regression cases."
        )
    if archive:
        content = archive.read_bytes()
    else:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("External archive changed; review it before changing the pinned checksum")
    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        for case in cases:
            paragraphs = source_text(source.read(case["archive_path"]).decode())
            pdf = f"pdfs/{case['id']}.pdf"
            render_pdf(root / pdf, paragraphs, family=0)
            Lease.model_validate(case["expected"])
            rows.append(
                {
                    **case,
                    "split": "test",
                    "family": "external-aleasebert",
                    "pdf": pdf,
                    "text": read_pdf(root / pdf),
                    "source": SOURCE,
                    "license": "CC-BY-NC-4.0",
                    "attribution": ATTRIBUTION,
                    "changes": "HTML reflowed as a selectable-text PDF; typographic punctuation normalized. Original layout not retained.",
                    "reviewed_by": "Codex; independent reading of the source contract, not model-generated labels",
                }
            )
    version = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    write_jsonl(root / "test.jsonl", rows)
    write_jsonl(root / "examples.jsonl", rows)
    manifest = {
        "version": version,
        "counts": {"test": len(rows)},
        "source": SOURCE,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC-BY-NC-4.0",
        "license_source": SOURCE,
        "attribution": ATTRIBUTION,
        "scope": scope,
    }
    write_json(root / "manifest.json", manifest)
    (root / "LICENSE.txt").write_text(
        f"{ATTRIBUTION}\nSource: {SOURCE}\n"
        "The dataset authors specify CC BY-NC 4.0 in their DATA/LICENSE description.\n"
        "https://creativecommons.org/licenses/by-nc/4.0/\n"
        "These documents retain that separate license; the repository Apache-2.0 license does not apply to them.\n"
        "Changes: HTML converted to PDF with line wrapping and normalized punctuation.\n"
    )
    return manifest
