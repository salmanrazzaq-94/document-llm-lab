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


def build_real_dataset(root: Path, archive: Path | None = None) -> dict:
    if archive:
        content = archive.read_bytes()
    else:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("External archive changed; review it before changing the pinned checksum")
    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        for case in benchmark_cases():
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
        "scope": "Two real lease texts rendered as new PDFs. Tiny qualitative challenge set, not a representative benchmark. Evaluation only; excluded from all training.",
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
