"""Deterministic original fixtures. Educational examples, not legal forms."""

import hashlib
import json
import random
import textwrap
from pathlib import Path

from reportlab.pdfgen import canvas

from .pdf import read_pdf
from .schema import Lease

FAMILIES = {"train": list(range(8)), "valid": [8, 9], "test": [10, 11]}


def validate_dataset(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "examples.jsonl").read_text().splitlines()]
    version = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if version != manifest["version"]:
        raise ValueError("Dataset content does not match its recorded version")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate document IDs")
    family_owners = {}
    for row in rows:
        Lease.model_validate(row["expected"])
        previous = family_owners.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError("A template family occurs in multiple splits")
    for split, count in manifest["counts"].items():
        expected = [r for r in rows if r["split"] == split]
        actual = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        if len(expected) != count or actual != expected:
            raise ValueError(f"Dataset {split} file disagrees with its versioned examples")
    if {r["split"] for r in rows} != set(manifest["counts"]):
        raise ValueError("Dataset split counts are incomplete")
    return manifest


PARTIES = [
    "{landlord} (Landlord) leases to {tenant} (Tenant).",
    "The lessor is {landlord}; the lessee is {tenant}.",
    "Owner: {landlord}. Occupier and tenant: {tenant}.",
    "This lease is between landlord {landlord} and tenant {tenant}.",
    "Tenant named in this agreement: {tenant}. Landlord: {landlord}.",
    "{tenant}, as Tenant, accepts a lease granted by {landlord}, as Landlord.",
    "LESSOR / LANDLORD: {landlord}\nLESSEE / TENANT: {tenant}",
    "The parties are {tenant} (the tenant) and {landlord} (the landlord).",
    "The property owner, {landlord}, is the landlord; {tenant} is the tenant.",
    "Tenant party - {tenant}; landlord party - {landlord}.",
    "The Landlord ({landlord}) and the Tenant ({tenant}) enter this lease.",
    "{landlord} grants possession as lessor to the lessee, {tenant}.",
]
PREMISES = [
    "Premises address: {address}.",
    "The leased premises are located at {address}.",
    "Property subject to this lease: {address}.",
    "The tenant will occupy {address}.",
]
STARTS = [
    "The lease commences on {start}.",
    "Commencement date: {start}.",
    "The tenancy begins on {start}.",
    "Possession and the lease term start on {start}.",
]
ENDS = [
    "The lease expires on {end}.",
    "Expiration date: {end}.",
    "The tenancy ends on {end}.",
    "The lease term terminates on {end}.",
]
RENTS = [
    "Initial base rent: {currency} {amount:,.2f} per {period}.",
    "For the initial term, base rent is {currency} {amount:,.2f}, payable each {period}.",
    "The Tenant shall pay base rent of {currency} {amount:,.2f} per {period} initially.",
    "Opening base rental payment: {currency} {amount:,.2f} every {period}.",
]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def render_pdf(path: Path, paragraphs: list[str], family: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), invariant=1)
    pdf.setTitle("Synthetic educational lease")
    font = ["Helvetica", "Times-Roman", "Courier"][family % 3]
    wrap_width = 78 if font != "Courier" else 70
    y = 750
    for paragraph in paragraphs:
        for line in paragraph.splitlines():
            for wrapped in textwrap.wrap(line, width=wrap_width):
                if y < 65:
                    pdf.showPage()
                    y = 750
                pdf.setFont(font, 10)
                pdf.drawString(50, y, wrapped)
                y -= 15
        y -= 10
    pdf.save()


def make_example(root: Path, split: str, family: int, index: int, hard: bool = False) -> dict:
    rng = random.Random(42 + family * 1000 + index)
    document_id = f"{split}-f{family:02d}-{index:03d}"
    tenant = f"{rng.choice(['Juniper', 'Cedar', 'Orchard', 'Meadow'])} Studio {family}{index} LLC"
    landlord = (
        f"{rng.choice(['Harbor', 'Summit', 'Brook', 'Maple'])} Properties {family}{index} Ltd"
    )
    address = f"{rng.randint(10, 999)} {rng.choice(['Oak', 'Market', 'Pine'])} Street, Suite {index + 1}, Example City"
    start = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    end = "2029" + start[4:]
    currency = rng.choice(["USD", "GBP", "EUR"])
    amount = rng.randint(1000, 9000) + rng.choice([0, 0.5, 0.75])
    period = rng.choice(["month", "year"])
    expected = {
        "tenant": tenant,
        "landlord": landlord,
        "premises_address": address,
        "commencement_date": start,
        "expiration_date": end,
        "base_rent": {"amount": amount, "currency": currency, "period": period},
    }
    missing_rent = index % 7 == 0 or (hard and index % 2 == 0)
    ambiguous_date = index % 10 == 0 or (hard and index % 3 == 0)
    no_end = index % 11 == 0
    if missing_rent:
        expected["base_rent"] = None
    if ambiguous_date:
        start = "03/04/2026 (date convention not specified)"
        expected["commencement_date"] = None
    if no_end:
        expected["expiration_date"] = None
    rent_line = (
        "Base rent will be agreed separately; no rental amount is stated."
        if missing_rent
        else RENTS[family % 4].format(currency=currency, amount=amount, period=period)
    )
    clauses = [
        PARTIES[family].format(landlord=landlord, tenant=tenant),
        PREMISES[family % 4].format(address=address),
        STARTS[family % 4].format(start=start),
        "The expiration date is not specified." if no_end else ENDS[family % 4].format(end=end),
        rent_line,
        f"Security deposit: {currency} {amount * 2:,.2f}. This refundable sum is not base rent.",
        "Signed on 2025-12-01. The execution date is separate from commencement.",
        "Tenant must maintain the premises. Renewal is subject to a later written agreement.",
    ]
    rng.shuffle(clauses)
    paragraphs = [f"COMMERCIAL LEASE - educational synthetic example {document_id}", *clauses]
    relative_pdf = f"pdfs/{document_id}.pdf"
    render_pdf(root / relative_pdf, paragraphs, family)
    text = read_pdf(root / relative_pdf)
    Lease.model_validate(expected)
    return {
        "id": document_id,
        "split": split,
        "family": f"lease-family-{family:02d}",
        "pdf": relative_pdf,
        "text": text,
        "expected": expected,
        "source": "original-synthetic",
        "license": "Apache-2.0",
        "source_clauses": clauses,
    }


def build_dataset(root: Path, variant: str = "v1") -> dict:
    root = Path(root)
    if variant not in {"v1", "v2", "v3"}:
        raise ValueError("Dataset variant must be v1, v2 or v3")
    if variant == "v3":
        from .diversity import v3_examples

        rows = v3_examples(root)
    else:
        rows = [
            make_example(root, split, family, index)
            for split, families in FAMILIES.items()
            for family in families
            for index in range(25)
        ]
        if variant == "v2":
            rows.extend(
                make_example(root, "train", family, index, hard=True)
                for family in FAMILIES["train"]
                for index in range(25, 38)
            )
    version = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    write_jsonl(root / "examples.jsonl", rows)
    for split in FAMILIES:
        write_jsonl(root / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    manifest = {
        "version": version,
        "variant": variant,
        "seed": 42,
        "counts": {s: sum(r["split"] == s for r in rows) for s in FAMILIES},
        "families": {s: sorted({r["family"] for r in rows if r["split"] == s}) for s in FAMILIES},
        "license": "Apache-2.0",
        "scope": "synthetic educational leases; not a real-world benchmark",
    }
    write_json(root / "manifest.json", manifest)
    return manifest
