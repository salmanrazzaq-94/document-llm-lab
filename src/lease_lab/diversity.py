"""Original lease language covering formats absent from the first experiment.

No external contract text or model-generated labels are used here. Each label is
derived from the facts used to render its PDF, before PDF text extraction.
"""

import random
from datetime import date
from pathlib import Path

from .dataset import make_example, render_pdf
from .pdf import read_pdf
from .schema import Lease

FAMILIES = {"train": list(range(20, 28)), "valid": [28, 29], "test": [30, 31]}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DATE_STYLES = ["iso", "month_first", "day_first", "ordinal", "abbreviated"]
PARTIES = [
    "Party A: {landlord}\nParty B: {tenant}\nParty A owns the property and grants Party B possession in exchange for rent.",
    "Party A: {tenant}\nParty B: {landlord}\nParty A takes the property on lease from Party B and must pay Party B rent.",
    "{landlord} (the owner) permits {tenant} (the occupier) to use the property for the agreed rental payment.",
    "The first party, {tenant}, rents the property from the second party, {landlord}. The first party pays the second party.",
    "LESSOR\n{landlord}\nLESSEE\n{tenant}\nThe lessee accepts possession from the lessor.",
    "Between {tenant}, hereinafter called Tenant, and {landlord}, hereinafter called Landlord.",
    "The owner of the premises is {landlord}. The business renting the premises is {tenant}.",
    "{tenant} leases the property from {landlord}; rent is owed by the former to the latter.",
    "Contracting party one: {landlord}\nContracting party two: {tenant}\nParty one lets its building to party two for a rental fee.",
    "Occupying party: {tenant}\nOwning party: {landlord}\nThe occupying party shall remit rent to the owning party.",
    "Property provider: {landlord}\nProperty user: {tenant}\nThe provider lets the property to the user, who pays the rental charge.",
    "The premises are taken on rent by {tenant} from their owner, {landlord}.",
]
LOCATIONS = [
    "Property let under this agreement: {address}.",
    "The building rented to the occupier is at {address}.",
    "Location of the leased office: {address}.",
    "The subject property has the following address: {address}.",
    "The demised premises are situated at {address}.",
    "Only the office located at {address} is covered by this lease.",
    "The property to be made available is {address}.",
    "The address for the rental property is {address}.",
    "The space being let is located at {address}.",
    "For this tenancy, the premises location is {address}.",
    "The premises that the property user may occupy are at {address}.",
    "The rented building stands at {address}.",
]
TERMS = [
    "The letting runs from {start} until {end}.",
    "Occupation under the lease begins {start} and ends {end}.",
    "The agreed tenancy starts on {start}; its final day is {end}.",
    "Term of possession: {start} through {end}.",
    "Commencement: {start}\nExpiry: {end}",
    "The first day of the lease is {start}. The last day is {end}.",
    "This tenancy takes effect on {start} and expires on {end}.",
    "The term begins {start}, continuing through {end}.",
    "The lease is operative beginning {start} and terminates {end}.",
    "Possession is granted starting {start}, with the tenancy ending {end}.",
    "The property user may occupy under this lease from {start} to {end}.",
    "The rental term opens on {start} and closes on {end}.",
]
RENTS = [
    "The opening rent is {money} per {period}.",
    "For the first year, the rental rate is {money} each {period}.",
    "The initial rental fee amounts to {money} for every {period}.",
    "Agreed base rental: {money} per {period} at commencement.",
    "Initial rent amount: {money}\nRate period: one {period}",
    "The tenant owes an initial base rental of {money} per {period}.",
    "The agreed starting rental charge is {money} a {period}.",
    "For the opening term the rent shall be {money} for each {period}.",
    "At the beginning of the tenancy, rent is fixed at {money} per {period}.",
    "The rental charge at commencement is {money} per {period}.",
    "The user pays a starting rent of {money} for each {period}.",
    "Rent begins at a rate of {money} per {period}.",
]


def written_date(value: date, style: str) -> str:
    month = MONTHS[value.month - 1]
    day = value.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return {
        "iso": value.isoformat(),
        "month_first": f"{month} {day}, {value.year}",
        "day_first": f"{day} {month} {value.year}",
        "ordinal": f"{month} {day}{suffix}, {value.year}",
        "abbreviated": f"{month[:3]} {day}, {value.year}",
    }[style]


def diverse_example(root: Path, split: str, family: int, index: int) -> dict:
    rng = random.Random(73000 + family * 1000 + index)
    template = family - 20
    document_id = f"{split}-f{family:02d}-{index:03d}"
    words = [
        "Alder",
        "Fern",
        "Willow",
        "Stone",
        "Silver",
        "Elm",
        "River",
        "Birch",
        "Laurel",
        "West",
    ]
    tenant = f"{rng.choice(words)} {rng.choice(words)} {rng.choice(['Trading Co., Ltd.', 'Industries, Inc.', 'Services LLC', 'Research Ltd.'])}"
    landlord = f"{rng.choice(words)} {rng.choice(['Estates', 'Holdings', 'Investments'])} {rng.choice(['Co., Ltd.', 'Corporation', 'L.L.C.', 'Limited'])}"
    address = f"{rng.randint(1, 999)} {rng.choice(words)} {rng.choice(['Avenue', 'Road', 'Boulevard'])}, Unit {rng.randint(1, 40)}, {rng.choice(['Rochester', 'Kingston', 'Windsor', 'Fairview'])}"
    first = date(rng.randint(1995, 2030), rng.randint(1, 12), rng.randint(1, 28))
    last = date(first.year + rng.randint(1, 8), first.month, first.day)
    style = DATE_STYLES[index % len(DATE_STYLES)]
    start, end = written_date(first, style), written_date(last, style)
    amount = rng.randint(500, 180000) + rng.choice([0, 0.25, 0.5, 0.75])
    period = rng.choice(["month", "year"])
    currency, expression, kind = rng.choice(
        [
            ("CNY", "RMB", "named"),
            ("CNY", "Chinese yuan", "named"),
            ("USD", "US dollars", "named"),
            ("GBP", "pounds sterling", "named"),
            ("TWD", "New Taiwan dollars", "named"),
            ("EUR", "EUR", "iso"),
            ("USD", "USD", "iso"),
            (None, "$", "ambiguous"),
            (None, "¥", "ambiguous"),
        ]
    )
    money = f"{expression} {amount:,.2f}"
    expected = {
        "tenant": tenant,
        "landlord": landlord,
        "premises_address": address,
        "commencement_date": first.isoformat(),
        "expiration_date": last.isoformat(),
        "base_rent": {"amount": amount, "currency": currency, "period": period},
    }
    clauses = [
        PARTIES[template].format(tenant=tenant, landlord=landlord),
        LOCATIONS[template].format(address=address),
    ]
    if index % 17 == 0:
        expected["commencement_date"] = None
        clauses.append(
            f"The commencement entry reads 04/05/{first.year}; no date convention is supplied. The tenancy ends {end}."
        )
    elif index % 19 == 0:
        expected["expiration_date"] = None
        clauses.append(f"The tenancy starts on {start}. No expiration date has been agreed.")
    else:
        clauses.append(TERMS[template].format(start=start, end=end))
    collection_differs = period == "month" and index % 2 == 0
    if index % 13 == 0:
        expected["base_rent"] = None
        clauses.append(
            "The base rental fee has not been fixed and will be agreed in a separate writing."
        )
        collection_differs = False
    else:
        clauses.append(RENTS[template].format(money=money, period=period))
        if collection_differs:
            clauses.append(
                "Rent is collected quarterly in advance; the quoted rate remains the monthly rate."
            )
        if index % 3 == 0:
            clauses.append(
                f"Beginning in the second year, the rent increases to {expression} {amount * 1.1:,.2f} per {period}."
            )
    clauses.extend(
        [
            f"Refundable security deposit: {expression} {amount * 2:,.2f}.",
            f"This agreement was signed on {written_date(date(first.year - 1, 12, 1), style)}. The signature date does not set the term.",
            "The occupier pays utilities separately. Repairs require the owner's approval. Any renewal requires a new agreement.",
        ]
    )
    rng.shuffle(clauses)
    pdf = f"pdfs/{document_id}.pdf"
    render_pdf(root / pdf, ["PROPERTY LEASE AGREEMENT", *clauses], family)
    Lease.model_validate(expected)
    return {
        "id": document_id,
        "family": f"diverse-lease-{family}",
        "split": split,
        "pdf": pdf,
        "text": read_pdf(root / pdf),
        "expected": expected,
        "source": "original-synthetic",
        "license": "Apache-2.0",
        "source_clauses": clauses,
        "features": {
            "date_style": style,
            "indirect_roles": template in {0, 1, 2, 3, 6, 7, 8, 9, 10, 11},
            "currency_kind": kind,
            "collection_differs": collection_differs,
        },
    }


def v3_examples(root: Path) -> list[dict]:
    # Rehearsal: retain the original training split, never its validation/test rows.
    rows = [
        make_example(root, "train", family, index) for family in range(8) for index in range(25)
    ]
    rows.extend(
        diverse_example(root, split, family, index)
        for split, families in FAMILIES.items()
        for family in families
        for index in range(50)
    )
    return rows
