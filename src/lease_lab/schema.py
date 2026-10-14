"""Explicit facts only; absence is represented by null."""

import json
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Rent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    amount: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None
    period: Literal["month", "year"] | None


class Lease(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tenant: str | None
    landlord: str | None
    premises_address: str | None
    commencement_date: str | None
    expiration_date: str | None
    base_rent: Rent | None

    @field_validator("commencement_date", "expiration_date")
    @classmethod
    def explicit_iso_date(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 10 or date.fromisoformat(value).isoformat() != value
        ):
            raise ValueError("Dates must be unambiguous YYYY-MM-DD dates")
        return value

    @field_validator("tenant", "landlord", "premises_address")
    @classmethod
    def nonempty_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Use null for missing values, not an empty string")
        return value


def parse_prediction(text: str) -> Lease:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return Lease.model_validate(json.loads(text, object_pairs_hook=unique_keys))
