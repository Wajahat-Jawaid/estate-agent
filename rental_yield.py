"""
Area-level rental yield estimates for Karachi properties.

Yields are derived from the property's per-sqft rate (looked up from
listings_config.HOUSE_RATES / APARTMENT_RATES_BY_AREA) via a rate→yield
curve that reflects Karachi market behaviour: premium areas command higher
prices but thinner gross yields; mid-market areas yield higher; outer/
low-demand areas roll back slightly despite low prices (thin tenant pool).

All outputs are area-level estimates — NOT property-specific measurements.
Always surface the disclaimer to users.
"""
from __future__ import annotations
import logging
from data.listings_config import (
    HOUSE_RATES,
    APARTMENT_RATES_BY_AREA,
    APARTMENT_RATE_DEFAULT_BY_PARENT,
    APARTMENT_RATE_FALLBACK,
    SUBAREAS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate → yield curve  (house yields; apartments get +0.5pp on both ends)
# ---------------------------------------------------------------------------
# Each entry: (min_rate_inclusive, (yield_low%, yield_high%))
# Evaluated top-to-bottom; first match wins.
_YIELD_BANDS: list[tuple[int, tuple[float, float]]] = [
    (38_000, (4.5, 6.0)),   # ultra-premium: DHA 5-8, Clifton, Emaar
    (30_000, (5.0, 6.5)),   # premium: DHA 1-4, PECHS, Askari, Naya Naz
    (24_000, (5.5, 7.0)),   # upper-mid: Gulshan, Johar, FB Area, North Naz
    (18_000, (6.5, 8.0)),   # mid: Nazimabad, Scheme 33, Malir Cantt, Tariq
    (12_000, (7.0, 8.5)),   # budget: Korangi, Landhi, Surjani, Orangi
    (     0, (6.0, 7.5)),   # outer rolloff: Gadap, Bin Qasim, Hub River —
                            #   thin tenant pool caps effective yield
]
_APT_BOOST = 0.5            # apartments yield higher than equivalent houses

# ---------------------------------------------------------------------------
# Reverse-map: sub-area → parent area  (built once at import time)
# ---------------------------------------------------------------------------
_SUBAREA_TO_PARENT: dict[str, str] = {
    sub: parent for parent, subs in SUBAREAS.items() for sub in subs
}

_APARTMENT_TYPES = {"apartment", "penthouse"}


def _lookup_rate(subarea: str, prop_type: str) -> int:
    """Return the per-sqft PKR rate for *subarea* from the config tables."""
    parent = _SUBAREA_TO_PARENT.get(subarea, subarea)
    is_apt = prop_type.lower().strip() in _APARTMENT_TYPES

    if is_apt:
        if subarea in APARTMENT_RATES_BY_AREA:
            return APARTMENT_RATES_BY_AREA[subarea]
        if parent in APARTMENT_RATES_BY_AREA:
            return APARTMENT_RATES_BY_AREA[parent]
        if parent in APARTMENT_RATE_DEFAULT_BY_PARENT:
            return APARTMENT_RATE_DEFAULT_BY_PARENT[parent]
        return APARTMENT_RATE_FALLBACK
    else:
        if subarea in HOUSE_RATES:
            return HOUSE_RATES[subarea]
        if parent in HOUSE_RATES:
            return HOUSE_RATES[parent]
        logger.warning("rental_yield: no house rate found for %r (parent %r) — using 15000", subarea, parent)
        return 15_000


def _rate_to_yield(rate: int, is_apt: bool) -> tuple[float, float]:
    for threshold, (lo, hi) in _YIELD_BANDS:
        if rate >= threshold:
            if is_apt:
                return lo + _APT_BOOST, hi + _APT_BOOST
            return lo, hi
    # unreachable (threshold 0 always matches) but satisfies type checkers
    return _YIELD_BANDS[-1][1]


def estimate_rent(price_numeric: int, location: str, property_type: str) -> dict:
    """Return an area-level rental yield estimate for a property.

    Args:
        price_numeric: Price in lacs (as stored in Chroma metadata).
        location:      Location string from metadata, e.g. "DHA Phase 6, Karachi".
        property_type: Type string from metadata (house / apartment / penthouse …).

    Returns:
        {
            "rate":              42000,          # per-sqft rate used
            "yield_low":          4.5,
            "yield_high":         6.0,
            "monthly_rent_low":   ...,           # PKR
            "monthly_rent_high":  ...,           # PKR
            "disclaimer":         "Area-level estimate …"
        }
    """
    subarea = location.split(",")[0].strip()
    is_apt = property_type.lower().strip() in _APARTMENT_TYPES

    rate = _lookup_rate(subarea, property_type)
    y_low, y_high = _rate_to_yield(rate, is_apt)

    price_pkr = price_numeric * 100_000
    monthly_low  = int(price_pkr * (y_low  / 100) / 12)
    monthly_high = int(price_pkr * (y_high / 100) / 12)

    return {
        "rate": rate,
        "yield_low": y_low,
        "yield_high": y_high,
        "monthly_rent_low": monthly_low,
        "monthly_rent_high": monthly_high,
        "disclaimer": (
            f"Area-level estimate based on typical {subarea} rents "
            f"({y_low}–{y_high}% gross yield) — not a property-specific measurement."
        ),
    }
