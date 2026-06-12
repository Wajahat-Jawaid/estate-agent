#!/usr/bin/env python3
"""
generate_listings.py
====================
Generates TOTAL_PROPERTIES (700) listings into data/listings.json using only
constants from data/listings_config.py, then rebuilds ChromaDB via ingest.py.

Usage: python generate_listings.py
"""

import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.listings_config import (
    TOTAL_PROPERTIES, PENTHOUSE_MULTIPLIER,
    DEMO_AREA_COUNTS, NON_DEMO_AREA_WEIGHTS,
    SUBAREAS, HOUSE_RATES,
    APARTMENT_RATES_BY_AREA, APARTMENT_RATE_DEFAULT_BY_PARENT, APARTMENT_RATE_FALLBACK,
    PLOT_PROFILES, AREA_TO_PLOT_PROFILE,
    APARTMENT_SQFT_BY_BED,
    TYPE_WEIGHTS, AREA_PREMIUM_LEVEL, APT_BED_SKEW,
    house_beds_for_area,
    AMENITY_PROBABILITIES,
    AGENT_NAMES, IMAGE_SEEDS,
)

random.seed(42)

MAP_URL = "https://maps.app.goo.gl/UG71yexnwbADR7dt8"
MOBILE_PREFIXES = [
    "0300", "0301", "0302", "0303", "0311", "0312", "0313", "0314", "0315",
    "0321", "0322", "0323", "0324", "0325", "0330", "0331", "0332", "0333",
    "0334", "0335", "0341", "0342", "0343", "0344", "0345",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(weights_dict: dict):
    keys = [k for k, v in weights_dict.items() if v > 0]
    wts = [weights_dict[k] for k in keys]
    return random.choices(keys, weights=wts, k=1)[0]


def price_to_str(lacs: int) -> str:
    crore, lac = lacs // 100, lacs % 100
    if crore > 0 and lac > 0:
        return f"{crore} crore {lac} lac"
    if crore > 0:
        return f"{crore} crore"
    return f"{lac} lac"


def price_str_to_lacs(price_str: str) -> int:
    s = price_str.lower()
    crore_m = re.search(r"(\d+)\s*crore", s)
    lac_m = re.search(r"(\d+)\s*lac", s)
    return (int(crore_m.group(1)) if crore_m else 0) * 100 + (int(lac_m.group(1)) if lac_m else 0)


def get_premium(area: str) -> str:
    return AREA_PREMIUM_LEVEL.get(area, "BUDGET")


def get_plot_profile(area: str) -> dict:
    return PLOT_PROFILES[AREA_TO_PLOT_PROFILE.get(area, "BUDGET")]


def get_house_rate(subarea: str, area: str) -> int:
    return HOUSE_RATES.get(subarea) or HOUSE_RATES.get(area) or 15000


def get_apt_rate(subarea: str, area: str) -> int:
    if subarea in APARTMENT_RATES_BY_AREA:
        return APARTMENT_RATES_BY_AREA[subarea]
    if area in APARTMENT_RATES_BY_AREA:
        return APARTMENT_RATES_BY_AREA[area]
    return APARTMENT_RATE_DEFAULT_BY_PARENT.get(area, APARTMENT_RATE_FALLBACK)


def gen_images() -> tuple[str, list[str]]:
    def img():
        return f"https://picsum.photos/seed/{random.choice(IMAGE_SEEDS)}{random.randint(1, 20)}/800/600"
    return img(), [img() for _ in range(10)]


def gen_contact() -> str:
    return f"{random.choice(MOBILE_PREFIXES)}-{random.randint(1000000, 9999999)}"


def gen_amenities(premium: str) -> list[str]:
    return [a for a, p in AMENITY_PROBABILITIES.items() if random.random() < p.get(premium, 0)]


def gen_bathrooms(bedrooms: int) -> int:
    r = random.random()
    if r < 0.60:
        return bedrooms
    if r < 0.85:
        return bedrooms + 1
    return bedrooms + 2


# ---------------------------------------------------------------------------
# Area count distribution
# ---------------------------------------------------------------------------

def build_area_counts() -> dict[str, int]:
    counts: dict[str, int] = dict(DEMO_AREA_COUNTS)          # 350 exact
    remaining = TOTAL_PROPERTIES - sum(counts.values())       # 350 remaining
    total_w = sum(NON_DEMO_AREA_WEIGHTS.values())

    non_demo: dict[str, int] = {
        area: round(remaining * w / total_w)
        for area, w in NON_DEMO_AREA_WEIGHTS.items()
    }

    # Absorb rounding residual into the highest-weight non-demo area
    diff = TOTAL_PROPERTIES - sum(counts.values()) - sum(non_demo.values())
    if diff != 0:
        non_demo[max(NON_DEMO_AREA_WEIGHTS, key=NON_DEMO_AREA_WEIGHTS.get)] += diff

    counts.update(non_demo)
    assert sum(counts.values()) == TOTAL_PROPERTIES, f"Count mismatch: {sum(counts.values())}"
    return counts


# ---------------------------------------------------------------------------
# Single-property generator
# ---------------------------------------------------------------------------

def gen_property(prop_id: int, area: str) -> dict:
    subs = SUBAREAS.get(area)
    subarea = random.choice(subs) if subs else area

    premium = get_premium(area)
    prop_type = weighted_choice(TYPE_WEIGHTS[premium])

    area_sqft = None
    if prop_type in ("house", "upper portion", "lower portion"):
        area_sqyd: int = weighted_choice(get_plot_profile(area))
        bed_lo, bed_hi = house_beds_for_area(area_sqyd)
        bedrooms = random.randint(bed_lo, bed_hi)
        rate = get_house_rate(subarea, area)
        price_lacs = round(area_sqyd * 9 * rate / 100_000)
    else:  # apartment / penthouse
        bedrooms = weighted_choice(APT_BED_SKEW[premium])
        sqft_lo, sqft_hi = APARTMENT_SQFT_BY_BED[bedrooms]
        area_sqft = random.randint(sqft_lo, sqft_hi)
        area_sqyd = round(area_sqft / 9)
        rate = get_apt_rate(subarea, area)
        if prop_type == "penthouse":
            rate = round(rate * PENTHOUSE_MULTIPLIER)
        price_lacs = round(area_sqft * rate / 100_000)

    image_url, images = gen_images()

    prop = {
        "id": prop_id,
        "title": f"{bedrooms} bed {prop_type} in {subarea}",
        "type": prop_type,
        "bedrooms": bedrooms,
        "bathrooms": gen_bathrooms(bedrooms),
        "area_sqyd": area_sqyd,
        "price": price_to_str(price_lacs),
        "price_numeric": price_lacs,
        "location": f"{subarea}, Karachi",
        "agent": random.choice(AGENT_NAMES),
        "contact": gen_contact(),
        "image_url": image_url,
        "images": images,
        "map_url": MAP_URL,
        "amenities": gen_amenities(premium),
    }
    if area_sqft is not None:
        prop["area_sqft"] = area_sqft
    return prop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    area_counts = build_area_counts()

    listings = []
    prop_id = 1
    for area, count in area_counts.items():
        for _ in range(count):
            listings.append(gen_property(prop_id, area))
            prop_id += 1

    random.shuffle(listings)
    for i, p in enumerate(listings, 1):
        p["id"] = i

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "listings.json")
    with open(out_path, "w") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)
    print(f"Written {len(listings)} listings → {out_path}\n")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    subarea_to_parent: dict[str, str] = {
        s: parent for parent, subs in SUBAREAS.items() for s in subs
    }

    # in the stats defaultdict, replace "price_sum": 0 with "prices": []
    stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "prices": [], "sqyd_sum": 0, "types": defaultdict(int)}
    )
    for p in listings:
        loc = p["location"].split(",")[0].strip()
        parent = subarea_to_parent.get(loc, loc)
        stats[parent]["count"] += 1
        stats[parent]["prices"].append(price_str_to_lacs(p["price"]))
        stats[parent]["sqyd_sum"] += p["area_sqyd"]
        stats[parent]["types"][p["type"]] += 1

    # in the print loop, replace s['price_sum']/n with the median:
    med = statistics.median(s["prices"])
    
    print(f"{'Area':<32} {'N':>4}  {'Med Lac':>9}  {'Avg Sqyd':>9}  Type split")
    print("-" * 105)
    for area in sorted(stats, key=lambda a: stats[a]["count"], reverse=True):
        s = stats[area]
        n = s["count"]
        med = statistics.median(s["prices"])
        split = "  ".join(f"{t}:{c}" for t, c in sorted(s["types"].items(), key=lambda x: -x[1]))
        print(f"{area:<32} {n:>4}  {med:>9.0f}  {s['sqyd_sum']/n:>9.1f}  {split}")

    print(f"\nTotal: {sum(s['count'] for s in stats.values())}")

    type_totals: dict[str, int] = defaultdict(int)
    for p in listings:
        type_totals[p["type"]] += 1
    print("Type split: " + "  ".join(f"{t}:{c}" for t, c in sorted(type_totals.items(), key=lambda x: -x[1])))

    # ------------------------------------------------------------------
    # Apartment table (apartments only — penthouses excluded to preserve
    # area ranking; penthouse 1.4× multiplier skews the average when an
    # area draws proportionally more penthouses than its ranked neighbor)
    # ------------------------------------------------------------------
    apt_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "price_sum": 0, "sqft_sum": 0})
    for p in listings:
        if p["type"] == "apartment" and "area_sqft" in p:
            loc = p["location"].split(",")[0].strip()
            parent = subarea_to_parent.get(loc, loc)
            apt_stats[parent]["count"] += 1
            apt_stats[parent]["price_sum"] += p["price_numeric"]
            apt_stats[parent]["sqft_sum"] += p["area_sqft"]

    print(f"\n--- APARTMENTS (penthouses excluded) ---")
    print(f"{'Area':<32} {'N':>4}  {'Avg Lac':>9}  {'Avg Sqft':>9}  {'Avg PKR/sqft':>13}")
    print("-" * 75)
    for area in sorted(apt_stats, key=lambda a: apt_stats[a]["price_sum"] / apt_stats[a]["sqft_sum"] * 100_000, reverse=True):
        s = apt_stats[area]
        n = s["count"]
        avg_lac = s["price_sum"] / n
        avg_sqft = s["sqft_sum"] / n
        avg_per_sqft = s["price_sum"] * 100_000 / s["sqft_sum"]
        print(f"{area:<32} {n:>4}  {avg_lac:>9.0f}  {avg_sqft:>9.0f}  {avg_per_sqft:>13,.0f}")

    # ------------------------------------------------------------------
    # House table (house + upper/lower portion, sorted by avg PKR/sqft desc)
    # ------------------------------------------------------------------
    house_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "price_sum": 0, "sqft_sum": 0})
    for p in listings:
        if p["type"] in ("house", "upper portion", "lower portion"):
            loc = p["location"].split(",")[0].strip()
            parent = subarea_to_parent.get(loc, loc)
            sqft = p["area_sqyd"] * 9
            house_stats[parent]["count"] += 1
            house_stats[parent]["price_sum"] += p["price_numeric"]
            house_stats[parent]["sqft_sum"] += sqft

    print(f"\n--- HOUSES (incl. upper/lower portions) ---")
    print(f"{'Area':<32} {'N':>4}  {'Avg Lac':>9}  {'Avg Sqft':>9}  {'Avg PKR/sqft':>13}")
    print("-" * 75)
    for area in sorted(house_stats, key=lambda a: house_stats[a]["price_sum"] / house_stats[a]["sqft_sum"] * 100_000, reverse=True):
        s = house_stats[area]
        n = s["count"]
        avg_lac = s["price_sum"] / n
        avg_sqft = s["sqft_sum"] / n
        avg_per_sqft = s["price_sum"] * 100_000 / s["sqft_sum"]
        print(f"{area:<32} {n:>4}  {avg_lac:>9.0f}  {avg_sqft:>9.0f}  {avg_per_sqft:>13,.0f}")

    # ------------------------------------------------------------------
    # Rebuild ChromaDB
    # ------------------------------------------------------------------
    print("\nRebuilding ChromaDB…")
    base = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, os.path.join(base, "ingest.py")], cwd=base)
    if result.returncode != 0:
        print("ERROR: ingest.py failed")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
