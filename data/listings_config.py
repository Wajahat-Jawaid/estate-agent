"""
listings_config.py
==================
Single source of truth for regenerating Theorrem's estate-agent property dataset.

ALL numbers here are deliberately set and verified (Zameen May/June 2026 anchors
+ WJ's local market knowledge). Edit a value and re-run generate_listings.py to
reshape the dataset — never hand-edit listings.json.

UNITS:
  - House rates:      PKR per square FOOT. Price = area_sqyd * 9 * rate.
  - Apartment rates:  PKR per square FOOT. Price = area_sqft * rate.
                      area_sqyd stored = round(area_sqft / 9) for schema uniformity.
  - Penthouse:        apartment rate * PENTHOUSE_MULTIPLIER.
  - price_numeric:    in LACS, rounded to nearest lac.

BLOCK SPREAD LOGIC:
  Block rates for areas with sub-areas are spread evenly within a range that is
  the tighter of (midpoint ± spread%) and [lower_neighbor_midpoint, upper_neighbor_midpoint]
  from the master area ranking, so no block can exceed the midpoint of the area
  ranked immediately above, and no block can fall below the midpoint of the area
  ranked immediately below (Bahria Town exempt; North Karachi used as effective
  lower bound for FB Area).
"""

from __future__ import annotations

TOTAL_PROPERTIES = 700
PENTHOUSE_MULTIPLIER = 1.4

# ---------------------------------------------------------------------------
# 1. AREA DISTRIBUTION  (how many properties per area)
# ---------------------------------------------------------------------------
DEMO_AREA_COUNTS = {
    "DHA":              70,
    "Gulshan-e-Iqbal":  70,
    "Gulistan-e-Johar": 52,
    "FB Area":          35,
    "PECHS":            35,
    "Nazimabad":        28,
    "Scheme 33":        25,
    "Clifton":          17,
    "Askari":           18,
}  # sums to 350

NON_DEMO_AREA_WEIGHTS = {
    "North Nazimabad":      10,
    "Bahria Town":          12,
    "Naya Nazimabad":        6,
    "Malir":                 5,
    "Korangi":               6,
    "Landhi":                5,
    "North Karachi":         6,
    "New Karachi":           4,
    "KDA Scheme 1":          3,
    "Model Colony":          3,
    "Gulshan-e-Maymar":      4,
    "Gulshan-e-Hadeed":      3,
    "Shah Faisal Colony":    3,
    "Tariq Road":            3,
    "Safoora Goth":          3,
    "Liaquatabad":           4,
    "Orangi Town":           3,
    "Baldia Town":           3,
    "Surjani Town":          4,
    "Airport Area":          2,
    "Gadap Town":            2,
    "Bin Qasim":             2,
    "Steel Town":            2,
}

# ---------------------------------------------------------------------------
# 2. SUB-AREA (block/phase) lists for granular areas
# ---------------------------------------------------------------------------
SUBAREAS = {
    "DHA": [
        "DHA Phase 8", "DHA Phase 6", "DHA Phase 5", "DHA Phase 7",
        "DHA Phase 4", "DHA Phase 2", "DHA Phase 1", "DHA Phase 3",
    ],
    "PECHS": ["PECHS Block 2", "PECHS Block 6", "PECHS Block 3"],
    "Gulshan-e-Iqbal": [
        "Gulshan-e-Iqbal Block 2", "Gulshan-e-Iqbal Block 3", "Gulshan-e-Iqbal Block 4",
        "Gulshan-e-Iqbal Block 7", "Gulshan-e-Iqbal Block 13-D", "Gulshan-e-Iqbal Block 6",
        "Gulshan-e-Iqbal Block 17", "Gulshan-e-Iqbal Block 10-A", "Gulshan-e-Iqbal Block 1",
        "Gulshan-e-Iqbal Block 5", "Gulshan-e-Iqbal Block 13-C",
    ],
    "Askari": ["Askari 5", "Askari 4", "Askari 3", "Askari 2", "Askari 1"],
    "Gulistan-e-Johar": [
        "Gulistan-e-Johar Block 7", "Gulistan-e-Johar Block 8", "Gulistan-e-Johar Block 9",
        "Gulistan-e-Johar Block 12", "Gulistan-e-Johar Block 13", "Gulistan-e-Johar Block 14",
        "Gulistan-e-Johar Block 15", "Gulistan-e-Johar Block 17", "Gulistan-e-Johar Block 18",
        "Gulistan-e-Johar Block 3", "Gulistan-e-Johar Block 10", "Gulistan-e-Johar Block 19",
        "Gulistan-e-Johar Block 1",
    ],
    "FB Area": [
        "FB Area Block 6", "FB Area Block 7", "FB Area Block 8", "FB Area Block 10",
        "FB Area Block 11", "FB Area Block 12", "FB Area Block 13", "FB Area Block 14",
        "FB Area Block 15", "FB Area Block 16", "FB Area Block 17", "FB Area Block 18",
        "FB Area Block 5",
    ],
    "Scheme 33": [
        "Saadi Town", "Saadi Garden", "PCSIR Society", "Teachers Society",
        "Memon Nagar", "Gulzar-e-Hijri", "Sector 35-A",
    ],
    "Clifton": [
        "Clifton Block 2", "Clifton Block 4", "Clifton Block 5", "Clifton Block 8",
        "Clifton Block 1", "Clifton Block 3", "Clifton Block 7", "Clifton Block 9",
    ],
    "Bahria Town": [
        # High-end precincts
        "Bahria Precinct 1", "Bahria Precinct 8", "Bahria Precinct 17",
        # Mid precincts
        "Bahria Precinct 10", "Bahria Precinct 11", "Bahria Precinct 12",
        "Bahria Precinct 15", "Bahria Precinct 27", "Bahria Precinct 28", "Bahria Precinct 31",
        # Low precincts
        "Bahria Precinct 23", "Bahria Precinct 24", "Bahria Precinct 25", "Bahria Precinct 26",
        "Bahria Precinct 32", "Bahria Precinct 42", "Bahria Precinct 43", "Bahria Precinct 44",
        "Bahria Precinct 45", "Bahria Precinct 61", "Bahria Precinct 62", "Bahria Precinct 63",
    ],
}

# ---------------------------------------------------------------------------
# 3. HOUSE RATES  (PKR per sq ft)  — price = area_sqyd * 9 * rate
#    Block rates spread within [lower_neighbor_mid, upper_neighbor_mid] per ranking.
# ---------------------------------------------------------------------------
HOUSE_RATES = {
    # Clifton (rank 1): clamped [DHA_mid=39700, +inf], spread ±6% around 40400
    "Clifton Block 2": 42800, "Clifton Block 4": 42400, "Clifton Block 5": 41900,
    "Clifton Block 8": 41500, "Clifton Block 1": 41000, "Clifton Block 3": 40600,
    "Clifton Block 7": 40100, "Clifton Block 9": 39700,
    # DHA (rank 2): clamped [PECHS_mid=39100, Clifton_mid=40400], spread ±10% around 39700
    "DHA Phase 8": 40400, "DHA Phase 6": 40200, "DHA Phase 5": 40000,
    "DHA Phase 7": 39800, "DHA Phase 4": 39700, "DHA Phase 2": 39500,
    "DHA Phase 1": 39300, "DHA Phase 3": 39100,
    # PECHS (rank 3): clamped [KDA_mid=38400, DHA_mid=39700], spread ±6% around 39100
    "PECHS Block 2": 39700, "PECHS Block 6": 39000, "PECHS Block 3": 38400,
    # Askari (individual rates from ranking)
    "Askari 5": 32000, "Askari 4": 31700, "Askari 1": 31400,
    "Askari 3": 31000, "Askari 2": 30700,
    # Gulshan-e-Iqbal (rank 8): clamped [Askari1_mid=35200, Askari4_mid=36500], spread ±6% around 35800
    "Gulshan-e-Iqbal Block 2": 36500, "Gulshan-e-Iqbal Block 3": 36400,
    "Gulshan-e-Iqbal Block 4": 36200, "Gulshan-e-Iqbal Block 7": 36100,
    "Gulshan-e-Iqbal Block 13-D": 36000, "Gulshan-e-Iqbal Block 6": 35800,
    "Gulshan-e-Iqbal Block 17": 35700, "Gulshan-e-Iqbal Block 10-A": 35600,
    "Gulshan-e-Iqbal Block 1": 35500, "Gulshan-e-Iqbal Block 5": 35300,
    "Gulshan-e-Iqbal Block 13-C": 35200,
    # Gulistan-e-Johar (rank 12): clamped [NorthNaz_mid=32600, Askari2_mid=33900], spread ±6% around 33200
    "Gulistan-e-Johar Block 7": 33900, "Gulistan-e-Johar Block 8": 33800,
    "Gulistan-e-Johar Block 9": 33700, "Gulistan-e-Johar Block 12": 33600,
    "Gulistan-e-Johar Block 13": 33500, "Gulistan-e-Johar Block 14": 33400,
    "Gulistan-e-Johar Block 15": 33200, "Gulistan-e-Johar Block 17": 33100,
    "Gulistan-e-Johar Block 18": 33000, "Gulistan-e-Johar Block 3": 32900,
    "Gulistan-e-Johar Block 10": 32800, "Gulistan-e-Johar Block 19": 32700,
    "Gulistan-e-Johar Block 1": 32600,
    # FB Area (rank 20): clamped [NorthKarachi_mid=26500, PCSIRSoc_mid=28600], spread ±6% around 28000
    "FB Area Block 6": 28600, "FB Area Block 7": 28400, "FB Area Block 8": 28200,
    "FB Area Block 10": 28100, "FB Area Block 11": 27900, "FB Area Block 12": 27700,
    "FB Area Block 13": 27600, "FB Area Block 14": 27400, "FB Area Block 15": 27200,
    "FB Area Block 16": 27000, "FB Area Block 17": 26800, "FB Area Block 18": 26700,
    "FB Area Block 5": 26500,
    # Scheme 33 sub-areas (individual rates from ranking)
    "Teachers Society": 29300, "PCSIR Society": 28600, "Saadi Town": 25100,
    "Memon Nagar": 24000, "Gulzar-e-Hijri": 23000, "Saadi Garden": 22200,
    "Sector 35-A": 20700,
    # Single-rate areas
    "KDA Scheme 1": 38400, "Tariq Road": 37800, "North Nazimabad": 32600,
    "Naya Nazimabad": 24200, "Nazimabad": 26600,
    "North Karachi": 25500, "Gulshan-e-Maymar": 26200,
    "Malir": 24300, "Safoora Goth": 23600, "Shah Faisal Colony": 22900,
    "Liaquatabad": 21400, "Korangi": 20000, "Model Colony": 19300,
    "Baldia Town": 18500, "New Karachi": 17800, "Airport Area": 17100,
    "Landhi": 16400, "Surjani Town": 15600, "Gulshan-e-Hadeed": 14900,
    "Orangi Town": 14200, "Gadap Town": 13400, "Steel Town": 12700,
    "Bin Qasim": 12000,
    # Bahria Town precincts (exempt from ranking clamp; own tier rates)
    "Bahria Precinct 1": 22000, "Bahria Precinct 8": 22000, "Bahria Precinct 17": 22000,
    "Bahria Precinct 10": 18000, "Bahria Precinct 11": 18000, "Bahria Precinct 12": 18000,
    "Bahria Precinct 15": 18000, "Bahria Precinct 27": 18000, "Bahria Precinct 28": 18000,
    "Bahria Precinct 31": 18000,
    "Bahria Precinct 23": 15000, "Bahria Precinct 24": 15000, "Bahria Precinct 25": 15000,
    "Bahria Precinct 26": 15000, "Bahria Precinct 32": 15000, "Bahria Precinct 42": 15000,
    "Bahria Precinct 43": 15000, "Bahria Precinct 44": 15000, "Bahria Precinct 45": 15000,
    "Bahria Precinct 61": 15000, "Bahria Precinct 62": 15000, "Bahria Precinct 63": 15000,
}

# ---------------------------------------------------------------------------
# 4. APARTMENT RATES  (PKR per sq ft)  — price = area_sqft * rate
#    Block rates clamped to same neighbor midpoints as house rates above.
# ---------------------------------------------------------------------------
APARTMENT_RATES_BY_AREA = {
    # Clifton (rank 1): clamped [DHA_apt_mid=38200, +inf], spread ±6% around 39000
    "Clifton Block 2": 41300, "Clifton Block 4": 40900, "Clifton Block 5": 40400,
    "Clifton Block 8": 40000, "Clifton Block 1": 39500, "Clifton Block 3": 39100,
    "Clifton Block 7": 38600, "Clifton Block 9": 38200,
    # DHA (rank 2): clamped [PECHS_apt_mid=37400, Clifton_apt_mid=39000], spread ±10% around 38200
    "DHA Phase 8": 39000, "DHA Phase 6": 38800, "DHA Phase 5": 38500,
    "DHA Phase 7": 38300, "DHA Phase 4": 38100, "DHA Phase 2": 37900,
    "DHA Phase 1": 37600, "DHA Phase 3": 37400,
    # PECHS (rank 3): clamped [KDA_apt_mid=36600, DHA_apt_mid=39000], spread ±6% around 37400
    "PECHS Block 2": 38200, "PECHS Block 6": 37400, "PECHS Block 3": 36600,
    # Landmark towers (unchanged)
    "Emaar": 57000, "Marina Tower": 57000, "Ocean Tower": 57000,
    # Askari (individual)
    "Askari 5": 31800, "Askari 4": 31500, "Askari 1": 31000,
    "Askari 3": 30300, "Askari 2": 30100,
    # Gulshan-e-Iqbal (rank 8): clamped [Askari1_apt=32700, Askari4_apt=34300], spread ±6% around 33500
    "Gulshan-e-Iqbal Block 2": 34300, "Gulshan-e-Iqbal Block 3": 34100,
    "Gulshan-e-Iqbal Block 4": 34000, "Gulshan-e-Iqbal Block 7": 33800,
    "Gulshan-e-Iqbal Block 13-D": 33700, "Gulshan-e-Iqbal Block 6": 33500,
    "Gulshan-e-Iqbal Block 17": 33300, "Gulshan-e-Iqbal Block 10-A": 33200,
    "Gulshan-e-Iqbal Block 1": 33000, "Gulshan-e-Iqbal Block 5": 32900,
    "Gulshan-e-Iqbal Block 13-C": 32700,
    # Gulistan-e-Johar (rank 12): clamped [NorthNaz_apt=29500, Askari2_apt=31100], spread ±6% around 30300
    "Gulistan-e-Johar Block 7": 31100, "Gulistan-e-Johar Block 8": 31000,
    "Gulistan-e-Johar Block 9": 30800, "Gulistan-e-Johar Block 12": 30700,
    "Gulistan-e-Johar Block 13": 30600, "Gulistan-e-Johar Block 14": 30400,
    "Gulistan-e-Johar Block 15": 30300, "Gulistan-e-Johar Block 17": 30200,
    "Gulistan-e-Johar Block 18": 30000, "Gulistan-e-Johar Block 3": 29900,
    "Gulistan-e-Johar Block 10": 29800, "Gulistan-e-Johar Block 19": 29600,
    "Gulistan-e-Johar Block 1": 29500,
    # FB Area (rank 20): clamped [NorthKarachi_apt=23000, PCSIRSoc_apt=24800], spread ±6% around 24000
    "FB Area Block 6": 24800, "FB Area Block 7": 24600, "FB Area Block 8": 24500,
    "FB Area Block 10": 24400, "FB Area Block 11": 24200, "FB Area Block 12": 24000,
    "FB Area Block 13": 23900, "FB Area Block 14": 23800, "FB Area Block 15": 23600,
    "FB Area Block 16": 23400, "FB Area Block 17": 23300, "FB Area Block 18": 23200,
    "FB Area Block 5": 23000,
    # Single-rate areas
    "KDA Scheme 1": 36600, "Tariq Road": 35800, "North Nazimabad": 29500,
    "Naya Nazimabad": 21400, "Nazimabad": 23100,
    "North Karachi": 23000, "Gulshan-e-Maymar": 22500,
    "Malir": 21600, "Safoora Goth": 21100, "Shah Faisal Colony": 20600,
    "Liaquatabad": 19700, "Korangi": 18700, "Model Colony": 18200,
    "Baldia Town": 17800, "New Karachi": 17300, "Airport Area": 16800,
    "Landhi": 16300, "Surjani Town": 15600, "Gulshan-e-Hadeed": 14900,
    "Orangi Town": 14200, "Gadap Town": 13400, "Steel Town": 12700,
    "Bin Qasim": 12000,
    # Scheme 33 sub-areas (individual)
    "Teachers Society": 25600, "PCSIR Society": 24800, "Saadi Town": 22100,
    "Memon Nagar": 21000, "Gulzar-e-Hijri": 20500, "Saadi Garden": 20200,
    "Sector 35-A": 19200,
    # Bahria Town precincts (own tier rates)
    "Bahria Precinct 1": 20000, "Bahria Precinct 8": 20000, "Bahria Precinct 17": 20000,
    "Bahria Precinct 10": 16000, "Bahria Precinct 11": 16000, "Bahria Precinct 12": 16000,
    "Bahria Precinct 15": 16000, "Bahria Precinct 27": 16000, "Bahria Precinct 28": 16000,
    "Bahria Precinct 31": 16000,
    "Bahria Precinct 23": 13000, "Bahria Precinct 24": 13000, "Bahria Precinct 25": 13000,
    "Bahria Precinct 26": 13000, "Bahria Precinct 32": 13000, "Bahria Precinct 42": 13000,
    "Bahria Precinct 43": 13000, "Bahria Precinct 44": 13000, "Bahria Precinct 45": 13000,
    "Bahria Precinct 61": 13000, "Bahria Precinct 62": 13000, "Bahria Precinct 63": 13000,
}
# Fallback for blocks not individually listed (Gulshan/Johar/FB now all explicit above):
APARTMENT_RATE_DEFAULT_BY_PARENT = {
    "Gulshan-e-Iqbal": 33500,
    "Gulistan-e-Johar": 30300,
    "FB Area": 24000,
    "Scheme 33": 21000,
}
APARTMENT_RATE_FALLBACK = 13500

# ---------------------------------------------------------------------------
# 5. HOUSE PLOT-SIZE DISTRIBUTION  (sq yd)  — weighted draw per area
# ---------------------------------------------------------------------------
PLOT_SIZES = [80, 100, 120, 150, 200, 240, 250, 300, 350, 400, 600, 1000, 2000, 4000]

PLOT_PROFILES = {
    "DHA_CLIFTON": {600: 30, 1000: 30, 400: 12, 350: 8, 300: 6, 240: 4,
                    2000: 6, 4000: 3, 200: 1},
    "PECHS_ASKARI": {400: 22, 600: 25, 1000: 22, 300: 10, 240: 8, 200: 6,
                     2000: 5, 150: 2},
    "GULSHAN": {200: 16, 240: 18, 300: 16, 400: 16, 600: 14, 150: 6, 120: 4,
                1000: 6, 2000: 4},
    "JOHAR": {120: 12, 200: 18, 240: 18, 300: 16, 400: 14, 100: 6, 80: 4,
              600: 6, 1000: 4, 2000: 2},
    "FB_SCHEME33": {120: 16, 200: 20, 240: 18, 300: 16, 80: 8, 100: 8,
                    400: 8, 600: 4, 1000: 2},
    "BUDGET": {80: 22, 100: 22, 120: 20, 200: 18, 150: 8, 240: 6, 400: 3, 1000: 1},
    "BAHRIA": {240: 22, 250: 20, 300: 22, 350: 12, 400: 12, 200: 6, 500: 0, 600: 6},
}

AREA_TO_PLOT_PROFILE = {
    "DHA": "DHA_CLIFTON", "Clifton": "DHA_CLIFTON",
    "PECHS": "PECHS_ASKARI", "Askari": "PECHS_ASKARI",
    "Gulshan-e-Iqbal": "GULSHAN",
    "Gulistan-e-Johar": "JOHAR",
    "FB Area": "FB_SCHEME33", "Scheme 33": "FB_SCHEME33",
    "Naya Nazimabad": "GULSHAN", "North Nazimabad": "GULSHAN",
    "Nazimabad": "FB_SCHEME33", "KDA Scheme 1": "PECHS_ASKARI",
    "Malir": "FB_SCHEME33",
    "Model Colony": "FB_SCHEME33",
    "Gulshan-e-Maymar": "FB_SCHEME33", "Gulshan-e-Hadeed": "BUDGET",
    "Shah Faisal Colony": "BUDGET", "Tariq Road": "FB_SCHEME33",
    "Safoora Goth": "FB_SCHEME33", "North Karachi": "BUDGET",
    "New Karachi": "BUDGET", "Korangi": "BUDGET", "Landhi": "BUDGET",
    "Liaquatabad": "BUDGET", "Orangi Town": "BUDGET", "Baldia Town": "BUDGET",
    "Surjani Town": "BUDGET", "Airport Area": "BUDGET", "Gadap Town": "BUDGET",
    "Bin Qasim": "BUDGET", "Steel Town": "BUDGET",
    "Bahria Town": "BAHRIA",
}

# ---------------------------------------------------------------------------
# 6. APARTMENT AREA (sq ft) by bedroom count  (min, max)
# ---------------------------------------------------------------------------
APARTMENT_SQFT_BY_BED = {
    1: (500, 750),
    2: (800, 1300),
    3: (1400, 2100),
    4: (2000, 2950),
}

# ---------------------------------------------------------------------------
# 7. PROPERTY TYPE MIX  &  BEDROOM SKEW
# ---------------------------------------------------------------------------
TYPE_WEIGHTS = {
    "PREMIUM": {"house": 55, "apartment": 33, "penthouse": 7,
                "upper portion": 3, "lower portion": 2},
    "MID":     {"house": 60, "apartment": 33, "penthouse": 2,
                "upper portion": 3, "lower portion": 2},
    "BUDGET":  {"house": 55, "apartment": 40, "penthouse": 0,
                "upper portion": 3, "lower portion": 2},
}
AREA_PREMIUM_LEVEL = {
    "DHA": "PREMIUM", "Clifton": "PREMIUM", "PECHS": "PREMIUM", "Askari": "PREMIUM",
    "Gulshan-e-Iqbal": "MID", "Gulistan-e-Johar": "MID", "FB Area": "MID",
    "Naya Nazimabad": "MID", "North Nazimabad": "MID", "Nazimabad": "MID",
    "Scheme 33": "MID", "KDA Scheme 1": "MID", 
    "Tariq Road": "MID", "Bahria Town": "MID",
}

APT_BED_SKEW = {
    "PREMIUM": {1: 6, 2: 24, 3: 45, 4: 25},
    "MID":     {1: 14, 2: 40, 3: 36, 4: 10},
    "BUDGET":  {1: 30, 2: 45, 3: 22, 4: 3},
}

def house_beds_for_area(sqyd: int) -> tuple[int, int]:
    if sqyd <= 120:   return (2, 3)
    if sqyd <= 240:   return (3, 4)
    if sqyd <= 400:   return (4, 5)
    if sqyd <= 1000:  return (5, 6)
    return (6, 8)

# ---------------------------------------------------------------------------
# 8. AMENITIES  — probability of inclusion by premium level
# ---------------------------------------------------------------------------
AMENITY_PROBABILITIES = {
    "Parking Spaces":               {"PREMIUM": 0.95, "MID": 0.90, "BUDGET": 0.80},
    "Electricity Backup":           {"PREMIUM": 0.90, "MID": 0.65, "BUDGET": 0.35},
    "Water Storage Tank":           {"PREMIUM": 0.90, "MID": 0.85, "BUDGET": 0.80},
    "Boundary Wall":                {"PREMIUM": 0.85, "MID": 0.80, "BUDGET": 0.70},
    "Drawing Room":                 {"PREMIUM": 0.80, "MID": 0.60, "BUDGET": 0.35},
    "Servant Quarter":              {"PREMIUM": 0.70, "MID": 0.30, "BUDGET": 0.08},
    "Central Air Conditioning":     {"PREMIUM": 0.65, "MID": 0.25, "BUDGET": 0.05},
    "Lawn":                         {"PREMIUM": 0.60, "MID": 0.30, "BUDGET": 0.10},
    "Elevator":                     {"PREMIUM": 0.55, "MID": 0.35, "BUDGET": 0.12},
    "Standby Generator":            {"PREMIUM": 0.55, "MID": 0.25, "BUDGET": 0.08},
    "Community Gym":                {"PREMIUM": 0.45, "MID": 0.18, "BUDGET": 0.03},
    "Swimming Pool":                {"PREMIUM": 0.40, "MID": 0.12, "BUDGET": 0.01},
    "Community Lawn or Garden":     {"PREMIUM": 0.45, "MID": 0.25, "BUDGET": 0.10},
    "Maintenance Staff":            {"PREMIUM": 0.50, "MID": 0.25, "BUDGET": 0.08},
    "Security Staff":               {"PREMIUM": 0.70, "MID": 0.40, "BUDGET": 0.18},
    "CCTV Security":                {"PREMIUM": 0.65, "MID": 0.40, "BUDGET": 0.20},
    "Day Care Centre":              {"PREMIUM": 0.20, "MID": 0.10, "BUDGET": 0.03},
    "First Aid or Medical Centre":  {"PREMIUM": 0.20, "MID": 0.08, "BUDGET": 0.02},
    "Mosque":                       {"PREMIUM": 0.45, "MID": 0.40, "BUDGET": 0.35},
    "Community Centre":             {"PREMIUM": 0.35, "MID": 0.20, "BUDGET": 0.08},
}

# ---------------------------------------------------------------------------
# 9. POOLS for random fields (agents, contacts, images)
# ---------------------------------------------------------------------------
AGENT_NAMES = [
    "Ali Raza", "Sara Khan", "Farah Naz", "Bilal Ahmed", "Usman Tariq",
    "Ayesha Siddiqui", "Hassan Malik", "Zainab Fatima", "Imran Sheikh",
    "Maryam Iqbal", "Faisal Qureshi", "Nida Aslam", "Kamran Baig",
    "Hira Mansoor", "Tariq Hussain", "Saad Rauf", "Mehwish Ali", "Owais Khan",
]
IMAGE_SEEDS = [
    "luxury", "interior", "outdoor", "realestate", "building", "modern",
    "villa", "apartment", "lounge", "kitchen", "bedroom", "facade",
]

# Title is generated as: f"{bedrooms} bed {type} in {subarea}"
