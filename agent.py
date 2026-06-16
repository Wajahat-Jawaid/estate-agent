from __future__ import annotations
import os
import json
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from mortgage_handler import MortgageConversationHandler
from rental_yield import estimate_rent

load_dotenv()

# PHYSICAL in-property amenities ONLY. Proximity/"nearby" concepts (school,
# hospital, park, masjid, market, restaurant, transport) are deliberately NOT here
# — they are a conceptually separate dimension handled by the near_* scalar flags
# (see _SITUATIONAL_RULES / _SOFT_SIT_BONUS), not matched against the amenities array.
AMENITY_SYNONYMS = {
    "Security Staff":                  ["security", "guard", "secure", "mehfooz", "safe area", "safety"],
    "CCTV Security":                   ["cctv", "cameras", "surveillance"],
    "Maintenance Staff":               ["maintenance", "upkeep", "maintained"],
    "Community Gym":                   ["gym", "fitness", "workout"],
    "Swimming Pool":                   ["pool", "swimming"],
    "Kids Play Area":                  ["play area", "kids area", "playground", "bachon ke liye"],
    "Community Lawn or Garden":        ["lawn", "garden", "green", "outdoor space"],
    "Mosque":                          ["mosque", "masjid", "namaz"],
    "Parking Spaces":                  ["parking", "car park", "garage"],
    "Electricity Backup":              ["backup", "generator", "load shedding", "ups", "bijli backup"],
    "Central Air Conditioning":        ["central ac", "central air", "air conditioning"],
    "Servant Quarter":                 ["servant", "maid room", "quarter"],
    "Drawing Room":                    ["drawing room", "guest room", "baithak"],
    "Community Centre":                ["community centre", "community center", "clubhouse"],
}
_FAMILY_TRIGGERS = ["family", "ghar wale", "bachon", "kids", "children", "parents"]
_FAMILY_BEDROOM_FLOOR = 3

def map_to_canonical_amenities(query, extracted_features):
    haystack = query.lower()
    if extracted_features:
        haystack += " " + " ".join(str(f).lower() for f in extracted_features)
    return [c for c, trigs in AMENITY_SYNONYMS.items() if any(t in haystack for t in trigs)]

def _amenity_match_count(doc_metadata, wanted):
    if not wanted:
        return 0
    raw = doc_metadata.get("amenities", "[]")
    try:
        have = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except Exception:
        have = []
    have_set = set(have)
    return sum(1 for w in wanted if w in have_set)

# Tunable: weight of each amenity match vs semantic distance. Range 0.08–0.18.
PER_AMENITY_WEIGHT = 0.12
BED_FLOOR_BONUS = 0.08

# Inferred "nearby/lifestyle" fields are SOFT: hard-ANDing several ~20-40% booleans
# wipes the result set, so instead they nudge ranking. (Explicit structural prefs —
# floor_band, has_lift — stay hard filters in build_chroma_filter.)
_SOFT_SIT_BONUS = {
    "near_hospital": 0.06, "near_school": 0.06, "near_park": 0.05,
    "near_masjid": 0.05, "near_market": 0.05, "near_restaurant": 0.04,
    "near_transport": 0.05, "gated_community": 0.06, "west_open": 0.06,
}

def _rerank_by_amenities(results, filters):
    wanted = filters.get("amenities_wanted") or []
    bed_floor = filters.get("bedrooms_floor")
    soft_active = (filters.get("floor") == 0) or any(filters.get(f) for f in _SOFT_SIT_BONUS)
    if not results or (not wanted and not bed_floor and not soft_active):
        return results
    scores = [s for _, s in results]
    best, worst = min(scores), max(scores)
    span = (worst - best) or 1.0
    def blended(item):
        doc, score = item
        norm = (score - best) / span
        bonus = _amenity_match_count(doc.metadata, wanted) * PER_AMENITY_WEIGHT
        if bed_floor and (doc.metadata.get("bedrooms") or 0) >= bed_floor:
            bonus += BED_FLOOR_BONUS
        for fld, b in _SOFT_SIT_BONUS.items():
            if filters.get(fld) and doc.metadata.get(fld):
                bonus += b
        if filters.get("floor") == 0 and doc.metadata.get("floor") == 0:
            bonus += 0.06
        return norm - bonus
    return sorted(results, key=blended)

def _amenity_gap_summary(matched_listings, filters):
    """Returns (requested, satisfied, missing) canonical amenity lists based on
    the top visible results. 'satisfied' = wanted amenities present in AT LEAST
    one of the top results; 'missing' = wanted amenities present in NONE."""
    wanted = filters.get("amenities_wanted") or []
    if not wanted or not matched_listings:
        return wanted, [], []
    top = matched_listings[:GRID_VISIBLE_COUNT]
    present = set()
    for l in top:
        raw = l["metadata"].get("amenities", "[]")
        try:
            have = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            have = []
        present.update(have)
    satisfied = [w for w in wanted if w in present]
    missing = [w for w in wanted if w not in present]
    return wanted, satisfied, missing

def _results_overview(matched_listings: list) -> str:
    """Compact aggregate summary across the visible results, for the
    conversational overview. The LLM grounds its summary in this block so it
    describes the whole set instead of one property. Empty string if no data."""
    top = matched_listings[:GRID_VISIBLE_COUNT]
    if not top:
        return ""

    prices = [(int(l["metadata"].get("price_numeric") or 0), l) for l in top]
    prices = [(p, l) for p, l in prices if p > 0]
    beds = [l["metadata"].get("bedrooms") for l in top if l["metadata"].get("bedrooms")]
    areas = [(int(l["metadata"].get("area_sqyd") or 0), l) for l in top]
    areas = [(a, l) for a, l in areas if a > 0]

    # Distinct parent areas, order preserved
    locs = []
    for l in top:
        loc = (l["metadata"].get("location") or "").split(",")[0].strip()
        if loc and loc not in locs:
            locs.append(loc)

    # Amenity frequency across the set
    amen_count = {}
    for l in top:
        raw = l["metadata"].get("amenities", "[]")
        try:
            have = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            have = []
        for a in have:
            amen_count[a] = amen_count.get(a, 0) + 1

    def _title(l):
        return (l["metadata"].get("title") or "property").title()

    lines = [f"Total results shown: {len(top)}"]
    if prices:
        lo = min(prices, key=lambda x: x[0])
        hi = max(prices, key=lambda x: x[0])
        lines.append(f"Price range: {lacs_to_price(lo[0])} to {lacs_to_price(hi[0])}")
        lines.append(f"Cheapest: {_title(lo[1])} in {lo[1]['metadata'].get('location', '')} at {lacs_to_price(lo[0])}")
    if beds:
        lines.append(f"Bedrooms: all {min(beds)} bed" if min(beds) == max(beds) else f"Bedrooms: {min(beds)}-{max(beds)} bed")
    if areas:
        big = max(areas, key=lambda x: x[0])
        lines.append(f"Largest: {_title(big[1])} at {big[0]} sq yd")
    if locs:
        lines.append(f"Areas covered: {', '.join(locs[:5])}")
    best = max(top, key=lambda l: l.get("score", 0))
    lines.append(f"Best match ({best.get('score')}%): {_title(best)} in {best['metadata'].get('location', '')}")
    if amen_count:
        common = [a for a, _ in sorted(amen_count.items(), key=lambda x: -x[1])[:3]]
        lines.append(f"Common amenities: {', '.join(common)}")

    return "\n".join(lines)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# llm = ChatGroq(
#     # model="llama-3.3-70b-versatile",
#     model="llama-3.1-8b-instant",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.7
# )

llm_fast = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1  # low temp for structured decisions
)

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# llm_fast = ChatOpenAI(
#     model="gpt-5.4-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.1  # low temp for structured decisions
# )

# Per-user memory and search history
user_memories = {}
user_search_histories = {}
user_pending_queries = {}  # user_id -> {"original_query": str, "filters": dict}

_mortgage_handler = MortgageConversationHandler()

def get_user_memory(user_id: str) -> ConversationBufferMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(return_messages=True)
    return user_memories[user_id]

def get_user_search_history(user_id: str) -> list:
    if user_id not in user_search_histories:
        user_search_histories[user_id] = []
    return user_search_histories[user_id]

def price_to_lacs(price_str: str) -> int:
    price_str = price_str.lower().strip()
    crore, lac = 0.0, 0
    crore_match = re.search(r'(\d+(?:\.\d+)?)\s*crore', price_str)
    lac_match = re.search(r'(\d+)\s*lac', price_str)
    if crore_match:
        crore = float(crore_match.group(1))
    if lac_match:
        lac = int(lac_match.group(1))
    return int(crore * 100) + lac

def lacs_to_price(n: int) -> str:
    c, r = n // 100, n % 100
    if c and r:
        decimal = f"{c + r / 100:.2f}".rstrip('0').rstrip('.')
        return f"{decimal} crore"
    if c: return f"{c} crore"
    return f"{n} lac"

def _parse_budget_ceiling(query: str) -> str | None:
    """Regex fallback for budget answers the LLM misses — notably ranges like
    "2.2 to 2.6 crore" (returns the UPPER bound as a price string)."""
    q = query.lower()
    crores = [float(m) for m in re.findall(r'(\d+(?:\.\d+)?)\s*crore', q)]
    lacs = [int(m) for m in re.findall(r'(\d+)\s*lac', q)]
    if crores:
        return lacs_to_price(int(max(crores) * 100))
    if lacs:
        return lacs_to_price(max(lacs))
    return None

def _parse_budget_floor(query: str) -> str | None:
    """Lower budget bound — for band answers ("between 5 and 7 crore") and floors
    ("above 7 crore", "minimum 1.5 crore"). Returns the lower bound as a price
    string, or None when the message has no floor sense."""
    q = query.lower()
    # Explicit floor: "above/over/more than/minimum/at least/starting X crore|lac"
    m = re.search(r'(?:above|over|more than|minimum|at\s*least|starting)\s*(?:rs\.?\s*)?'
                  r'(\d+(?:\.\d+)?)\s*(crore|lac)', q)
    if m:
        n = float(m.group(1))
        return lacs_to_price(int(n * 100) if m.group(2) == 'crore' else int(n))
    # Range "X to Y crore" / "between X and Y crore" → lower bound
    if any(w in q for w in (" to ", " and ", "between", "-")):
        crores = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*crore', q)]
        lacs = [int(x) for x in re.findall(r'(\d+)\s*lac', q)]
        if len(crores) >= 2:
            return lacs_to_price(int(min(crores) * 100))
        if len(lacs) >= 2:
            return lacs_to_price(min(lacs))
    return None

def extract_filters(query: str) -> dict:
    prompt = f"""Extract property search filters from this query as JSON. Only include filters EXPLICITLY stated by the user — never infer or guess.

Query: "{query}"

Return ONLY a valid JSON object with these possible keys (omit any not mentioned):
- locations (array of strings) — ONLY specific Karachi neighbourhoods/areas explicitly named (e.g. DHA, Clifton, Gulshan). Do NOT include "Karachi" itself — it is the city, not a neighbourhood. Do NOT infer areas the user didn't mention.
- types (array of strings) — list EVERY property type mentioned from: house/apartment/upper portion/lower portion/penthouse/farmhouse. Omit if not mentioned.
- bedrooms (integer)
- min_bathrooms (integer)
- max_price (string, in Pakistani format e.g. "2 crore 50 lac") — use key "max_price", not "budget"
- features (array of strings) — any amenity, lifestyle, or quality wants (e.g. "security", "schools nearby", "family-friendly", "parking")

Return only raw JSON. No explanation. No markdown. No backticks."""
    response = llm_fast.invoke([HumanMessage(content=prompt)])
    filters = {}
    try:
        raw = response.content.strip()
        # strip markdown fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        filters = json.loads(raw)
    except:
        pass

    # Normalise any rogue key names the small LLM sometimes uses
    for alias in ("budget", "price", "max_budget"):
        if alias in filters and "max_price" not in filters:
            filters["max_price"] = filters.pop(alias)

    # Normalise plural type names the LLM sometimes returns ("houses" → "house")
    _TYPE_NORMALISE = {
        "houses": "house", "apartments": "apartment", "flats": "apartment",
        "flat": "apartment", "portions": "upper portion",
        "penthouses": "penthouse", "farmhouses": "farmhouse",
    }
    if filters.get("types"):
        filters["types"] = [_TYPE_NORMALISE.get(t.lower(), t.lower()) for t in filters["types"]]

    # Guard against location hallucination: the small LLM frequently invents
    # neighbourhoods (DHA, Clifton, Gulshan) the user never typed, which then
    # pulls in pricey areas and blows past the budget. Keep only locations that
    # actually appear in the query text, and never the city ("Karachi") itself.
    if filters.get("locations"):
        q_low = query.lower()
        kept = [loc for loc in filters["locations"]
                if isinstance(loc, str)
                and loc.lower() != "karachi"
                and loc.lower() in q_low]
        if kept:
            filters["locations"] = kept
        else:
            filters.pop("locations", None)

    canonical = map_to_canonical_amenities(query, filters.get("features"))
    if canonical:
        filters["amenities_wanted"] = canonical
    if filters.get("bedrooms") is None and any(t in query.lower() for t in _FAMILY_TRIGGERS):
        filters["bedrooms_floor"] = _FAMILY_BEDROOM_FLOOR
    return filters

# Phrases that back-reference the budget/area the user already stated, instead of
# repeating the figure. When one of these appears and the current message did not
# extract its own budget, we inherit the previous search's budget rather than
# silently dropping the ceiling (which surfaced wildly over-budget results).
_BUDGET_CONTINUITY_KW = (
    "within the budget", "within budget", "in the budget", "in budget",
    "same budget", "same price", "budget mein", "budget me", "budget main",
    "usi budget", "isi budget", "us budget", "if within",
)

def _filters_to_query(filters: dict, fallback: str = "property") -> str:
    """Build a semantic-search query string from structured filters, used when the
    raw user message is too terse to search on (e.g. a REFINE like '2 bed is better')."""
    parts = []
    if filters.get("bedrooms"):
        parts.append(f"{filters['bedrooms']} bed")
    for t in (filters.get("types") or []):
        parts.append(str(t))
    for loc in (filters.get("locations") or []):
        parts.append(str(loc))
    for am in (filters.get("amenities_wanted") or []):
        parts.append(str(am))
    return " ".join(parts).strip() or fallback

_REVERSE_MORTGAGE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s*(?:mein|main|me)\s*(?:kya|kia)\b",
    r"\bmonthly\s+budget\s*(?:hai|is|of)?\s*\d",
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s+(?:per|har)\s+(?:month|mahina|maheena)\b",
    r"\bmonthly\s+\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\b",
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s*(?:monthly|per\s+month|har\s+month)\b",
    # Require a possessive/copula word and lakh-scale: "budget hai 2 lakh" is a
    # monthly income cue; a plain capital budget like "budget 2 crore" is NOT.
    r"\bbudget\s+(?:hai|mera|meri|hamara)\s+\d+\s*(?:lakh|lac)\b",
]

_MORTGAGE_EXPLICIT_KW = {
    "home loan", "installment", "instalment", "mortgage",
    "down payment", "downpayment", "afford kar sakta", "afford ho sakta",
    "kitna loan", "loan milega", "bank loan", "qist", "qarz",
    "monthly kitna", "monthly payment", "interest rate", "bank financing",
    "finance kar sakta", "housing loan", "property loan", "griha loan",
}

_AFFORDABILITY_HINT_KW = {
    "too expensive", "bohot mehnga", "bohat mehnga", "bahut mehnga",
    "budget mein nahi", "budget nahi hai", "afford nahi",
    "ziada mehnga", "zyada mehnga", "budget se bahar",
    "affordable nahi", "mehnga lag raha", "thoda kam price chahiye",
}

# Investment intent — checked BEFORE mortgage regex so "invest X crore" never
# gets swallowed by REVERSE_MORTGAGE patterns.
_INVESTMENT_KW = {
    "invest", "investment", "best yield", "best return", "rental return",
    "rental income", "passive income", "kahan invest", "invest karna",
    "best area to invest", "highest yield", "maximum return", "paisa lagana",
    "paisa invest", "return milega", "yield chahiye",
}

# Phrases that negate investment intent even though the word "invest" appears
# ("not for investment or rent", "we want to live there ourselves").
_INVESTMENT_NEGATIONS = (
    "not for investment", "not an investment", "not investment", "no investment",
    "not investing", "not to invest", "not for invest", "not as an investment",
    "not buying for investment", "neither investment", "investment nahi",
    "invest nahi", "nahi invest", "nhi invest", "not for rent or investment",
    "not rent or investment", "rent or investment", "not for rental",
)
_LIVING_INTENT_KW = (
    "live there", "to live", "for living", "live ourselves", "live myself",
    "live in it", "we want to live", "rehne ke liye", "rehna hai", "khud rehne",
    "apne rehne", "personal use", "own use", "end use", "self use",
)

def detect_investment_intent(query: str) -> bool:
    q = query.lower()
    if not any(kw in q for kw in _INVESTMENT_KW):
        return False
    # Guard: the word "invest" can appear inside a denial ("NOT for investment")
    # or alongside a clear live-in intent — neither is a real investment query.
    if any(neg in q for neg in _INVESTMENT_NEGATIONS):
        return False
    if any(kw in q for kw in _LIVING_INTENT_KW):
        return False
    return True

def _extract_investment_budget(query: str) -> int | None:
    """Regex-only budget extraction for investment queries (no LLM call)."""
    q = query.lower()
    crore_m = re.search(r'(\d+(?:\.\d+)?)\s*crore', q)
    lac_m   = re.search(r'(\d+)\s*lac', q)
    if crore_m:
        return int(float(crore_m.group(1)) * 100) + (int(lac_m.group(1)) if lac_m else 0)
    if lac_m:
        return int(lac_m.group(1))
    return None


def detect_mortgage_intent(query: str) -> str | None:
    """Fast regex-based mortgage intent detection — no LLM call."""
    q = query.lower()
    for pattern in _REVERSE_MORTGAGE_PATTERNS:
        if re.search(pattern, q):
            return "REVERSE_MORTGAGE"
    if re.search(r"\bemi\b", q) or re.search(r"\bloan\b", q) or re.search(r"\binstall?ment\b", q) or re.search(r"\bafford\b", q):
        return "MORTGAGE_EXPLICIT"
    if any(kw in q for kw in _MORTGAGE_EXPLICIT_KW):
        return "MORTGAGE_EXPLICIT"
    if any(kw in q for kw in _AFFORDABILITY_HINT_KW):
        return "AFFORDABILITY_HINT"
    return None


def classify_query(query: str, search_history: list) -> dict:
    history_summary = ""
    for i, entry in enumerate(search_history):
        history_summary += f"Search {i}: {entry['topic']}\n"
    if not history_summary:
        history_summary = "(no previous searches)"

    prompt = f"""Classify this real estate chatbot query.

Previous searches:
{history_summary}

Current query: "{query}"

CLASSIFICATION RULES:

SMALLTALK — greetings, general conversation, out of scope (weather, jokes etc)

NEWSEARCH — mentions a new location, new property type, significantly different budget, or is exploring a new category

FOLLOWUP — asking about details of already shown properties, comparing shown properties, selecting by number ("1", "2", "3"), asking about agent/contact of shown properties

CHEAPER — wants cheaper/more affordable/lower price options than what was shown

LARGER — wants vaguely bigger/larger/more rooms than what was shown ("bigger ones", "something larger"), WITHOUT naming an exact bedroom count

REFINE — user is adjusting the CURRENT search by adding or changing a CONCRETE constraint (an exact bedroom number, a property type, an amenity) while keeping the rest of what they already told you — especially their budget and area. Use this (NOT NEWSEARCH) when they give a specific new criterion but do NOT name a new location or a new budget figure. Use this (NOT CHEAPER/LARGER) when they name an exact bedroom count or type rather than a vague "cheaper"/"bigger".
  ✓ "2 bed would be better, within the budget" (keep budget + area, set bedrooms=2)
  ✓ "make it a house instead" (keep budget + area, change type to house)
  ✓ "ones with a pool" (keep everything, add the amenity)
  ✗ "show me places in DHA" → NEWSEARCH (names a new location)
  ✗ "budget 2 crore now" → NEWSEARCH (states a new budget figure)
  ✗ "cheaper ones" / "bigger ones" → CHEAPER / LARGER

IMAGES — wants to see photos/pictures/images of a property already shown. May reference a property number (e.g. "images of property 2", "show me photos of the first one")

MORTGAGE_EXPLICIT — user directly asks about EMI, loan, installment, or affordability calculation (kitna loan milega, EMI kya hogi, monthly installment, home loan, afford kar sakta hun)

AFFORDABILITY_HINT — user hints price is a concern but doesn't request a calculation (too expensive, mehnga hai, budget mein nahi, afford nahi kar sakta) — NOT a request for cheaper listings

REVERSE_MORTGAGE — user states a monthly budget and asks what property they can afford (2 lakh mein kya milega, monthly X budget hai, X per month afford kar sakta). The user is asking about AFFORDABILITY from INCOME, not capital deployment.

INVESTMENT — user has a lump-sum capital and wants to know which areas/properties give the best rental yield, with NO fixed location in mind. The user is choosing WHERE to deploy capital for passive income, not asking what they can afford.
  ✓ "invest 1 crore for best yield"
  ✓ "kahan invest karun best return ke liye"
  ✓ "best area to invest 50 lac for rental income"
  ✓ "passive income ke liye property kahan lun"
  ✗ NOT INVESTMENT if a specific location is stated ("invest in DHA?" → NEWSEARCH)
  ✗ NOT INVESTMENT if asking about monthly affordability → REVERSE_MORTGAGE

Return ONLY valid JSON with no explanation:
{{"type": "NEWSEARCH", "index": null, "property_num": null}}
{{"type": "FOLLOWUP", "index": 0, "property_num": null}}
{{"type": "CHEAPER", "index": 0, "property_num": null}}
{{"type": "LARGER", "index": 0, "property_num": null}}
{{"type": "REFINE", "index": 0, "property_num": null}}
{{"type": "SMALLTALK", "index": null, "property_num": null}}
{{"type": "IMAGES", "index": 0, "property_num": 2}}
{{"type": "MORTGAGE_EXPLICIT", "index": null, "property_num": null}}
{{"type": "AFFORDABILITY_HINT", "index": null, "property_num": null}}
{{"type": "REVERSE_MORTGAGE", "index": null, "property_num": null}}
{{"type": "INVESTMENT", "index": null, "property_num": null}}

Index = which previous search is being referenced. Default to last search index if unclear.
property_num = 1-based property number the user referenced (null if not specified)."""

    response = llm_fast.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        print(f">> Classification: {result}")
        return result
    except:
        return {"type": "NEWSEARCH", "index": None}

def build_chroma_filter(filters: dict):
    conditions = []

    if filters.get("bedrooms") is not None:
        conditions.append({"bedrooms": {"$eq": filters["bedrooms"]}})

    if filters.get("min_bathrooms") is not None:
        conditions.append({"bathrooms": {"$gte": filters["min_bathrooms"]}})

    types = filters.get("types") or ([filters["type"]] if filters.get("type") else [])
    if types:
        valid_types = [t for t in types if isinstance(t, str)]
        if len(valid_types) == 1:
            conditions.append({"type": {"$eq": valid_types[0]}})
        elif len(valid_types) > 1:
            conditions.append({"type": {"$in": valid_types}})

    # Price range intentionally NOT applied here — Chroma's HNSW ANN stops early
    # when range filters reject most nearby vectors, returning far too few results.
    # Price filtering is done in Python after retrieval instead.

    # near_*/gated/floor==0 are SOFT (ranking only, see _SOFT_SIT_BONUS); hard-ANDing
    # several low-frequency booleans empties the result set. Only explicit structural
    # preferences (middle floor, lift) are applied as hard equality filters here.
    if filters.get("floor_band"):
        conditions.append({"floor_band": {"$eq": filters["floor_band"]}})
    if filters.get("has_lift"):
        conditions.append({"has_lift": {"$eq": True}})
    if filters.get("possession"):
        conditions.append({"possession": {"$eq": filters["possession"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def search_properties(query: str, filters: dict, k: int = 10) -> list:
    chroma_filter = build_chroma_filter(filters)
    locations = filters.get("locations") or []
    fetch_k = max(k * 10, 100)

    if len(locations) > 1:
        per_loc = []
        for loc in locations:
            loc_lower = loc.lower()
            q = f"{query} {loc}"
            try:
                res = vectorstore.similarity_search_with_score(q, k=fetch_k, filter=chroma_filter) if chroma_filter else vectorstore.similarity_search_with_score(q, k=fetch_k)
            except:
                res = vectorstore.similarity_search_with_score(q, k=fetch_k)

            loc_matches = [(doc, score) for doc, score in res if loc_lower in doc.metadata.get("location", "").lower()]
            if filters.get("max_price"):
                max_lacs = price_to_lacs(str(filters["max_price"]))
                if max_lacs > 0:
                    price_ok = [(d, s) for d, s in loc_matches if int(d.metadata.get("price_numeric") or 0) <= max_lacs]
                    if price_ok:
                        loc_matches = price_ok
            loc_matches = sorted(loc_matches, key=lambda x: x[1])
            per_loc.append(loc_matches)
            print(f">> {loc}: {len(loc_matches)} candidates")

        seen_ids = set()
        merged = []
        iters = [iter(b) for b in per_loc]
        exhausted = [False] * len(iters)
        while len(merged) < k and not all(exhausted):
            for i, it in enumerate(iters):
                if exhausted[i] or len(merged) >= k:
                    continue
                try:
                    doc, score = next(it)
                    doc_id = doc.metadata.get("id", doc.page_content[:80])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append((doc, score))
                except StopIteration:
                    exhausted[i] = True
        merged = _rerank_by_amenities(merged, filters)
        results = merged
        # fall through to shared price-proximity sort below

    else:
        try:
            results = vectorstore.similarity_search_with_score(query, k=fetch_k, filter=chroma_filter) if chroma_filter else vectorstore.similarity_search_with_score(query, k=fetch_k)
        except Exception as e:
            print(f"Filter failed, falling back: {e}")
            results = vectorstore.similarity_search_with_score(query, k=fetch_k)

        if not results and chroma_filter:
            results = vectorstore.similarity_search_with_score(query, k=fetch_k)

        # Python-side price filter (reliable; Chroma range filters break HNSW retrieval)
        if filters.get("max_price"):
            max_lacs = price_to_lacs(str(filters["max_price"]))
            if max_lacs > 0:
                price_filtered = [(d, s) for d, s in results if int(d.metadata.get("price_numeric") or 0) <= max_lacs]
                if price_filtered:
                    results = price_filtered

        if locations:
            loc_lower = locations[0].lower()
            filtered = [(doc, score) for doc, score in results if loc_lower in doc.metadata.get("location", "").lower()]
            if filtered:
                results = filtered

        results = _rerank_by_amenities(results, filters)
    if not filters.get("locations"):
        def _parent_area(location: str) -> str:
            area = location.split(",")[0].strip()
            # "DHA Phase 1", "DHA Phase 5", "North Nazimabad Block B" → parent brand
            return re.sub(r'\s+(?:phase|block|sector|extension|scheme)\s+\S+.*$', '', area, flags=re.IGNORECASE).strip()

        area_count: dict = {}
        diverse, leftovers = [], []
        for item in results:
            area = _parent_area(item[0].metadata.get("location", ""))
            c = area_count.get(area, 0)
            if c < 2:
                area_count[area] = c + 1
                diverse.append(item)
            else:
                leftovers.append(item)
        results = diverse + leftovers

    # Price-proximity sort: regex on the raw query is the most reliable signal —
    # handles "around X", "1.5 crore", etc. regardless of what the LLM extracted.
    q_lower = query.lower()
    crore_m = re.search(r'(\d+(?:\.\d+)?)\s*crore', q_lower)
    lac_m = re.search(r'(\d+)\s*lac', q_lower)
    target_lacs = (int(float(crore_m.group(1)) * 100) if crore_m else 0) + (int(lac_m.group(1)) if lac_m else 0)
    # Fall back to extracted filter if regex found nothing
    if target_lacs == 0 and filters.get("max_price"):
        target_lacs = price_to_lacs(str(filters["max_price"]))

    print(f">> Price sort: target={target_lacs} lacs, candidates={len(results)}, prices={sorted(set(int(d.metadata.get('price_numeric') or 0) for d,_ in results))[:10]}")

    if target_lacs > 0:
        results = sorted(results, key=lambda x: abs(int(x[0].metadata.get("price_numeric") or 0) - target_lacs))
        print(f">> After sort top-5 prices: {[int(d.metadata.get('price_numeric',0)) for d,_ in results[:5]]}")

    # Hard budget ceiling — the single guarantee that no wildly over-budget
    # property is ever returned. The per-path filters above keep over-budget
    # items as a "show something" fallback when an area has nothing in range;
    # that leak ends here. No fallback: if nothing fits, return empty and let
    # the no-results path suggest a higher budget or different area.
    if filters.get("max_price"):
        ceil = price_to_lacs(str(filters["max_price"]))
        if ceil > 0:
            ceil = int(ceil * 1.1)  # small grace for "around X" budgets
            results = [(d, s) for d, s in results if int(d.metadata.get("price_numeric") or 0) <= ceil]
    if filters.get("min_price"):
        floor = price_to_lacs(str(filters["min_price"]))
        if floor > 0:
            floor = int(floor * 0.9)  # small grace below the band floor
            results = [(d, s) for d, s in results if int(d.metadata.get("price_numeric") or 0) >= floor]

    # Hard minimum bedrooms ("at least 4", "2 or 3 bed") — python-filtered here
    # rather than in the Chroma where-clause, since range filters break HNSW.
    if filters.get("bedrooms") is None and filters.get("bedrooms_min") is not None:
        bmin = filters["bedrooms_min"]
        results = [(d, s) for d, s in results if int(d.metadata.get("bedrooms") or 0) >= bmin]

    return results[:k]

def fetch_cheaper(locations: list, max_price_lacs: int, prev_filters: dict, k: int = 10) -> list:
    min_price_lacs = max(int(max_price_lacs * 0.50), 30)
    price_filter = {"$and": [
        {"price_numeric": {"$lte": max_price_lacs}},
        {"price_numeric": {"$gte": min_price_lacs}},
    ]}
    types = prev_filters.get("types", [])
    if types:
        type_cond = {"type": {"$eq": types[0]}} if len(types) == 1 else {"type": {"$in": types}}
        price_filter = {"$and": [price_filter, type_cond]}

    query = " ".join(locations) if locations else "property"
    seen_ids = {}
    for loc in (locations if locations else [""]):
        q = f"{query} {loc}".strip()
        try:
            res = vectorstore.similarity_search_with_score(q, k=200, filter=price_filter)
        except:
            try:
                simple = {"$and": [{"price_numeric": {"$lte": max_price_lacs}}, {"price_numeric": {"$gte": min_price_lacs}}]}
                res = vectorstore.similarity_search_with_score(q, k=200, filter=simple)
            except:
                res = vectorstore.similarity_search_with_score(q, k=200)

        for doc, score in res:
            if not loc or loc.lower() in doc.metadata.get("location", "").lower():
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids[doc_id] = (doc, score)

    sorted_results = sorted(seen_ids.values(), key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
    return sorted_results[:k]

GRID_VISIBLE_COUNT = 10  # must match frontend renderCards(listings.slice(0, N))

def _results_to_listings(results: list, filters: dict = None) -> tuple:
    """
    Returns (matched_listings, context).
    context only covers the first GRID_VISIBLE_COUNT results so the LLM
    can only reference properties that are immediately visible in the grid.
    """
    context = ""
    matched_listings = []
    raw = list(results)
    if not raw:
        return matched_listings, context
    scores = [s for _, s in raw]
    best = min(scores)
    worst = max(scores)
    span = worst - best
    filters = filters or {}
    for i, (doc, score) in enumerate(raw):
        if i < GRID_VISIBLE_COUNT:
            context += f"\n---\n{doc.page_content}\n"
        norm = 95 if span == 0 else 95 - ((score - best) / span) * 23
        meta = doc.metadata
        rent = estimate_rent(
            price_numeric=int(meta.get("price_numeric") or 0),
            location=meta.get("location", ""),
            property_type=meta.get("type", "house"),
        )

        # Build honest match_reason from actual filter criteria
        match_parts = []
        if filters.get("bedrooms") and meta.get("bedrooms") == filters["bedrooms"]:
            match_parts.append(f"{filters['bedrooms']} bed")
        if filters.get("max_price"):
            max_lacs = price_to_lacs(str(filters["max_price"]))
            if max_lacs > 0 and int(meta.get("price_numeric") or 0) <= max_lacs:
                match_parts.append(f"within {filters['max_price']} budget")
        if filters.get("amenities_wanted"):
            raw_am = meta.get("amenities", "[]")
            try:
                have_am = json.loads(raw_am) if isinstance(raw_am, str) else (raw_am or [])
            except Exception:
                have_am = []
            matched_am = [a for a in filters["amenities_wanted"] if a in have_am]
            if matched_am:
                match_parts.append(", ".join(matched_am[:2]))
        if filters.get("locations"):
            loc_lower = filters["locations"][0].lower()
            prop_loc = meta.get("location", "")
            if loc_lower in prop_loc.lower():
                match_parts.append(prop_loc.split(",")[0])

        if match_parts:
            match_reason = "Matches: " + ", ".join(match_parts)
        else:
            facts = []
            if meta.get("area_sqyd"):
                facts.append(f"{meta['area_sqyd']} sq yd")
            if meta.get("bedrooms"):
                facts.append(f"{meta['bedrooms']} bed")
            raw_am = meta.get("amenities", "[]")
            try:
                have_am = json.loads(raw_am) if isinstance(raw_am, str) else (raw_am or [])
            except Exception:
                have_am = []
            if have_am:
                facts.append(have_am[0])
            match_reason = "Closest match" + (f" — {', '.join(facts)}" if facts else "")

        matched_listings.append({"metadata": meta, "score": round(norm), "rental_yield": rent, "match_reason": match_reason})
    return matched_listings, context

def decide_actions(
    user_query: str,
    classification_type: str,
    matched_listings: list,
    filters: dict,
    search_history: list,
    channel: str,
    history_text: str,
    open_dims_hint: str = ""
) -> dict:
    """
    Second LLM call — decides the follow-up message and contextual actions.
    Returns: {"follow_up": str|null, "actions": [{"id": str, "label": str}]}
    """
    is_overview = classification_type in ("NEWSEARCH", "CHEAPER", "LARGER")
    no_results = len(matched_listings) == 0
    prices = [int(l["metadata"].get("price_numeric") or 0) for l in matched_listings if l["metadata"].get("price_numeric")]
    min_price = min(prices) if prices else 0
    max_price_shown = max(prices) if prices else 0
    bedrooms_shown = list(set([l["metadata"].get("bedrooms") for l in matched_listings if l["metadata"].get("bedrooms")]))
    has_price_filter = bool(filters.get("max_price"))
    locations = filters.get("locations", [])
    
    max_actions = 2 if channel == "whatsapp" else 3

    prompt = f"""You are deciding what follow-up options to offer after a real estate search.

Context:
- User query: "{user_query}"
- Query type: {classification_type}
- Properties found: {len(matched_listings)}
- Price range shown: {lacs_to_price(min_price)} to {lacs_to_price(max_price_shown)} {"(no price filter applied)" if not has_price_filter else "(price filter was used)"}
- Bedrooms shown: {bedrooms_shown}
- Locations: {locations}
- Channel: {channel}
- Dimensions the user has NOT pinned down yet: {open_dims_hint or "unknown"}
- Conversation: {history_text[-200:] if history_text else "none"}

Available action IDs and when to use them:
- "cheaper": user might want cheaper options. Use if results found AND there's room to go cheaper AND no very tight price filter already applied
- "larger": user might want bigger properties. Use if results found AND properties aren't already very large (5+ beds)
- "contact": user wants agent contacts. Use if results found
- "new_search": user wants to search something different. Use if it makes sense contextually
- "increase_budget": user has no results and might consider higher budget. Use only if no_results=true
- "different_area": user has no results and might try different area. Use only if no_results=true

Rules:
- If SMALLTALK or greeting: return empty actions and null follow_up
- If no results: offer increase_budget and/or different_area only
- If results found: offer relevant actions only (max {max_actions})
- The follow_up MUST be ONE specific, contextual question that moves the user forward — never generic. NEVER "Anything else I can help with?" or "What matters most: price, size, or location?".
- For a fresh result set ({"this is a fresh set" if is_overview else "NOT a fresh set"}): target a dimension the user hasn't pinned down yet (see "Dimensions the user has NOT pinned down yet" above). Pick the most useful one or two and ask concretely, e.g. "Want me to narrow these by area, or by how many bedrooms you need?".
- For a single property the user drilled into: ask something specific to it — offer to line it up against a cheaper option, pull the agent's contact, or show more photos.
- Match the language style of the conversation (Urdu/English)

Return ONLY valid JSON:
{{
  "follow_up": "contextual follow-up message or null",
  "actions": [
    {{"id": "action_id", "label": "emoji + short label"}}
  ]
}}

No explanation. No markdown. No backticks."""

    try:
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        # Safety: cap actions at max
        result["actions"] = result.get("actions", [])[:max_actions]
        result["follow_up"] = result.get("follow_up")
        print(f">> Actions decided: {result}")
        return result
    except Exception as e:
        print(f">> Action decision failed: {e}")
        # Safe fallback
        if no_results:
            return {"follow_up": "Want to try a different area or adjust your budget?", "actions": [{"id": "different_area", "label": "🗺️ Different area"}, {"id": "increase_budget", "label": "💰 Adjust budget"}]}
        return {"follow_up": None, "actions": [{"id": "contact", "label": "📞 Agent contacts"}, {"id": "new_search", "label": "🔍 New search"}]}

def _handle_mortgage_intent(intent: str, user_id: str, query: str, search_history: list) -> dict:
    """Route a detected mortgage intent to the appropriate handler."""
    context_price_rupees = None
    if search_history:
        last_listings = search_history[-1].get("listings", [])
        if last_listings:
            price_numeric = last_listings[0]["metadata"].get("price_numeric")
            if price_numeric:
                context_price_rupees = int(price_numeric) * 100_000  # lacs → rupees

    if intent == "MORTGAGE_EXPLICIT":
        return _mortgage_handler.handle_mortgage_explicit(user_id, query, context_price_rupees=context_price_rupees)

    if intent == "AFFORDABILITY_HINT":
        return _mortgage_handler.handle_affordability_hint(user_id, query)

    if intent == "REVERSE_MORTGAGE":
        result = _mortgage_handler.handle_reverse_mortgage(user_id, query)
        if result["meta"].get("trigger_search"):
            max_lacs = result["meta"]["max_price_lacs"]
            filters = {"max_price": lacs_to_price(max_lacs)}
            search_results = search_properties(query, filters)
            search_results.sort(key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
            matched_listings, context = _results_to_listings(search_results, filters)
            result["listings"] = matched_listings
            result["meta"]["new_results"] = True
            topic = f"budget ≤ {lacs_to_price(max_lacs)}"
            search_history.append({
                "topic": topic,
                "listings": matched_listings,
                "context": context,
                "filters": filters,
            })
        return result

    return {"response": "Let me help with that mortgage question!", "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False}}


def _handle_investment_intent(user_query: str, user_id: str, search_history: list, channel: str = "web") -> dict:
    """
    Structured investment response: ranks tiers by yield for a lump-sum budget,
    shows 2-3 tiers (highest yield first + a premium appreciation counterpoint),
    each with a representative listing pulled from real stock.
    """
    from collections import defaultdict

    budget_lacs = _extract_investment_budget(user_query)
    if not budget_lacs:
        return {
            "response": "What's your investment budget? E.g. 1 crore, 50 lac, 2 crore 50 lac.",
            "listings": [], "filters": {}, "follow_up": None, "actions": [],
            "meta": {"no_results": False, "action": "investment"},
        }

    budget_str = lacs_to_price(budget_lacs)

    try:
        raw = vectorstore.similarity_search_with_score(
            "rental investment property income yield",
            k=80,
            filter={"price_numeric": {"$lte": budget_lacs}},
        )
    except Exception:
        raw = vectorstore.similarity_search_with_score("property", k=80)
        raw = [(d, s) for d, s in raw if int(d.metadata.get("price_numeric") or 0) <= budget_lacs]

    if not raw:
        return {
            "response": f"No properties found under {budget_str}. Try a higher budget.",
            "listings": [], "filters": {}, "follow_up": None, "actions": [],
            "meta": {"no_results": True, "action": "investment"},
        }

    # ── Annotate every doc with its yield estimate ───────────────────────────
    annotated: list[tuple] = []  # (doc, score, rent_dict)
    for doc, score in raw:
        meta = doc.metadata
        rent = estimate_rent(
            price_numeric=int(meta.get("price_numeric") or 0),
            location=meta.get("location", ""),
            property_type=meta.get("type", "house"),
        )
        annotated.append((doc, score, rent))

    # ── Bucket into yield bands ───────────────────────────────────────────────
    # HIGH  ≥ 6.5 y_low  →  best monthly cash flow (Korangi-tier rates)
    # MID   ≥ 5.5 y_low  →  balanced yield + appreciation (Gulshan-tier rates)
    # PREMIUM < 5.5 y_low →  lowest yield but strongest resale (DHA/Clifton rates)
    _BAND_BRIEF = {
        "HIGH":    "Maximum monthly cash flow — suits income-focused investors. Weaker resale market.",
        "MID":     "Solid monthly cash flow with established demand and balanced appreciation.",
        "PREMIUM": "Lowest monthly yield but the strongest resale value and long-term capital appreciation.",
    }

    def _band(y_low: float) -> str:
        if y_low >= 6.5:
            return "HIGH"
        if y_low >= 5.5:
            return "MID"
        return "PREMIUM"

    band_docs: dict[str, list] = defaultdict(list)
    for doc, score, rent in annotated:
        band_docs[_band(rent["yield_low"])].append((doc, score, rent))

    # Sort bands: HIGH first (best yield), then MID, then PREMIUM.
    # Always include PREMIUM as appreciation counterpoint if it has listings.
    yield_order = [b for b in ("HIGH", "MID", "PREMIUM") if b in band_docs]
    display_bands: list[str] = list(yield_order[:2])
    if "PREMIUM" in band_docs and "PREMIUM" not in display_bands:
        display_bands.append("PREMIUM")

    # ── Representative listing per band (highest price = best quality) ────────
    def _rep(items: list) -> tuple:
        return max(items, key=lambda x: int(x[0].metadata.get("price_numeric") or 0))

    # ── Build structured response text ───────────────────────────────────────
    lines = [
        f"For {budget_str} aimed at rental income, here's where it works hardest "
        f"— these are area-level estimates, not guarantees:\n"
    ]

    for i, band in enumerate(display_bands):
        rep_doc, _, rep_rent = _rep(band_docs[band])
        meta = rep_doc.metadata
        area_name = meta.get("location", "").split(",")[0]

        pos_label = "Best yield" if i == 0 else ("Premium hold" if band == "PREMIUM" else "Balanced")
        y_low, y_high = rep_rent["yield_low"], rep_rent["yield_high"]

        lines.append(f"🔹 {pos_label} — {area_name} (est. {y_low}–{y_high}%)")
        lines.append(
            f"~PKR {rep_rent['monthly_rent_low']:,}–{rep_rent['monthly_rent_high']:,}/mo on a "
            f"{lacs_to_price(int(meta.get('price_numeric', 0)))} property. {_BAND_BRIEF[band]}"
        )
        lines.append(
            f"Example: {meta.get('title', 'Property').title()}, "
            f"{area_name} — {meta.get('price', '')}"
        )
        lines.append("")

    # ── LLM: 2-sentence tradeoff + next-step ─────────────────────────────────
    rep_high = _rep(band_docs[display_bands[0]])
    highest_area = rep_high[0].metadata.get("location", "").split(",")[0]
    prem_band    = next((b for b in display_bands if b == "PREMIUM"), display_bands[-1])
    premium_area = _rep(band_docs[prem_band])[0].metadata.get("location", "").split(",")[0]
    band_areas   = " / ".join(
        _rep(band_docs[b])[0].metadata.get("location", "").split(",")[0]
        for b in display_bands
    )

    tradeoff = llm_fast.invoke([HumanMessage(content=
        f"""Write exactly 2 sentences for a property investor:
1. Start with "Quick read:" — contrast {highest_area} (best monthly yield) vs {premium_area} (lower yield but stronger resale). Be specific and direct.
2. Ask if they want actual listings in any specific area ({band_areas}).
Budget: {budget_str}. Match language of: "{user_query}" (Urdu or English). Conversational, not formal."""
    )])
    lines.append(tradeoff.content.strip())

    # ── Build matched_listings for property cards ─────────────────────────────
    seen_ids: set = set()
    matched_listings = []
    for band in display_bands:
        rep_doc, _, rep_rent = _rep(band_docs[band])
        priority = [(rep_doc, rep_rent)] + [
            (d, r) for d, _, r in band_docs[band] if d is not rep_doc
        ]
        for doc, rent in priority[:3]:
            doc_id = doc.metadata.get("id")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            matched_listings.append({"metadata": doc.metadata, "score": 85, "rental_yield": rent})

    filters = {"max_price": budget_str}
    search_history.append({
        "topic": f"investment ≤ {budget_str}",
        "listings": matched_listings,
        "context": "",
        "filters": filters,
    })
    print(f">> Investment: budget={budget_str}, bands={display_bands}, listings={len(matched_listings)}")

    return {
        "response": "\n".join(lines),
        "listings": matched_listings,
        "filters": filters,
        "follow_up": None,
        "actions": [{"id": "new_search", "label": "🔍 New search"}],
        "meta": {"no_results": False, "action": "investment", "new_results": True},
    }


def _needs_clarification(filters: dict) -> bool:
    """True when the query is too vague to yield useful results — no location AND no budget."""
    return not filters.get("locations") and not filters.get("max_price")


def _generate_clarifying_question(query: str, filters: dict, history_text: str) -> str:
    has_beds = filters.get("bedrooms")
    has_type = filters.get("types")

    known_parts = []
    if has_beds:
        known_parts.append(f"{has_beds} bedroom{'s' if has_beds > 1 else ''}")
    if has_type:
        known_parts.append(", ".join(has_type))
    known_str = " ".join(known_parts) if known_parts else ""

    prompt = f"""You are a friendly real estate assistant for Karachi, Pakistan.
The user is looking for a property. Here's what they said: "{query}"
{f"You already know they want: {known_str}" if known_str else "You don't have specific property details yet."}
You still need: their preferred area/location in Karachi, and their budget.

Ask for BOTH the area and budget in ONE natural, warm question — like a knowledgeable friend would ask.
- If you already know some details (beds, type), acknowledge them briefly before asking
- Ask for both area and budget together, not separately
- Be concise — 1-2 sentences max
- Match the user's language (Urdu or English)
- Don't use bullet points or lists in the question

Return ONLY the question, nothing else."""

    response = llm_fast.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ──────────────────────────────────────────────────────────────────────────
# Progressive discovery
# ──────────────────────────────────────────────────────────────────────────
# Discovery is active from the first property query until the first set of
# results is revealed, after which the normal follow-up machinery
# (cheaper/larger/refine/images) owns the conversation via search_history.
user_discovery_states = {}  # user_id -> {"filters": dict, "asked": set, "turns": int, "active": bool}

DISCOVERY_MAX_QUESTIONS = 8

# Escape hatch — "stop asking, show me what you've got".
_SHOW_RESULTS_KW = (
    "show me", "show now", "just show", "dikhao", "dikha do", "dekhna", "results",
    "see options", "show options", "show properties", "show listings", "bata do",
    "let's see", "lets see", "go ahead", "abhi dikhao", "jaldi", "skip",
)

# Situational inference: life-situation phrases → the scalar inference fields
# added in Phase 0. Deliberately keyword-based (auditable, deterministic) and
# kept SEPARATE from extract_filters so we never reopen its anti-hallucination
# guards. These values are *inferred*, not stated, so they are subject to
# walk-back if they over-narrow the result set to zero.
_SITUATIONAL_RULES = [
    (("parent", "parents", "father", "mother", "walid", "walida", "ammi", "abbu",
      "buzurg", "elderly", "old age", "ghutno", "knees", "wheelchair"),
     {"floor": 0, "near_hospital": True}),
    (("family", "kids", "kid", "children", "child", "bachay", "bachon", "bache",
      "school going", "growing family"),
     {"near_school": True, "near_park": True, "gated_community": True}),
    (("masjid", "mosque", "namaz", "namaaz", "jamaat", "azaan", "azan"),
     {"near_masjid": True}),
    (("security", "secure", "safe", "safety", "gated", "guard", "mehfooz"),
     {"gated_community": True}),
    (("hospital", "clinic", "medical", "doctor"),
     {"near_hospital": True}),
    (("school", "schools", "education", "bachon ka school", "study ke liye"),
     {"near_school": True}),
    (("park", "jogging", "playground"),
     {"near_park": True}),
    (("market", "markets", "mall", "malls", "shopping", "bazaar", "bazar", "grocery"),
     {"near_market": True}),
    (("restaurant", "restaurants", "dining", "eatery", "food street", "khane"),
     {"near_restaurant": True}),
    (("public transport", "bus", "metro", "commute", "transport"),
     {"near_transport": True}),
]

_SITUATIONAL_FIELDS = ("floor", "near_hospital", "near_school", "near_park",
                       "near_masjid", "near_market", "near_restaurant",
                       "near_transport", "gated_community")

# Deterministic "this is clearly about property" signal. Used to bypass the
# unreliable small-talk LLM gate: a message with any of these is treated as a
# property/search/follow-up turn, so discovery reliably starts and follow-ups
# like "show cheaper ones" aren't misfired as small talk. Word boundaries keep
# short tokens from matching inside other words ("rent" inside "parent").
_PROPERTY_SIGNAL_RE = re.compile(
    r"\b(?:flats?|apartments?|house|houses|portion|penthouse|farmhouse|bungalow|"
    r"plots?|villas?|makaan?|ghar|bedrooms?|bhk|budget|crores?|lakh|lac|marla|sqyd|"
    r"property|properties|rental|rent|cheaper|larger|bigger|smaller|images?|photos?|"
    r"agent|contact)\b"
    r"|looking for|look for|want a|need a|want to buy|buy a|purchase|show me|find me|"
    r"for my family|chahiye|chahie|dikhao|dhoond|real estate|sq\s*yd|square yard",
    re.IGNORECASE,
)

def _has_property_signal(query: str) -> bool:
    return bool(_PROPERTY_SIGNAL_RE.search(query))


_TYPE_WORDS = ("flat", "apartment", "house", "portion", "penthouse", "farmhouse",
               "bungalow", "villa", "makan", "makaan", "ghar", "plot")

def _mentions_type(query: str) -> bool:
    ql = query.lower()
    return any(w in ql for w in _TYPE_WORDS)


# Surface word → canonical type present in the data (house, apartment, penthouse,
# upper/lower portion). "portion" is handled separately since it's two words.
_TYPE_SURFACE = {
    "apartment": "apartment", "apartments": "apartment", "flat": "apartment", "flats": "apartment",
    "house": "house", "houses": "house", "bungalow": "house", "bungalows": "house",
    "villa": "house", "villas": "house", "kothi": "house", "makan": "house", "makaan": "house",
    "penthouse": "penthouse", "penthouses": "penthouse",
    "farmhouse": "farmhouse", "farmhouses": "farmhouse",
}

def _match_types(query: str) -> list:
    """Deterministic property-type capture. The small extraction LLM routinely
    returns {} even for a plain "apartment"/"house" answer, so — exactly like areas
    and bedrooms — we parse types ourselves and treat the result as authoritative
    when the message names one. Returns canonical types in mention order."""
    ql = " " + query.lower() + " "
    found = []
    if re.search(r'\bportions?\b', ql):  # "upper/lower portion" — else both
        up, lo = bool(re.search(r'\bupper\b', ql)), bool(re.search(r'\blower\b', ql))
        if up:
            found.append("upper portion")
        if lo:
            found.append("lower portion")
        if not up and not lo:
            found += ["upper portion", "lower portion"]
    for word, canon in _TYPE_SURFACE.items():
        if re.search(r'\b' + re.escape(word) + r'\b', ql):
            found.append(canon)
    return list(dict.fromkeys(found))


# Roman-Urdu markers — used to pick the reply language from the *sentence*, not a
# greeting. "Assalam o Alaikum. I'm looking for a flat" is an English message.
_ROMAN_URDU_MARKERS = (
    " hai", " mein ", " aur ", " kya", " ke ", " ki ", " ka ", " aap", " mujhe",
    " chahiye", " kar ", " nahi", " liye", " kitne", " kitna", " hain", " apni",
    " apna", " thora", " thoda", " dikha", " ko ", " ya ", " mera", " meri",
)

def _detect_language(text: str) -> str:
    if any('؀' <= c <= 'ۿ' for c in text):  # Urdu script
        return "Urdu"
    t = " " + text.lower() + " "
    hits = sum(1 for m in _ROMAN_URDU_MARKERS if m in t)
    return "Urdu" if hits >= 2 else "English"


def infer_situational_filters(query: str) -> dict:
    """Map life-situation phrases to inferred scalar filters. Kept separate from
    extract_filters; never touches budget/location/type."""
    q = query.lower()
    inferred = {}
    for triggers, fields in _SITUATIONAL_RULES:
        if any(t in q for t in triggers):
            inferred.update(fields)
    return inferred


# Floor / lift preference (deep discovery). "Not ground, not top" => middle.
_FLOOR_PREF_RULES = [
    (("middle floor", "not ground", "not top", "beech wali", "darmiyan", "neither ground"),
     {"floor_band": "middle"}),
    (("ground floor", "bottom floor", "neeche wali", "ground hi"),
     {"floor_band": "ground"}),
    (("top floor", "topmost", "upper most", "sabse upar", "highest floor"),
     {"floor_band": "top"}),
]
# Word-boundary match — a bare substring test set has_lift on "C-lift-on"
# (Clifton), silently forcing a lift requirement whenever the area was mentioned.
_LIFT_RE = re.compile(r'\b(?:lift|elevator)\b', re.IGNORECASE)
_WEST_OPEN_KW = ("west open", "west-open", "west khula", "westopen")

def infer_floor_pref(query: str) -> dict:
    q = query.lower()
    out = {}
    for triggers, fields in _FLOOR_PREF_RULES:
        if any(t in q for t in triggers):
            out.update(fields)
    if _LIFT_RE.search(query):
        out["has_lift"] = True
    return out

def infer_extras(query: str) -> dict:
    q = query.lower()
    out = {}
    if any(t in q for t in _WEST_OPEN_KW):
        out["west_open"] = True
    return out

# Household composition → suggested bedrooms. The agent in our gold-standard
# script reasons "couple + two kids (+ occasional parent) => 3-bed".
_KID_RE = re.compile(r"(\d+|one|two|three|four|do|teen|char|ek)\s*(kids?|children|child|bachay|bachon|bache|beta|beti)", re.I)
_COUPLE_KW = ("husband", "wife", "couple", "mian biwi", "me and my wife", "my wife", "we are", "shadi")
_PARENT_KW = ("mother", "father", "parents", "mom", "dad", "ammi", "abbu", "walid", "walida")

def infer_household(query: str) -> dict:
    q = query.lower()
    has_couple = any(w in q for w in _COUPLE_KW)
    has_kids = bool(_KID_RE.search(q))
    has_parent = any(w in q for w in _PARENT_KW)
    if has_couple or has_kids or has_parent:
        # SOFT bedroom hint (rerank), never a hard exact filter — an explicit bed
        # number/range from the user always wins over this inference.
        return {"bedrooms_floor": 3} if (has_kids or has_parent) else {"bedrooms_floor": 2}
    return {}


# Purpose (own living vs investment) — the natural opening question a human agent
# asks. Living is inferred from family/live-in language; an explicit investment
# query is handled upstream by detect_investment_intent.
_LIVING_PURPOSE_MARKERS = (
    " live there", " to live", " for living", " own living", " we live", " want to live",
    " rehne", " khud rehne", " for my family", " my family", " joint family", " my parents",
    " for my parents", " my kids", " my children", " my wife", " my husband",
    " move back", " moving back", " planning to move", " relocat", " shift back",
    " settle back", " move-in", " moving to karachi", " shifting",
)

def infer_purpose(query: str) -> dict:
    q = " " + query.lower() + " "
    if any(m in q for m in _LIVING_PURPOSE_MARKERS):
        return {"purpose": "living"}
    return {}


# Ready-possession preference (overseas / document-cautious buyers).
_READY_POSSESSION_KW = (
    "ready possession", "ready to move", "ready-to-move", "move-in ready", "move in ready",
    "want possession", "i want possession", "ready property", "ready flat", "no files",
    "no file", "no under construction", "no under-construction", "not under construction",
    "no booking", "no future possession", "immediate possession", "completed project",
)
# Signals that make the agent proactively raise possession/legal clarity.
_POSSESSION_SIGNAL_KW = (
    "abroad", "overseas", "move back", "moving back", "relocat", "possession",
    "under construction", "under-construction", "no files", "no file", "document",
    "legal", "ownership", "transfer", "booking", "dispute", "fraud", "bad experience",
)

def infer_possession(query: str) -> dict:
    q = " " + query.lower() + " "
    if any(k in q for k in _READY_POSSESSION_KW):
        return {"possession": "ready"}
    return {}


# Soft, additive LLM catch-all. The keyword rules above are deterministic but blind to
# the long tail ("my wife is pregnant", "my father uses a wheelchair", "we host guests
# often"). This pass reasons over the message and returns ONLY soft hint fields — every
# key here is a rerank signal (see _SOFT_SIT_BONUS / BED_FLOOR_BONUS), NEVER a hard
# filter. It deliberately CANNOT emit budget, an exact bedroom count, a location, or a
# property type — those stay with the deterministic parsers — so it can never resurface
# the over-budget / hallucinated-filter bugs we engineered out. Worst case it adds a
# slightly-off rerank nudge, which the search already tolerates.
_SOFT_CATCHALL_BOOLS = ("near_hospital", "near_school", "near_park", "near_masjid",
                        "near_market", "near_restaurant", "near_transport", "gated_community")

def _infer_situation_and_ack(user_query: str, lang: str = "English") -> tuple:
    """ONE LLM call doing two jobs that share the same understanding of the message:
    (1) the soft situational hints (rerank-only fields), and (2) a short warm
    acknowledgement of anything personal the buyer shared. Returns (hints_dict,
    ack_str). The conservative prompt keeps the fast 8B's over-firing at bay (it
    invented bedrooms_floor/near_* on plain budget answers); the capable model is
    used. Returns ({}, "") on trivial input, doubt, or any parse failure."""
    q = (user_query or "").strip()
    if len(q.split()) < 3:   # trivial / single-tap pill answers need neither pass
        return {}, ""
    prompt = f"""A property buyer said: "{user_query}"

Return a JSON object with EXACTLY two keys, "hints" and "ack".

"hints": an object of SOFT situational filters the buyer's words DIRECTLY imply (be conservative — when in doubt, omit). Allowed keys ONLY:
- "bedrooms_floor": integer — the MINIMUM bedrooms implied by WHO will live there, ONLY when implied without an exact number (couple expecting a baby -> 3; parents + kids living together -> 4). Omit if the message gives no household detail.
- "near_hospital","near_school","near_park","near_masjid","near_market","near_restaurant","near_transport","gated_community": true — ONLY when that specific need is directly implied (elderly/medical -> near_hospital; school-going kids -> near_school; explicit safety/security -> gated_community; regular prayer/masjid -> near_masjid).
NEVER include budget, price, an exact requested bedroom count, a location/area, or a property type. A plain budget ("3.5 crore"), a bare area name, or a generic "for my family" with no further detail -> "hints": {{}}.

"ack": a SHORT, warm, natural ONE-sentence acknowledgement of the specific personal or situational thing they shared (a baby on the way, an elderly parent, a safety worry, a firm requirement) — like a thoughtful human agent. NO question, NO budget/price/area names, NO advice or lists. If they shared nothing noteworthy (just a plain answer like a budget figure or an area name), use an empty string "". Write the "ack" ONLY in {lang}.

Output ONLY the JSON object."""
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f">> _infer_situation_and_ack failed: {e}")
        return {}, ""
    hints_in = data.get("hints") if isinstance(data.get("hints"), dict) else {}
    hints = {}
    bf = hints_in.get("bedrooms_floor")
    if isinstance(bf, bool):           # JSON true/false sneaks past int checks
        bf = None
    if isinstance(bf, int) and 1 <= bf <= 7:
        hints["bedrooms_floor"] = bf
    for k in _SOFT_CATCHALL_BOOLS:
        if hints_in.get(k) is True:
            hints[k] = True
    ack = (data.get("ack") or "").strip().strip('"')
    if len(ack) > 220 or ack.endswith("?") or ack.endswith("؟"):  # guard padding/questions
        ack = ""
    return hints, ack


# Deterministic Karachi-area capture — the small LLM frequently drops a bare area
# answer ("Gulshan or Johar"), so we also match against known areas directly.
# Canonical (left) is a substring of the data's location field; aliases (right)
# are what users actually type. Longer aliases are consumed first so overlaps
# ("north nazimabad" vs "nazimabad") don't double-match.
_AREA_ALIASES = {
    "Gulshan-e-Hadeed": ["gulshan-e-hadeed", "hadeed"],
    "Gulshan-e-Maymar": ["gulshan-e-maymar", "maymar"],
    "Gulistan-e-Johar": ["gulistan-e-johar", "gulistan e johar", "gulshan-e-johar", "johar"],
    "Gulshan-e-Iqbal": ["gulshan-e-iqbal", "gulshan e iqbal", "gulshan iqbal", "gulshan"],
    "North Nazimabad": ["north nazimabad"],
    "Naya Nazimabad": ["naya nazimabad"],
    "Nazimabad": ["nazimabad"],
    "North Karachi": ["north karachi"],
    "New Karachi": ["new karachi"],
    "DHA": ["dha", "defence", "defense"],
    "Clifton": ["clifton"],
    "Bath Island": ["bath island"],
    "PECHS": ["pechs"],
    "SMCHS": ["smchs"],
    "Bahadurabad": ["bahadurabad"],
    "Tariq Road": ["tariq road"],
    "FB Area": ["fb area", "f.b. area", "f b area"],
    "Bahria": ["bahria"],
    "Scheme 33": ["scheme 33", "scheme-33", "scheme thirty"],
    "Saadi Town": ["saadi town", "saadi garden", "saadi"],
    "Korangi": ["korangi"],
    "Landhi": ["landhi"],
    "Malir": ["malir"],
    "Liaquatabad": ["liaquatabad", "lalukhet"],
    "Surjani Town": ["surjani"],
    "Orangi Town": ["orangi"],
    "Baldia Town": ["baldia"],
    "Gadap Town": ["gadap"],
    "Model Colony": ["model colony"],
    "Shah Faisal Colony": ["shah faisal"],
    "Safoora Goth": ["safoora"],
    "Steel Town": ["steel town"],
    "Bin Qasim": ["bin qasim"],
    "KDA": ["kda"],
    "Askari": ["askari"],
}
_AREA_ALIAS_PAIRS = sorted(
    ((alias, canon) for canon, aliases in _AREA_ALIASES.items() for alias in aliases),
    key=lambda p: -len(p[0]),
)
# Rejection cues near an area mean the user is ruling it OUT, not asking for it.
_AREA_REJECT_CUES = ("far", "not ", "n't", "avoid", "exclude", "skip", "nahi", "nhi",
                     "door", " dur", "except", "other than", "apart from", "besides", "alag")

def _match_karachi_areas(text: str) -> list:
    t = " " + text.lower() + " "
    found = []
    for alias, canon in _AREA_ALIAS_PAIRS:
        pat = r'(?<![a-z])' + re.escape(alias) + r'(?![a-z])'
        m = re.search(pat, t)
        if not m:
            continue
        # Skip if a rejection cue sits right around the mention ("Landhi feels far").
        window = t[max(0, m.start() - 18): m.end() + 18]
        if any(cue in window for cue in _AREA_REJECT_CUES):
            t = re.sub(pat, " ", t)
            continue
        if canon not in found:
            found.append(canon)
        t = re.sub(pat, " ", t)  # consume so shorter overlapping aliases don't re-match
    return found


# Commute landmarks (roads/hubs people anchor on) → nearby residential areas with
# stock, so "office near Shahrah-e-Faisal" doesn't get stored as a non-matching
# "location" and fall back to far-flung results.
_LANDMARK_AREAS = {
    ("shahrah-e-faisal", "shahra-e-faisal", "shahrah e faisal", "sharea faisal", "shahrae faisal"):
        ["PECHS", "Bahadurabad", "Gulshan-e-Iqbal", "Tariq Road", "KDA"],
    ("saddar", "i.i. chundrigar", "ii chundrigar", "tower ", "business district", "city centre"):
        ["PECHS", "Clifton", "Bahadurabad"],
    ("airport", "faisal cantt", "drigh road"):
        ["Gulistan-e-Johar", "Malir", "Shah Faisal Colony"],
    ("site area", "site industrial", "korangi industrial"):
        ["North Nazimabad", "Nazimabad", "Korangi"],
}

def _landmark_to_areas(query: str) -> list:
    q = " " + query.lower() + " "
    for keys, areas in _LANDMARK_AREAS.items():
        if any(k in q for k in keys):
            return areas
    return []


_BED_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5}
_BED_RE = re.compile(
    r'(\d+|one|two|three|four|five|six|ek|do|teen|char|paanch)\s*-?\s*'
    r'(?:bed|beds|bedroom|bedrooms|bhk)\b', re.IGNORECASE)

def _to_n(tok):
    return int(tok) if tok.isdigit() else _BED_WORD.get(tok)

def _parse_bed_requirement(query: str):
    """Deterministic bed requirement → (value, kind) where kind is 'exact' or 'min'.
    Only looks at bed-adjacent patterns so a budget range ('3.5 to 4 crore') never
    leaks into the bedroom parse. Ranges/'at least'/'ideally' become a minimum
    (the LOWER bound); a bare 'N bed' is exact."""
    q = query.lower()
    num = r'(\d+|one|two|three|four|five|six|ek|do|teen|char|paanch)'
    # "2 or 3 bed", "2 to 3 bed", "2-3 bed" → min of the two
    m = re.search(num + r'\s*(?:or|to|-)\s*' + num + r'\s*-?\s*(?:bed|beds|bedroom|bedrooms|bhk)', q)
    if m:
        a, b = _to_n(m.group(1)), _to_n(m.group(2))
        if a and b:
            return (min(a, b), "min")
    # "at least 4", "minimum 3", "3+ bed"
    m = re.search(r'(?:at\s*least|atleast|minimum|min|more than|over)\s*' + num, q)
    if m and _to_n(m.group(1)):
        return (_to_n(m.group(1)), "min")
    m = re.search(num + r'\s*\+\s*(?:bed|beds|bedroom|bedrooms|bhk)', q)
    if m and _to_n(m.group(1)):
        return (_to_n(m.group(1)), "min")
    # plain "N bed"; if "ideally/prefer" present, treat as a minimum (open to more)
    m = _BED_RE.search(q)
    if m:
        n = _to_n(m.group(1))
        if n is not None:
            return (n, "min" if ("ideally" in q or "prefer" in q) else "exact")
    return (None, None)


def _count_where(filters: dict):
    """Build a metadata-only where clause for an EXACT match count. Unlike
    build_chroma_filter this DOES include price + situational fields, because
    .get() does a metadata scan rather than HNSW search."""
    conds = []
    # Mirror build_chroma_filter exactly so the live count matches the eventual
    # reveal search. bedrooms_floor is a SOFT family hint, not hard-filtered there,
    # so it must not be hard-filtered here either (it would over-narrow the count).
    if filters.get("bedrooms") is not None:
        conds.append({"bedrooms": {"$eq": filters["bedrooms"]}})
    elif filters.get("bedrooms_min") is not None:
        conds.append({"bedrooms": {"$gte": filters["bedrooms_min"]}})
    if filters.get("min_bathrooms") is not None:
        conds.append({"bathrooms": {"$gte": filters["min_bathrooms"]}})
    types = filters.get("types") or ([filters["type"]] if filters.get("type") else [])
    types = [t for t in types if isinstance(t, str)]
    if len(types) == 1:
        conds.append({"type": {"$eq": types[0]}})
    elif len(types) > 1:
        conds.append({"type": {"$in": types}})
    if filters.get("max_price"):
        ceil = int(price_to_lacs(str(filters["max_price"])) * 1.1)
        if ceil > 0:
            conds.append({"price_numeric": {"$lte": ceil}})
    if filters.get("min_price"):
        floor = int(price_to_lacs(str(filters["min_price"])) * 0.9)
        if floor > 0:
            conds.append({"price_numeric": {"$gte": floor}})
    # near_*/gated/floor==0 are SOFT (handled in _rerank_by_amenities), not hard
    # filters — see _SOFT_SIT_BONUS. Only explicit structural prefs are hard here.
    if filters.get("floor_band"):
        conds.append({"floor_band": {"$eq": filters["floor_band"]}})
    if filters.get("has_lift"):
        conds.append({"has_lift": {"$eq": True}})
    if filters.get("possession"):
        conds.append({"possession": {"$eq": filters["possession"]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def count_matches(filters: dict) -> int:
    """Exact count of listings matching accumulated filters. Location is
    substring-matched in Python (Chroma where has no contains operator)."""
    where = _count_where(filters)
    try:
        got = vectorstore._collection.get(where=where, include=["metadatas"])
        metas = got.get("metadatas", []) or []
    except Exception as e:
        print(f">> count_matches failed: {e}")
        return 0
    locations = [l.lower() for l in (filters.get("locations") or []) if isinstance(l, str)]
    if locations:
        metas = [m for m in metas if any(loc in (m.get("location", "").lower()) for loc in locations)]
    return len(metas)


# "No preference / you decide" answers. For dims where an empty value is a legit
# answer (area, type, floor, extras, possession, purpose), the buyer saying "open to
# your suggestions" must RESOLVE the dim — otherwise extraction finds nothing and we
# keep re-asking the same question until the ask-limit runs out.
_NO_PREF_PATTERNS = (
    "open to suggestion", "open to your suggestion", "your suggestion", "you suggest",
    "you decide", "up to you", "whatever you", "no preference", "doesn't matter",
    "does not matter", "doesnt matter", "any area", "anywhere", "any is fine",
    "anything is fine", "anything works", "no specific", "not specific", "i'm open",
    "im open", "i am open", "you choose", "you pick", "as you suggest", "as you see fit",
    # roman urdu
    "koi bhi", "aap batao", "aap bata", "aap suggest", "jo aap", "aap decide",
    "koi preference nahi", "koi khaas nahi", "aap k hisab", "aap ke hisab",
)

def _is_no_preference(query: str) -> bool:
    q = " " + (query or "").lower() + " "
    return any(p in q for p in _NO_PREF_PATTERNS)

# dim → the filter keys that, once set, mean the buyer DID give a concrete value
# (so a no-preference phrase shouldn't override it).
_DIM_VALUE_KEYS = {
    "area": ("locations",),
    "type": ("types",),
    "floor": ("floor_band", "has_lift"),
    "extras": ("features",),
    "possession": ("possession",),
    "purpose": ("purpose",),
}


def _merge_discovery_filters(state: dict, user_query: str, lang: str = "English") -> list:
    """Extract + infer from the user's message, merge into accumulated state.
    Walk-back: if the newly inferred situational filters zero out the result
    set, drop them (keeping explicit ones) and report which were relaxed."""
    # extract_filters returns keys for unmentioned fields as None/[]/0; drop those
    # so a later turn never wipes a value the user already gave (e.g. budget) and a
    # stray bedrooms:0 from the LLM never collapses the result set to nothing.
    explicit = {k: v for k, v in extract_filters(user_query).items() if v not in (None, [], "", {}, 0)}
    # Drop hallucinated types: extract_filters sometimes returns every property type
    # for a message that never mentions one (e.g. a floor-preference turn). Only
    # trust extracted types when the message actually names a type.
    # Deterministic type capture is AUTHORITATIVE (the small LLM often returns {} for
    # a bare "apartment"). Fall back to the LLM's types only if our matcher finds none
    # and the message actually names a type.
    matched_types = _match_types(user_query)
    if matched_types:
        explicit["types"] = matched_types
    elif explicit.get("types") and not _mentions_type(user_query):
        explicit.pop("types")
    # Deterministic area capture. When our matcher finds areas it is AUTHORITATIVE
    # (it's rejection-aware), overriding the LLM's locations for this turn — the LLM
    # isn't rejection-aware and would re-add an area the user just ruled out
    # ("Landhi feels far"). Only fall back to the LLM's locations if we match none.
    # Known Karachi areas first; else a commute landmark maps to nearby areas.
    matched_areas = _match_karachi_areas(user_query) or _landmark_to_areas(user_query)
    if matched_areas:
        explicit["locations"] = matched_areas
    elif explicit.get("locations"):
        # LLM extracted something we don't recognise as an area (often a road/landmark
        # that matches no listings) — drop it so it can't silently break the search.
        explicit.pop("locations", None)
    # Budget + bedrooms: trust ONLY deterministic parsing. The LLM hallucinates these
    # (a budget on a turn that mentioned no money; "8 people" → 2 beds), and a stray
    # value would clobber the correct accumulated one. Implicit beds still come from
    # household inference below.
    ql = user_query.lower()
    det_ceiling = _parse_budget_ceiling(user_query)
    det_floor = _parse_budget_floor(user_query)
    # "above/over X" with no upper bound is a FLOOR, not a ceiling — the ceiling
    # parser would otherwise misread the same number as a max.
    floor_only = (det_floor is not None
                  and bool(re.search(r'\b(?:above|over|more than|minimum|at\s*least|starting)\b', ql))
                  and not any(w in ql for w in (" to ", "between", "-")))
    if det_floor:
        explicit["min_price"] = det_floor
    if floor_only:
        explicit.pop("max_price", None)
    elif det_ceiling:
        explicit["max_price"] = det_ceiling
    else:
        explicit.pop("max_price", None)
    explicit.pop("bedrooms", None)
    bed_n, bed_kind = _parse_bed_requirement(user_query)
    if bed_n:
        if bed_kind == "min":
            # HARD minimum (>=), python-filtered so it doesn't break HNSW retrieval.
            explicit["bedrooms_min"] = bed_n
        else:
            explicit["bedrooms"] = bed_n
    inferred = {}
    inferred.update(infer_situational_filters(user_query))
    inferred.update(infer_floor_pref(user_query))
    inferred.update(infer_extras(user_query))
    inferred.update(infer_household(user_query))
    inferred.update(infer_purpose(user_query))
    inferred.update(infer_possession(user_query))
    if any(k in (" " + user_query.lower() + " ") for k in _POSSESSION_SIGNAL_KW):
        state["possession_relevant"] = True

    # Soft catch-all + acknowledgement in ONE LLM call (they share the same reading of
    # the message). Hints fold in additively — bedrooms_floor keeps the HIGHER of rule
    # vs catch-all (more rooms wins for a soft floor); bools union as True. The ack is
    # stashed for the wording layer to prepend when this is a templated question turn.
    soft, ack = _infer_situation_and_ack(user_query, lang)
    state["_last_soft"] = soft
    state["_last_ack"] = ack
    if soft.get("bedrooms_floor") and inferred.get("bedrooms_floor"):
        soft["bedrooms_floor"] = max(soft["bedrooms_floor"], inferred["bedrooms_floor"])
    inferred.update(soft)

    # Merge into accumulated state. List-valued filters (locations, types,
    # amenities_wanted, features) UNION across turns; scalars are overwritten with
    # explicit winning over inferred winning over what we already had.
    before_locs = list(state["filters"].get("locations") or [])
    merged = dict(state["filters"])
    for src in (inferred, explicit):
        for k, v in src.items():
            if isinstance(v, list):
                merged[k] = list(dict.fromkeys([*(merged.get(k) or []), *v]))
            else:
                merged[k] = v

    # Areas: a new area answer usually REPLACES the old (esp. when the buyer pivots,
    # "PECHS or Gulshan then" instead of DHA/Clifton) — only ADD when they say so.
    if matched_areas:
        add_cue = any(c in (" " + user_query.lower() + " ")
                      for c in (" also", " add ", "as well", " plus ", " and also", " too "))
        merged["locations"] = (list(dict.fromkeys([*before_locs, *matched_areas]))
                               if add_cue else matched_areas)

    # An explicit floor-band preference ("middle floor") overrides the inferred
    # ground-floor (floor==0) that a stray "mother"/"parents" mention may have set.
    if merged.get("floor_band") and "floor" in merged:
        merged.pop("floor")

    # Budget fallback: extract_filters misses ranges like "2.2 to 2.6 crore".
    # Capture the upper bound from the raw message so a budget answer is never lost.
    # Skip on a floor-only turn ("above 7 crore") — there the number is a min, not a max.
    if not merged.get("max_price") and not floor_only:
        ceiling = _parse_budget_ceiling(user_query)
        if ceiling:
            merged["max_price"] = ceiling

    # Reconcile the band floor: a fresh single-ceiling answer (no range/floor words)
    # clears an earlier min; a floor-only answer clears an earlier ceiling.
    if det_ceiling and det_floor is None and not floor_only:
        merged.pop("min_price", None)
    if floor_only:
        merged.pop("max_price", None)

    # Walk-back: only the explicit STRUCTURAL prefs (middle floor, lift) are hard
    # filters that can over-narrow; the near_*/gated/floor==0 fields are soft now.
    # Relax floor_band before has_lift (lift is the stated "must"). Never relax the
    # user's budget/area/type/bedrooms.
    dropped = []
    sit_order = ["floor_band", "has_lift"]
    present = [f for f in sit_order if f in merged]
    if present and count_matches(merged) == 0:
        relaxed = None
        for f in present:                       # prefer dropping a single field
            trial = {k: v for k, v in merged.items() if k != f}
            if count_matches(trial) > 0:
                relaxed, dropped = trial, [f]
                break
        if relaxed is None:                      # no single drop helps — drop progressively
            trial = dict(merged)
            for f in present:
                trial.pop(f, None)
                dropped.append(f)
                if count_matches(trial) > 0:
                    break
            relaxed = trial
        merged = relaxed

    # "Open to your suggestions" on the dim we just asked → mark it RESOLVED so we
    # stop re-asking and move the consultation forward. Only when the buyer gave no
    # concrete value for that dim this turn (an answer like "Gulshan, or you suggest"
    # still pins Gulshan and is not treated as no-preference).
    last_dim = state.get("last_dim")
    if last_dim in _DIM_VALUE_KEYS and _is_no_preference(user_query):
        if not any(merged.get(k) for k in _DIM_VALUE_KEYS[last_dim]):
            state.setdefault("resolved", set()).add(last_dim)

    state["filters"] = merged
    return dropped


MAX_ASK_PER_DIM = 2  # re-ask an unanswered spec dimension at most twice

def _next_discovery_dim(state: dict):
    """Deterministic, human-shaped order: purpose → household → type → area/budget
    opener → budget → possession → area → floor/lift → extras, then reveal (no
    confirmation step — when the requirements are clear, re-asking "did I get that
    right?" just wastes a turn). Understanding the HOME (who lives there, house vs
    apartment) before money/area matches how a real agent consults; when the buyer
    already stated beds+type up front those are skipped and the area/budget opener
    leads, which is exactly what we want. (A measured A/B showed a fixed order beats
    letting an LLM pick — cheaper + stable.) A dim is skipped once KNOWN; re-asked up
    to MAX_ASK_PER_DIM if still unknown (extraction is flaky); never loops."""
    f = state["filters"]
    asked = state["asked"]
    resolved = state.get("resolved") or set()  # dims the buyer left to us ("you suggest")

    def need(dim, known):
        return not known and dim not in resolved and asked.get(dim, 0) < MAX_ASK_PER_DIM

    if not f.get("purpose") and "purpose" not in resolved and asked.get("purpose", 0) < 1:
        return "purpose"
    # Household before money/area. A CONFIRMED bedroom count suppresses it; but a soft
    # bedrooms_floor (inferred from "two kids, my mother") only counts AFTER we've asked
    # once — so the buyer gets the question + bedroom pills, yet describing the family
    # rather than naming a number doesn't loop us (the soft floor still drives the
    # search). Without this guard household re-asks until its limit and reveals early.
    household_known = (f.get("bedrooms") or f.get("bedrooms_min")
                       or (f.get("bedrooms_floor") and asked.get("household", 0) >= 1))
    if need("household", household_known):
        return "household"
    if need("type", f.get("types")):
        return "type"
    # Smart opener: when we know NEITHER budget NOR area, don't fire a bare "what's
    # your budget?" — ask the one branching question a real agent asks ("do you have
    # areas in mind, or should I suggest based on budget?"). It lets the buyer answer
    # either way and routes the rest of the flow (fixed-area → honest budget check;
    # no area → budget-first → area suggestions). Asked at most once.
    budget_known = f.get("max_price") or f.get("min_price")
    if (not budget_known and not f.get("locations")
            and "area" not in resolved and asked.get("area_budget", 0) < 1):
        return "area_budget"
    # Budget counts as answered with EITHER a ceiling or a floor — a "above X" /
    # "minimum X" answer sets only min_price, and must not loop the budget question.
    if need("budget", f.get("max_price") or f.get("min_price")):
        return "budget"
    # Possession is a top concern for overseas / document-cautious buyers — ask it
    # right after budget, the way a real agent does, not buried at the end.
    if state.get("possession_relevant") and need("possession", f.get("possession")):
        return "possession"
    if need("area", f.get("locations")):
        return "area"
    if need("floor", f.get("floor_band") or f.get("has_lift")):
        return "floor"
    if "extras" not in resolved and asked.get("extras", 0) < 1:
        return "extras"
    return None


def _discovery_ready(state: dict) -> bool:
    return state["turns"] >= DISCOVERY_MAX_QUESTIONS or _next_discovery_dim(state) is None


_DISCOVERY_DIM_PROMPT = {
    "purpose": "Open by asking, naturally, whether this is for their own living or for investment — it "
               "shapes everything that follows. One short, warm question.",
    "household": "Ask how many family members will live there and who (kids, parents), so you can judge "
                 "the right number of bedrooms. Warm and brief.",
    "type": "Ask whether they prefer a house, apartment, or portion (and note if they're open to portions).",
    "budget": "Ask warmly what budget you should keep in mind. Just ask their budget plainly — "
              "NEVER say the words 'lac' or 'crore' (it's understood they'll answer in those).",
    "area": "Ask which area(s) of Karachi they prefer, or whether they're open to your suggestions.",
    "floor": "Ask their floor preference (ground / middle / top) and whether a lift is required — this "
             "matters for elderly family members visiting.",
    "possession": "Ask whether they need ready possession / ready-to-move only (with clear documents), "
                  "or are open to under-construction — important for overseas or document-cautious buyers.",
    "extras": "Ask if there are any must-have features or nice-to-haves — e.g. dedicated parking, "
              "generator/electricity backup, drawing room, west-open, good maintenance, schools nearby. "
              "Ask it naturally, not as a long checklist.",
}


# Per-area typical price band, computed from the actual listings (not synthetic),
# for the requested type+bedrooms — powers the "expected pricing in Gulshan vs
# Johar" advisory and area prioritization.
def _area_price_bands(filters: dict) -> str:
    locs = filters.get("locations") or []
    if not locs:
        return ""
    conds = []
    types = [t for t in (filters.get("types") or []) if isinstance(t, str)]
    if len(types) == 1:
        conds.append({"type": {"$eq": types[0]}})
    elif len(types) > 1:
        conds.append({"type": {"$in": types}})
    if filters.get("bedrooms") is not None:
        conds.append({"bedrooms": {"$eq": filters["bedrooms"]}})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
    try:
        metas = vectorstore._collection.get(where=where, include=["metadatas"]).get("metadatas", []) or []
    except Exception:
        return ""
    lines = []
    for loc in locs:
        ll = loc.lower()
        prices = sorted(int(m.get("price_numeric") or 0)
                        for m in metas if ll in (m.get("location", "").lower()) and m.get("price_numeric"))
        if not prices:
            lines.append(f"{loc}: no matching listings in our data")
            continue
        lo = prices[len(prices) // 4]
        hi = prices[min(len(prices) - 1, (3 * len(prices)) // 4)]
        lines.append(f"{loc}: typically {lacs_to_price(lo)}–{lacs_to_price(hi)} ({len(prices)} listings)")
    return "\n".join(lines)


def _areas_for_budget(filters: dict, limit: int = 6) -> list:
    """Parent areas where the buyer's budget is a GOOD FIT for their type + beds,
    ranked PREMIUM-FIRST — the way a real buyer thinks ("I'd rather a 3.75cr flat in
    Gulshan than in Malir"). An area is eligible only if it has a real cluster of
    in-budget stock (depth ≥ 3) AND the budget reaches at least its 25th-percentile
    price (so the buyer can afford something decent there, not just the dregs). Among
    eligible areas we rank by overall price level (median for this type+beds), so the
    more desirable area surfaces first. Note: we deliberately do NOT chase the area
    where the budget is merely 'typical' — that just surfaces value areas where the
    budget tops out the local market."""
    conds = []
    types = [t for t in (filters.get("types") or []) if isinstance(t, str)]
    if len(types) == 1:
        conds.append({"type": {"$eq": types[0]}})
    elif len(types) > 1:
        conds.append({"type": {"$in": types}})
    if filters.get("bedrooms") is not None:
        conds.append({"bedrooms": {"$eq": filters["bedrooms"]}})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
    try:
        metas = vectorstore._collection.get(where=where, include=["metadatas"]).get("metadatas", []) or []
    except Exception:
        return []
    from collections import Counter, defaultdict

    def parent(loc):
        a = loc.split(",")[0].strip()
        return re.sub(r'\s+(?:phase|block|sector|extension|scheme)\s+\S+.*$', '', a, flags=re.IGNORECASE).strip()

    by_area = defaultdict(list)
    for m in metas:
        loc, p = m.get("location"), int(m.get("price_numeric") or 0)
        if loc and p:
            by_area[parent(loc)].append(p)

    max_lacs = price_to_lacs(str(filters["max_price"])) if filters.get("max_price") else 0
    min_lacs = price_to_lacs(str(filters["min_price"])) if filters.get("min_price") else 0

    if not max_lacs and not min_lacs:  # no budget yet → fall back to most-stock areas
        counts = Counter({a: len(ps) for a, ps in by_area.items() if a})
        return [a for a, _ in counts.most_common(limit)]

    ceil = int(max_lacs * 1.1) if max_lacs else 0
    floor = int(min_lacs * 0.9) if min_lacs else 0
    MIN_DEPTH = 3  # need a real cluster of in-budget stock, not one fluke listing

    def _rank(min_depth: int):
        scored = []
        for area, prices in by_area.items():
            if not area:
                continue
            ps = sorted(prices)
            in_budget = [p for p in ps if (not ceil or p <= ceil) and (not floor or p >= floor)]
            if len(in_budget) < min_depth:
                continue
            # Budget must reach at least the area's lower-middle (25th pct), else the
            # buyer is priced into only the cheapest dregs of a too-premium area. This
            # drops DHA/Clifton at a mid budget automatically — no tier table needed.
            p25 = ps[len(ps) // 4]
            if max_lacs and ceil < p25:
                continue
            # Premium-first: rank by the area's overall price level (desirability proxy
            # for this type+beds), then by how much attainable in-budget stock it has.
            median = ps[len(ps) // 2]
            scored.append((-median, -len(in_budget), area))
        scored.sort()
        return [a for _, _, a in scored[:limit]]

    # Prefer areas with a real in-budget cluster; if none qualify, relax depth so we
    # still surface suggestions rather than handing back an empty screen.
    return _rank(MIN_DEPTH) or _rank(1)


# ── Context-grounded discovery pills ────────────────────────────────────────
# Tappable quick-replies offered alongside a discovery question. They are derived
# from the ACTUAL matching stock for what the buyer has said so far, so the budget
# bands start at the real floor price for (area + type + beds), the bedroom chips
# only show counts that exist, etc. Pills are purely additive — each one just sends
# the same text a user could type, and free-typing is always accepted.

def _nice_round(n: int, down: bool = False) -> int:
    """Round a lac amount to a human-friendly boundary (25 lac under 1cr, 50 lac
    under 10cr, else 1cr)."""
    if n <= 0:
        return 0
    step = 25 if n < 100 else (50 if n < 1000 else 100)
    return (n // step) * step if down else ((n + step - 1) // step) * step


def _matching_metas(filters: dict, drop: tuple = ()):
    """Listing metadatas matching the accumulated filters, optionally ignoring some
    keys (drop the dimension we're about to ask about, so its pills span the real
    available range rather than collapsing to whatever's already pinned)."""
    f = {k: v for k, v in filters.items() if k not in drop}
    try:
        metas = vectorstore._collection.get(where=_count_where(f), include=["metadatas"]).get("metadatas", []) or []
    except Exception:
        return []
    if "locations" not in drop:
        locs = [l.lower() for l in (filters.get("locations") or []) if isinstance(l, str)]
        if locs:
            metas = [m for m in metas if any(loc in (m.get("location", "").lower()) for loc in locs)]
    return metas


def _budget_pills(filters: dict) -> list:
    metas = _matching_metas(filters, drop=("max_price", "min_price"))
    prices = sorted(int(m.get("price_numeric") or 0) for m in metas if m.get("price_numeric"))
    if len(prices) < 6:
        return []  # too little stock to band meaningfully — let them type
    q1 = _nice_round(prices[len(prices) // 4])
    q2 = _nice_round(prices[len(prices) // 2])
    q3 = _nice_round(prices[(3 * len(prices)) // 4])
    cuts = sorted({c for c in (q1, q2, q3) if c > 0})
    if len(cuts) < 2:
        return []
    pills = [{"label": f"Under {lacs_to_price(cuts[0])}", "value": f"under {lacs_to_price(cuts[0])}"}]
    for lo, hi in zip(cuts, cuts[1:]):
        pills.append({"label": f"{lacs_to_price(lo)} – {lacs_to_price(hi)}",
                      "value": f"between {lacs_to_price(lo)} and {lacs_to_price(hi)}"})
    pills.append({"label": f"Above {lacs_to_price(cuts[-1])}", "value": f"above {lacs_to_price(cuts[-1])}"})
    return pills


def _bedroom_pills(filters: dict) -> list:
    metas = _matching_metas(filters, drop=("bedrooms", "bedrooms_min"))
    counts = sorted({int(m.get("bedrooms") or 0) for m in metas if m.get("bedrooms")})
    pills, capped = [], False
    for c in counts:
        if c <= 0:
            continue
        if c >= 5:
            if not capped:
                pills.append({"label": "5+ bed", "value": "5+ bedrooms"})
                capped = True
        else:
            pills.append({"label": f"{c} bed", "value": f"{c} bedrooms"})
    return pills


_TYPE_LABELS = {"house": "House", "apartment": "Apartment", "upper portion": "Upper portion",
                "lower portion": "Lower portion", "penthouse": "Penthouse", "farmhouse": "Farmhouse"}

def _type_pills(filters: dict) -> list:
    metas = _matching_metas(filters, drop=("types", "type"))
    types = sorted({(m.get("type") or "").lower() for m in metas if m.get("type")})
    return [{"label": _TYPE_LABELS.get(t, t.title()), "value": t} for t in types if t][:6]


def _family_areas_for_budget(filters: dict, limit: int = 6) -> list:
    """Realistic areas to SUGGEST when the buyer has a budget but no area yet, ranked by
    how much IN-BUDGET stock each area has of the buyer's EXACT size (most options
    first). Exact-size first is deliberate: suggesting areas off a ±1 bedroom window
    surfaced nicer names (Johar, North Nazimabad) that then had ZERO exact stock at the
    buyer's budget, so the reveal contradicted the suggestion. We only widen to the ±1
    window when the exact slice is too sparse to suggest from (<3 areas), and only then
    fall back to cheapest areas. No budget yet → most-stock areas."""
    types = [t for t in (filters.get("types") or []) if isinstance(t, str)]
    beds = filters.get("bedrooms")
    max_lacs = price_to_lacs(str(filters["max_price"])) if filters.get("max_price") else 0
    from collections import defaultdict

    def parent(loc):
        a = loc.split(",")[0].strip()
        return re.sub(r'\s+(?:phase|block|sector|extension|scheme)\s+\S+.*$', '', a, flags=re.IGNORECASE).strip()

    def _by_area(lo_b, hi_b):
        conds = []
        if len(types) == 1:
            conds.append({"type": {"$eq": types[0]}})
        elif len(types) > 1:
            conds.append({"type": {"$in": types}})
        if beds:
            conds.append({"bedrooms": {"$gte": lo_b}})
            conds.append({"bedrooms": {"$lte": hi_b}})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        try:
            metas = vectorstore._collection.get(where=where, include=["metadatas"]).get("metadatas", []) or []
        except Exception:
            return {}
        ba = defaultdict(list)
        for m in metas:
            loc, p = m.get("location"), int(m.get("price_numeric") or 0)
            if loc and p:
                ba[parent(loc)].append(p)
        return ba

    def _rank(by_area):
        if not max_lacs:  # no budget yet → most-stock areas
            ranked = sorted(((a, ps) for a, ps in by_area.items() if a), key=lambda kv: -len(kv[1]))
            return [a for a, _ in ranked][:limit]
        ceil = int(max_lacs * 1.1)
        scored = []
        for area, prices in by_area.items():
            if not area:
                continue
            in_budget = [p for p in prices if p <= ceil]
            if len(in_budget) < 2:  # need a small real cluster, not one fluke
                continue
            median = sorted(prices)[len(prices) // 2]
            scored.append((-len(in_budget), median, area))  # most in-budget stock first
        scored.sort()
        return [a for _, _, a in scored[:limit]]

    # Exact size first — what the reveal can actually deliver. Widen only if too sparse.
    exact = _rank(_by_area(beds, beds)) if beds else _rank(_by_area(None, None))
    if len(exact) >= 3 or not beds or not max_lacs:
        if exact:
            return exact
    window = _by_area(max(1, beds - 1), beds + 1)
    out = _rank(window)
    if out:
        return out
    # Budget reaches almost nothing even widened — name the cheapest areas so we still
    # point them somewhere real rather than handing back nothing.
    ranked = sorted(((a, ps) for a, ps in window.items() if a), key=lambda kv: sorted(kv[1])[0])
    return [a for a, _ in ranked][:limit]


def _area_pills(filters: dict) -> list:
    return [{"label": a, "value": a} for a in _family_areas_for_budget(filters, limit=6)]


def _floor_pills(filters: dict) -> list:
    return [
        {"label": "Ground floor", "value": "ground floor"},
        {"label": "Middle floor", "value": "middle floor"},
        {"label": "Top floor", "value": "top floor"},
        {"label": "No preference", "value": "no floor preference"},
    ]


def _must_have_pills(filters: dict) -> list:
    """Tappable must-haves for the area-discussion turn — values are phrased so the
    extractors/inference pick them up (lift→has_lift, the rest→features)."""
    return [
        {"label": "Lift", "value": "lift is a must"},
        {"label": "Parking", "value": "dedicated parking"},
        {"label": "Backup", "value": "generator backup"},
        {"label": "Family building", "value": "proper family building"},
        {"label": "All of these", "value": "lift, parking, backup and a family building are all must-haves"},
    ]


# dim → (builder, is_multi_select). Dims not listed (purpose, possession, extras,
# recap) get no pills; the user free-types those.
_DISCOVERY_PILL_BUILDERS = {
    "budget": (_budget_pills, False),
    "household": (_bedroom_pills, False),   # the bedrooms question
    "type": (_type_pills, False),
    "area": (_area_pills, True),
    "floor": (_floor_pills, False),
    "area_intel": (_must_have_pills, True),  # the block-level area discussion turn
}

def _discovery_options(dim, filters: dict) -> dict:
    """{"options": [{label, value}...], "multi": bool} for a discovery question."""
    fn_multi = _DISCOVERY_PILL_BUILDERS.get(dim)
    if not fn_multi:
        return {"options": [], "multi": False}
    fn, multi = fn_multi
    try:
        opts = fn(filters or {})
    except Exception as e:
        print(f">> _discovery_options({dim}) failed: {e}")
        opts = []
    return {"options": opts, "multi": multi}


_PREMIUM_AREA_KW = ("dha", "defence", "clifton", "bath island", "khayaban-e")

def _budget_reality_message(filters: dict, lang: str = "English"):
    """If the buyer wants premium-only areas at a budget that has essentially no
    matching stock for the size, return a candid pushback + better-value areas
    (the human-agent move in conv 6). Otherwise None."""
    locs = [l.lower() for l in (filters.get("locations") or []) if isinstance(l, str)]
    if not locs or not all(any(p in l for p in _PREMIUM_AREA_KW) for l in locs):
        return None
    if not filters.get("max_price") or not (filters.get("bedrooms") or filters.get("types")):
        return None
    # Fire when the premium area has essentially nothing for this size at this budget
    # — zero, or only a token handful that would force real compromises. A budget that
    # comfortably clears the area (e.g. 6 crore in DHA) has plenty of stock and is left
    # alone; a tight one (e.g. 3 crore in DHA for a 3-bed) gets the honest heads-up.
    if count_matches(filters) >= 3:
        return None  # comfortably feasible — no pushback
    alt = {k: v for k, v in filters.items() if k != "locations"}
    areas = _areas_for_budget(alt, limit=5)
    areas_str = ", ".join(areas) if areas else "areas like PECHS, Gulshan, or Gulistan-e-Johar"
    beds = filters.get("bedrooms")
    size = f"{beds}-bed " if beds else ""
    typ = "/".join(filters.get("types") or ["property"])
    prompt = f"""You are a candid, helpful Karachi real-estate agent. The buyer wants a {size}{typ} in {', '.join(filters.get('locations'))} at {filters.get('max_price')}, but at that budget there is essentially nothing there without big compromises (old building, very small unit, no parking, poor block).
In about 2 natural sentences: honestly but warmly tell them this budget is very tight for that size in those premium areas, name the kind of compromise it would force, and ask whether they're open to better-value areas ({areas_str}) or whether those areas are a must. Sound like a real advisor giving honest guidance, not a bot. Keep it tight. Reply ONLY in {lang}. No markdown."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _widen_discovery_search(filters: dict, search_query: str, target: int = 5):
    """A real agent shortlists ~5 options and NEVER hands back an empty screen.
    Our hard filters (floor/lift/possession + exact bedrooms + a tight budget in a
    specific area) often AND down to 0-1, so on a discovery reveal we progressively
    relax — least-painful first — until there's a proper shortlist:

        1. inferred structural prefs (floor_band, has_lift)  — silent
        2. exact bedrooms → ±1 window                        — tagged "bedrooms"
        3. budget ceiling → ~25% over                        — tagged "budget"
        4. specific area (last resort, only if still empty)  — tagged "area"

    Possession and property *type* are never relaxed (a buyer who said "apartment"
    does not want a house). Budget/bedroom/area easements TAG each affected listing
    (`listing["relaxed"]`) so the cards and the reveal copy stay honest about what
    was stretched. Returns (listings, context, relaxed_field_names)."""

    def _run(f):
        return _results_to_listings(search_properties(search_query, f), f)

    orig = dict(filters)
    cur = dict(filters)
    relaxed = []
    ml, ctx = _run(cur)

    # 1) Inferred structural prefs — relax silently (the buyer never insisted).
    for fld in ("floor_band", "has_lift"):
        if len(ml) >= target:
            break
        if fld in cur:
            cur = {k: v for k, v in cur.items() if k != fld}
            relaxed.append(fld)
            ml, ctx = _run(cur)

    # 2) Exact bedrooms → ±1 (a 3-bed for a wanted-4 is the easiest substitute).
    if len(ml) < target and cur.get("bedrooms"):
        want = cur["bedrooms"]
        trial = {k: v for k, v in cur.items() if k != "bedrooms"}
        trial["bedrooms_min"] = max(1, want - 1)
        ml2, ctx2 = _run(trial)
        ml2 = [l for l in ml2 if abs(int(l["metadata"].get("bedrooms") or 0) - want) <= 1]
        ml2.sort(key=lambda l: abs(int(l["metadata"].get("bedrooms") or 0) - want))
        if len(ml2) > len(ml):
            cur, ml, ctx = trial, ml2, ctx2
            relaxed.append("bedrooms")
        for l in ml:
            if int(l["metadata"].get("bedrooms") or 0) != want:
                l.setdefault("relaxed", []).append("bedrooms")

    # 3) Budget — show slightly-over options when nothing else fits, clearly tagged.
    #    Kept modest on purpose: the buyer's AREA is the priority, so we only nudge
    #    the ceiling a little here and let step 4 (drop area) be the true last resort.
    #    Effective cap after search_properties' internal 1.1x grace ≈ 1.25x original.
    if len(ml) < target and orig.get("max_price"):
        orig_lacs = price_to_lacs(str(orig["max_price"]))
        if orig_lacs > 0:
            trial = dict(cur)
            trial["max_price"] = lacs_to_price(int(orig_lacs * 1.25 / 1.1))
            ml3, ctx3 = _run(trial)
            if len(ml3) > len(ml):
                cur, ml, ctx = trial, ml3, ctx3
                if "budget" not in relaxed:
                    relaxed.append("budget")
            for l in ml:
                if int(l["metadata"].get("price_numeric") or 0) > orig_lacs:
                    l.setdefault("relaxed", []).append("budget")

    # 4) Specific area — last resort, only if the area genuinely has nothing.
    if len(ml) == 0 and cur.get("locations"):
        trial = {k: v for k, v in cur.items() if k != "locations"}
        ml4, ctx4 = _run(trial)
        if ml4:
            cur, ml, ctx = trial, ml4, ctx4
            relaxed.append("area")
            for l in ml:
                l.setdefault("relaxed", []).append("area")

    return ml, ctx, relaxed


# Plain, polite, fixed phrasing for each slot question. We use these verbatim when
# the buyer didn't actually ask anything — the LLM kept smuggling in salesy preamble
# ("Absolutely — a 3-bed is the sweet spot…") and even the words 'lac/crore' despite
# being told not to, so for a question this simple we just don't ask the LLM.
_CANNED_DIM_QUESTION = {
    "purpose": {"English": "Is this for your own living, or more of an investment?",
                "Urdu": "Yeh apni rehaish ke liye hai ya investment ke liye?"},
    "household": {"English": "How many family members will be living there?",
                  "Urdu": "Ghar mein kitne family members rahenge?"},
    "type": {"English": "Would you prefer a house, an apartment, or a portion?",
             "Urdu": "Aap house, apartment ya portion mein se kya pasand karenge?"},
    "budget": {"English": "What budget should I keep in mind for you?",
               "Urdu": "Aap ka budget kitna rakhein main aap ke liye?"},
    "area": {"English": "Do you have an area in mind, or would you like me to suggest a few?",
             "Urdu": "Koi area zehan mein hai, ya main aap ko kuch tajweez karun?"},
    "floor": {"English": "Any floor preference — ground, middle, or top? And do you need a lift?",
              "Urdu": "Floor ki koi preference — ground, middle ya top? Aur lift chahiye?"},
    "possession": {"English": "Do you need a ready-to-move property with clear documents, or are you open to under-construction?",
                   "Urdu": "Aap ko ready-to-move property chahiye saaf documents ke sath, ya under-construction bhi chalega?"},
    "extras": {"English": "Any must-have features — like parking, a generator, or schools nearby?",
               "Urdu": "Koi zaroori feature — jaise parking, generator, ya qareeb school?"},
}

# Cues that the buyer ASKED something / raised a concern this turn — in that case we
# want the LLM to answer them as an expert, not fire back a canned slot question.
_QUESTION_CUES = (
    "?", "؟", "kya", "kia", "kaise", "kyun", "kyu", "what", "which", "why", "how",
    "can ", "could ", "should ", "is it", "are there", "do you", "expensive",
    "cheaper", "worried", "concern", "but ", "however", "afford",
)

def _buyer_asked_something(user_query: str) -> bool:
    q = " " + (user_query or "").lower().strip() + " "
    return any(cue in q for cue in _QUESTION_CUES)


# Budget and area are the "clarity" questions — keep them crisp and plain (the
# fixed canned line). Every OTHER dimension gets a brief, warm human touch via the
# LLM below (a few-word acknowledgement + the question), because the bare canned
# lines ("Any must-have features…") read too robotic on the softer questions.
_STRAIGHTFORWARD_DIMS = {"budget", "area"}


def _is_premium_only(filters: dict) -> bool:
    locs = [l for l in (filters.get("locations") or []) if isinstance(l, str)]
    return bool(locs) and all(any(p in l.lower() for p in _PREMIUM_AREA_KW) for l in locs)


def _size_phrase(filters: dict, with_family: bool = True) -> str:
    """e.g. '3-bed family apartment' / '3-bed apartment' / 'apartment'."""
    beds = filters.get("bedrooms")
    typ = (filters.get("types") or ["property"])[0]
    fam = "family " if (with_family and filters.get("purpose") == "living") else ""
    sizing = f"{beds}-bed " if beds else ""
    return f"{sizing}{fam}{typ}".strip()


def _article(phrase: str) -> str:
    """'a' / 'an' for the leading sound of a phrase (so we never say 'a apartment')."""
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _area_budget_opener(filters: dict, lang: str) -> str:
    """The branching first question — lets the buyer answer with areas OR defer to
    our suggestion based on budget. Grounded in what they've already told us."""
    desc = _size_phrase(filters, with_family=False)
    living = filters.get("purpose") == "living"
    if lang == "Urdu":
        p = " rehaish ke liye" if living else ""
        return (f"Zaroor, main madad kar sakta hoon. Kyunke yeh {desc}{p} hai, area aur "
                f"budget hi tay karte hain ke kya realistic hai. Aap ke zehan mein koi "
                f"pasandeeda area hai, ya main aap ke budget ke hisaab se munasib areas suggest karun?")
    purpose = " for family living" if living else ""
    return (f"Sure, I can help. Since it's {_article(desc)} {desc}{purpose}, the area and budget really "
            f"shape what's realistic. Do you already have a preferred area or two in Karachi, "
            f"or should I suggest suitable areas based on your budget?")


def _budget_question(filters: dict, lang: str) -> str:
    """Context-aware budget ask. A fixed premium area earns an honest 'I'll be straight
    with you' framing; an open area earns a promise to suggest areas once we know it;
    otherwise the plain line."""
    desc = _size_phrase(filters)
    if _is_premium_only(filters):
        names = "/".join(filters.get("locations"))
        if lang == "Urdu":
            return (f"{names} ke liye aap kitne budget mein comfortable hain? Agar woh ek "
                    f"proper {desc} ke liye tight hua to main saaf bata dunga.")
        return (f"For {names}, what budget are you comfortable with? I'll be straight with "
                f"you if it's tight for a proper {desc} there.")
    if not filters.get("locations"):
        if lang == "Urdu":
            return (f"Aap ka budget kitna rakhun? Pata chalte hi main aap ko ek {desc} ke "
                    f"liye realistic areas bata dunga.")
        return (f"What budget should I keep in mind? Once I know that, I can point you to "
                f"realistic areas for a {desc}.")
    canned = _CANNED_DIM_QUESTION["budget"]
    return canned.get(lang, canned["English"])


def _area_suggestion_message(filters: dict, lang: str) -> str:
    """Budget is known but no area yet → warm, positive, routine-first guidance: affirm
    the budget, name a few genuinely realistic family areas (grounded in our in-budget
    stock), steer toward routine + building quality over the area name, then ask which
    side of Karachi suits them. Never name-drops premium areas — that pushback only
    belongs when the buyer themselves fixed on DHA/Clifton (see _budget_reality_message).
    Falls back to the plain ask if we can't surface areas at all."""
    areas = _family_areas_for_budget(filters, limit=5)
    if not areas:
        canned = _CANNED_DIM_QUESTION["area"]
        return canned.get(lang, canned["English"])
    desc = _size_phrase(filters)                       # e.g. "3-bed family apartment"
    price = filters.get("max_price") or filters.get("min_price")
    living = filters.get("purpose") == "living"
    workable = len(areas) >= 3
    areas_str = ", ".join(areas[:4])
    a1 = areas[0]
    a2 = areas[1] if len(areas) > 1 else None

    if lang == "Urdu":
        lead = (f"Bohat acha — {price} ek proper {desc} ke liye Karachi mein workable budget hai."
                if workable else
                f"{price} is size ke liye thora tight hai, lekin hum phir bhi achi options dekh sakte hain.")
        routine = (" Sahi choice aap ke daily routine par depend karti hai — office, bachon ka school, "
                   "family qareeb hona, aur commute." if living else
                   " Sahi choice aap ki priorities par hai — commute, kiraye ki demand, aur resale.")
        quality = ("\n\nFamily living ke liye main area ke naam se zyada building ki condition, lift, "
                   "parking aur backup dekhne ka mashwara dunga." if living else "")
        sides = f"{a1} side ya {a2} side" if a2 else f"{a1} side"
        return (f"{lead}\n\nIs range mein achi family options {areas_str} jaise areas mein milti hain."
                f"{routine}{quality}\n\nAap ke routine ke hisaab se Karachi ka kaunsa side behtar "
                f"rahega — {sides}, ya main suggest karun?")

    lead = (f"Great — {price} is a workable budget for a proper {desc} in Karachi."
            if workable else
            f"{price} is on the tighter side for a {desc}, but we can still find solid options.")
    routine = (" The best choice really comes down to your daily routine — office, the kids' school, "
               "family nearby, and commute." if living else
               " The best choice depends on your priorities — commute, rental demand, and resale.")
    quality = ("\n\nFor family living I'd weigh building condition, lift, parking and backup more than "
               "the area name alone." if living else "")
    sides = f"the {a1} side, the {a2} side" if a2 else f"the {a1} side"
    return (f"{lead}\n\nIn this range we can look at good family options in areas like {areas_str}."
            f"{routine}{quality}\n\nWhich side of Karachi suits your routine better — {sides}, or "
            f"would you like me to suggest?")


def _area_blocks(area_label: str, limit: int = 8) -> list:
    """Real sub-localities (Block 7, Phase 5, Precinct 10…) of a parent area that
    actually exist in our inventory — so the area-intel discussion only ever names
    blocks we can fulfil, never an invented one."""
    al = area_label.split(",")[0].strip().lower()
    if not al:
        return []
    try:
        metas = vectorstore._collection.get(include=["metadatas"]).get("metadatas", []) or []
    except Exception:
        return []
    from collections import Counter
    blocks = Counter()
    for m in metas:
        loc = m.get("location") or ""
        if al in loc.lower():
            mt = re.search(r'\b((?:block|phase|sector|precinct)\s+[\w-]+)', loc, re.IGNORECASE)
            if mt:
                blocks[mt.group(1).title()] += 1
    return [b for b, _ in blocks.most_common(limit)]


def _area_intel_fallback(area_disp: str, desc: str, budget: str, blocks: list,
                         living: bool, exact_count: int = 3) -> str:
    """Deterministic stand-in for the LLM area discussion (used if the call fails) —
    keeps the same shape: honest budget-fit affirmation, block-level read, two questions."""
    fam = "family " if living else ""
    if exact_count >= 3:
        opener = f"Great — {area_disp} makes good sense for a proper {desc} at around {budget}."
    elif exact_count >= 1:
        opener = (f"{area_disp} is a good {fam}area, though proper {desc}s right at {budget} are "
                  f"limited there — we'll look carefully and may stretch a little.")
    else:
        opener = (f"{area_disp} is a lovely {fam}area, but honestly a proper {desc} at {budget} is "
                  f"hard to find there — we may need to stretch the budget or look at nearby value.")
    pockets = ""
    if blocks:
        first = ", ".join(blocks[:4])
        pockets = (f" Within {area_disp} the block matters a lot — pockets like {first} tend to be "
                   f"more residential and balanced for {fam}living, while blocks nearer the main "
                   f"chowrangis give better connectivity but can feel busier.")
    return (f"{opener}{pockets}"
            f"\n\nShould I prioritise a quieter {fam}environment, or better access to main roads, "
            f"markets and schools? And should I treat lift, parking, backup, water and a proper "
            f"{fam}building as must-haves?")


def _area_intel_message(state: dict, history_text: str, lang: str) -> str:
    """Fact-constrained LLM discussion fired once the buyer commits to a single area:
    the engine supplies the real blocks + budget + size; the LLM supplies the local
    block-level read, adaptive detail, and human tone, then asks the quiet-vs-connected
    and must-haves questions. Falls back to a deterministic version on any LLM error."""
    f = state["filters"]
    locs = [l for l in (f.get("locations") or []) if isinstance(l, str)]
    if not locs:
        return ""
    area_disp = locs[0].split(",")[0].strip()
    desc = _size_phrase(f)
    budget = f.get("max_price") or f.get("min_price")
    living = f.get("purpose") == "living"
    blocks = _area_blocks(area_disp)
    blocks_str = ", ".join(blocks)
    fam = "family " if living else ""
    if blocks_str:
        locality_note = (f"Real sub-localities of {area_disp} that exist in our inventory (ONLY "
                         f"reference these, NEVER invent a block/phase name): {blocks_str}.")
    else:
        locality_note = (f"{area_disp} isn't split into distinct blocks in our inventory, so speak "
                         f"about location WITHIN it generally — main-road/commercial frontage vs "
                         f"quieter interior lanes — without inventing specific block names.")
    # Budget-fit reality for THIS exact size in THIS area — so a buyer who self-types a
    # premium-priced area we don't actually stock at their budget gets an honest heads-up
    # NOW, instead of a confident affirmation that the reveal then flatly contradicts.
    exact_count = count_matches(f)
    if exact_count >= 3:
        fit_directive = (f"1. Warmly affirm in ONE line that {area_disp} is a sensible, workable choice "
                         f"for a proper {desc} in this budget (we do have options there).")
    elif exact_count >= 1:
        fit_directive = (f"1. Affirm {area_disp} as a good {fam}area, but be HONEST in that same line "
                         f"that proper {desc}s right at {budget} are limited there, so we'll look carefully "
                         f"and may consider stretching the budget a little or a nearby value pocket.")
    else:
        fit_directive = (f"1. IMPORTANT — at {budget} our inventory has essentially NO proper {desc} in "
                         f"{area_disp} (they typically cost more there). Open warmly but HONESTLY: {area_disp} "
                         f"is a lovely {fam}area, yet a proper {desc} at this budget is genuinely hard to find "
                         f"there. Gently flag that we'd likely need to stretch the budget, consider a slightly "
                         f"smaller size, or look at nearby value areas — do NOT pretend options exist.")
    prompt = f"""You are a sharp, warm, genuinely knowledgeable Karachi real-estate advisor mid-consultation, before showing any listings. Talk like an experienced human agent who knows the city block by block — NEVER a form or a bot.

The buyer wants a {desc} at around {budget} and has chosen to focus on {area_disp}.
{locality_note}

Write a natural, discussion-style reply that:
{fit_directive}
2. Shows real local expertise: explain that within {area_disp} the specific block/pocket matters a lot, and give a SHORT, honest read — which pockets tend to be quieter and more residential for {fam}living, versus which are more central/connected but can feel busier or more crowded. Use real Karachi knowledge but ONLY name blocks/areas from the note above (or speak generally if none were given). Never overstate.
3. Ask which they'd prioritise — a quieter {fam}environment, or better access to main roads, markets and schools.
4. Then ask whether to treat lift, parking, generator/backup, water, and a proper {fam}building as must-haves.

Judge the right DEGREE OF DETAIL and warmth from how the buyer is engaging in the conversation below — match their depth, never over-talk. Keep it tight: at most two short paragraphs, then the two closing questions. No markdown, no headings, no bullet lists. Reply ONLY in {lang}.

Conversation so far:
{history_text}
Return ONLY your reply."""
    try:
        return llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        print(f">> _area_intel_message LLM failed: {e}")
        return _area_intel_fallback(area_disp, desc, budget, blocks, living, exact_count)


def _discovery_question(dimension: str, filters: dict, history_text: str, count: int,
                        lang: str = "English", user_query: str = "", ack: str = "") -> str:
    asked_something = _buyer_asked_something(user_query)
    budget_known = bool(filters.get("max_price") or filters.get("min_price"))

    # NOTE: template questions are returned VERBATIM — no LLM-written preamble. An earlier
    # "acknowledge what the buyer shared" prepend fired inconsistently (temp 0.7) and stuck
    # filler like "Thanks — I'll keep that in mind" on plain messages. Cut on purpose:
    # predictable copy beats a clever line that misfires. `ack` is kept in the signature
    # but unused; the situational *hints* from the same call still feed ranking silently.
    def _tmpl(body: str) -> str:
        return body

    # Combined smart opener — area-or-budget as one branching question.
    if dimension == "area_budget":
        return _tmpl(_area_budget_opener(filters, lang))

    # Budget — context-aware framing (premium area → honest; open area → promise
    # suggestions), unless the buyer asked a question (then answer it via the LLM).
    if dimension == "budget" and not asked_something:
        return _tmpl(_budget_question(filters, lang))

    # Area, once budget is known — honest, grounded area guidance instead of a bare ask.
    if dimension == "area" and budget_known and not asked_something:
        return _tmpl(_area_suggestion_message(filters, lang))

    # Other clarity dims (or budget/area when nothing context-specific applies) → plain line.
    if (dimension in _STRAIGHTFORWARD_DIMS
            and not asked_something
            and dimension in _CANNED_DIM_QUESTION):
        canned = _CANNED_DIM_QUESTION[dimension]
        return _tmpl(canned.get(lang, canned["English"]))

    known = []
    if filters.get("max_price") and filters.get("min_price"):
        known.append(f"budget {filters['min_price']}–{filters['max_price']}")
    elif filters.get("max_price"):
        known.append(f"budget ~{filters['max_price']}")
    elif filters.get("min_price"):
        known.append(f"budget above {filters['min_price']}")
    if filters.get("locations"):
        known.append("areas: " + ", ".join(filters["locations"]))
    if filters.get("types"):
        known.append("type: " + ", ".join(filters["types"]))
    if filters.get("bedrooms"):
        known.append(f"{filters['bedrooms']} bed")
    known_str = "; ".join(known) if known else "nothing specific yet"
    extra = ""
    if dimension == "area":
        # Area suggestions ride on the tappable pills, NOT the message — keep the copy
        # clean and don't name-drop areas (or DHA/Clifton) in the sentence itself.
        extra = ("\nThe buyer will see tappable area suggestions below your message, so do NOT list "
                 "or name specific areas in your sentence — just ask, in one short line, whether they "
                 "have an area in mind or want your suggestions.")
    prompt = f"""You are a sharp, warm Karachi real-estate advisor mid-consultation (before showing listings) — talk like an experienced human agent, NEVER a form or slot-filler.
What you already know about this buyer: {known_str}.
The customer just said: "{user_query}"

Respond the way a real agent would:
1. If they ASKED a question or raised a concern, ANSWER it directly and helpfully FIRST — you're the expert (e.g. actually explain the compromises, give honest guidance). Never dodge their question by asking another one.
2. Otherwise, you MAY open with a SHORT, warm acknowledgement of what they just said — a few words at most ("Got it." / "Perfect.") — then ask the next thing. Keep it genuinely human but brief: NO salesy clichés ("Absolutely —", "Great choice!"), NO justifying or selling their choice back to them ("you'll get the right balance of space/resale/demand"), NO explanation paragraph.
3. Move forward by learning this next: {_DISCOVERY_DIM_PROMPT[dimension]}{extra}

Hard rules: NEVER ask a question you've already asked in the conversation below — read it first; if you already asked something, do not ask it again. Do NOT interrogate — at most ONE new question. When you're simply asking the next question (they didn't ask anything), keep it to at most TWO short sentences — an optional few-word acknowledgement, then the question — never a paragraph, no filler, no lists/bullets/markdown. Reply ONLY in {lang}.
Conversation so far:
{history_text}
Return ONLY your reply."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _discovery_ask_response(disc: dict) -> dict:
    """Build the API response for a discovery question turn (no results yet)."""
    opts = _discovery_options(disc.get("dim"), disc.get("filters") or {})
    # When the agent gives advice then asks (e.g. budget answer → area suggestions +
    # "which are you leaning toward?"), put that trailing question on its own line —
    # same treatment the results turn gets. No-op when the text is a bare question.
    question = _split_closing_question(disc["question"])
    return {
        "response": question,
        "stage": "discovery",
        "discovery_complete": False,
        "match_count": disc["count"],
        "accumulated_filters": disc["filters"],
        "next_question": question,
        "listings": [],
        "filters": disc["filters"],
        "follow_up": None,
        "actions": [],
        "options": opts["options"],
        "options_multi": opts["multi"],
        "dimension": disc.get("dim"),
        "meta": {"no_results": False, "action": "discovery"},
    }


def _discovery_step(user_id: str, user_query: str, history_text: str, start_ok: bool = True):
    """Drive one turn of progressive discovery. Returns None when discovery does
    not apply, else a dict with mode 'ask' or 'reveal'. With start_ok=False it
    only CONTINUES an already-active session (used early, before intent checks,
    so discovery answers aren't hijacked); with start_ok=True it may START one."""
    state = user_discovery_states.get(user_id)
    search_history = get_user_search_history(user_id)
    if state is None:
        if search_history or not start_ok:
            return None  # results already shown, or not allowed to start here
        state = {"filters": {}, "asked": {}, "turns": 0, "active": True}
        user_discovery_states[user_id] = state
    elif not state["active"]:
        return None

    lang = _detect_language(user_query + " " + history_text)
    dropped = _merge_discovery_filters(state, user_query, lang)
    count = count_matches(state["filters"])
    # "show me ..." is an escape hatch out of discovery — but only honour it once we
    # have something concrete to shortlist on (budget, area, or an explicit bedroom
    # count). Otherwise a first-turn "show me some flats" — which sets only a property
    # TYPE — would skip qualification entirely and dump random listings. A bare type
    # (or a soft, inferred bedrooms_floor) is not enough; keep discovering and ask the
    # first qualifying question instead.
    _qual_keys = ("max_price", "min_price", "locations", "bedrooms", "bedrooms_min")
    _has_qualifier = any(state["filters"].get(k) for k in _qual_keys)
    wants_results = (any(kw in user_query.lower() for kw in _SHOW_RESULTS_KW)
                     and _has_qualifier)

    # Budget reality check (once): premium-only areas at a budget with no stock for
    # the size — be honest and redirect, like a real agent, instead of forcing it.
    if not state.get("pushed_back"):
        reality = _budget_reality_message(state["filters"], lang)
        if reality:
            state["pushed_back"] = True
            state["turns"] += 1
            # dim="area" so better-value areas ride along as tappable pills — the buyer
            # can pivot to one in a single tap instead of retyping.
            return {"mode": "ask", "question": reality, "count": count, "dim": "area",
                    "filters": dict(state["filters"]), "dropped": dropped}

    # Area discussion (once): the buyer has committed to a SINGLE area and we know the
    # budget — drop down into a knowledgeable, block-level conversation about it instead
    # of firing the bare floor question. This is the fact-constrained-LLM turn: the engine
    # hands it the real blocks, the LLM supplies local knowledge + adaptive tone. It also
    # asks the must-haves, so we mark floor+extras resolved and head to the reveal next.
    _locs = [l for l in (state["filters"].get("locations") or []) if isinstance(l, str)]
    if (not state.get("area_discussed") and not wants_results
            and (state["filters"].get("max_price") or state["filters"].get("min_price"))
            and len(_locs) == 1):
        intel = _area_intel_message(state, history_text, lang)
        if intel:
            state["area_discussed"] = True
            state.setdefault("resolved", set()).update({"floor", "extras"})
            state["last_dim"] = "area_intel"
            state["turns"] += 1
            return {"mode": "ask", "question": intel, "count": count, "dim": "area_intel",
                    "filters": dict(state["filters"]), "dropped": dropped}

    dim = None if (wants_results or _discovery_ready(state)) else _next_discovery_dim(state)
    if dim is None:
        state["active"] = False
        return {
            "mode": "reveal",
            "filters": dict(state["filters"]),
            "search_query": _filters_to_query(state["filters"], fallback=user_query),
            "count": count,
            "dropped": dropped,
        }

    state["asked"][dim] = state["asked"].get(dim, 0) + 1
    state["last_dim"] = dim  # so a "you suggest" reply next turn resolves THIS dim
    state["turns"] += 1
    question = _discovery_question(dim, state["filters"], history_text, count, lang, user_query,
                                   ack=state.get("_last_ack") or "")
    return {
        "mode": "ask",
        "question": question,
        "count": count,
        "dim": dim,
        "filters": dict(state["filters"]),
        "dropped": dropped,
    }


def _split_closing_question(text: str) -> str:
    """Put a trailing follow-up question on its own line.

    The advisory copy ends with one contextual question; the LLM often keeps it
    glued to the explanation paragraph. The web UI renders \\n as <br>, so we
    insert a blank line before the final question. No-op when the text doesn't
    end in a question, so it's safe to call on any channel/branch.
    """
    if not text:
        return text
    s = text.rstrip()
    if not (s.endswith("?") or s.endswith("؟")):  # latin or arabic '?'
        return text
    body = s[:-1]
    # last sentence terminator (latin + urdu/arabic) followed by whitespace
    matches = list(re.finditer(r"[.!?؟۔]\s+", body))
    if not matches:
        return text
    split_at = matches[-1].end()
    head = s[:split_at].rstrip()
    question = s[split_at:].strip()
    if not head or not question:
        return text
    return head + "\n\n" + question


# ──────────────────────────────────────────────────────────────────────────
# Seller flow (valuation / handoff)
# ──────────────────────────────────────────────────────────────────────────
# A seller is NOT a buyer. "I want to sell my flat" must switch out of buyer
# discovery into a short, one-question-at-a-time valuation intake that ends in an
# agent handoff — never a buyer property search. State is per-user so follow-up
# answers ("DHA Phase 5", "3 bed") stay in the seller flow instead of leaking back
# into a buyer area/bedroom search.
user_seller_states = {}  # user_id -> {"data": dict, "pending": str|None}

# Deterministic seller-intent detection. Word-boundary guarded so buyer phrasing
# ("should I sell or rent first", "best seller area", "resell value") does NOT
# match — only a clear intent to sell THEIR property.
_SELLER_INTENT_RE = re.compile(
    r"\b(?:i\s+(?:want|need|wish|would\s+like|am\s+looking|am\s+planning|am\s+trying)\s+to\s+sell"
    r"|want\s+to\s+sell|wanna\s+sell|looking\s+to\s+sell|planning\s+to\s+sell|trying\s+to\s+sell"
    r"|i['’]?m\s+selling|sell\s+(?:my|our|this)\b)"
    r"|\blist\s+my\s+(?:flat|house|home|property|apartment|portion|plot|shop|bungalow)\b"
    r"|\b(?:bechna|bechni|bech\s*do|becham|becho)\b"
    r"|\bmera\s+(?:ghar|flat|makan|makaan|plot|portion)\s+(?:bech|sale)"
    r"|\bvaluation\s+(?:of|for)\s+my\b",
    re.IGNORECASE,
)


def detect_seller_intent(query: str) -> bool:
    return bool(_SELLER_INTENT_RE.search(query or ""))


# One question at a time (CLAUDE.md hard rule). Order mirrors a real listing intake.
_SELLER_STEPS = ["location", "size", "bedrooms", "condition", "price", "contact"]

_SELLER_Q = {
    "English": {
        "location": "Sure — I can help you get it listed and valued. Which area or block is your property in?",
        "size": "Got it. How big is it — covered area or yards?",
        "bedrooms": "And how many bedrooms does it have?",
        "condition": "What condition is it in — newly built, well-maintained, or does it need some work?",
        "price": "What price do you have in mind?",
        "contact": "Perfect. What's the best number for our listing agent to reach you on?",
    },
    "Urdu": {
        "location": "Zaroor — main aap ki property list aur value karwa sakta hoon. Property kis area ya block mein hai?",
        "size": "Theek hai. Property kitni bari hai — covered area ya gaz mein?",
        "bedrooms": "Is mein kitne bedrooms hain?",
        "condition": "Condition kaisi hai — nayi bani, achi maintain, ya thori kaam ki zaroorat hai?",
        "price": "Aap ke zehan mein kya price hai?",
        "contact": "Bilkul. Hamare listing agent ke liye behtareen contact number kya hai?",
    },
}

_SELLER_LEAD = {
    "English": "Sure — I can help you get it listed and valued. ",
    "Urdu": "Zaroor — main aap ki property list aur value karwa sakta hoon. ",
}


def _seller_summary(data: dict, lang: str) -> str:
    loc = data.get("location", "—"); size = data.get("size", "—")
    beds = data.get("bedrooms", "—"); cond = data.get("condition", "—")
    price = data.get("price", "—"); contact = data.get("contact", "—")
    if lang == "Urdu":
        return (f"Shukriya! Main ne aap ki property note kar li hai — {loc}, {beds} bed, "
                f"{size}, condition: {cond}, expected price {price}. Hamara listing agent "
                f"jald hi aap ko {contact} par call kar ke valuation aur agle steps arrange karega. 🙏")
    return (f"Thanks! I've logged your property — {loc}, {beds} bed, {size}, condition: {cond}, "
            f"expected price {price}. Our listing agent will call you on {contact} shortly to "
            f"arrange a valuation and next steps. 🙏")


def _seller_resp(text: str, action: str) -> dict:
    return {"response": text, "listings": [], "filters": {}, "follow_up": None,
            "actions": [], "meta": {"no_results": False, "action": action}}


def _seller_step(user_id: str, user_query: str, lang: str, is_start: bool):
    """Drive one turn of the seller valuation intake. On start, capture an area if
    the trigger message named one. Otherwise record the answer to the question we
    last asked, then ask the next unfilled field — or hand off when complete."""
    if is_start:
        state = {"data": {}, "pending": None}
        areas = _match_karachi_areas(user_query)
        if areas:
            state["data"]["location"] = ", ".join(areas)
        user_seller_states[user_id] = state
    else:
        state = user_seller_states.get(user_id)
        if not state:
            return None
        if state.get("pending"):
            state["data"][state["pending"]] = user_query.strip()
            state["pending"] = None

    data = state["data"]
    nxt = next((s for s in _SELLER_STEPS if s not in data), None)
    if nxt is None:  # all fields collected → hand off and end the session
        user_seller_states.pop(user_id, None)
        return _seller_resp(_seller_summary(data, lang), "seller_handoff")

    state["pending"] = nxt
    q = _SELLER_Q[lang][nxt]
    # If we skipped the location question (captured from the trigger), keep the warm
    # lead-in so the first reply still acknowledges we understood the seller intent.
    if is_start and nxt != "location":
        q = _SELLER_LEAD[lang] + q
    return _seller_resp(q, "seller")


def get_response(user_query: str, user_id: str = "web", channel: str = "web") -> dict:
    memory = get_user_memory(user_id)
    search_history = get_user_search_history(user_id)

    # Conversation history
    history = memory.chat_memory.messages
    history_text = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # ── Seller flow (CONTINUE an active valuation session) ──
    # Runs first so a seller's answers ("DHA Phase 5", "3 bed") stay in the seller
    # intake and are never hijacked as a buyer area/bedroom search.
    _seller_lang = _detect_language(user_query + " " + history_text)
    if user_id in user_seller_states:
        result = _seller_step(user_id, user_query, _seller_lang, is_start=False)
        if result is not None:
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(result["response"])
            return result

    # ── Seller flow (START on detecting seller intent) ──
    # Switches out of buyer discovery into a valuation/handoff flow. Checked before
    # the discovery-continue block so a buyer can pivot ("actually I want to sell"),
    # and any in-progress buyer discovery is abandoned.
    if detect_seller_intent(user_query):
        user_discovery_states.pop(user_id, None)
        result = _seller_step(user_id, user_query, _seller_lang, is_start=True)
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(result["response"])
        return result

    discovery_filters = None
    discovery_search_query = None
    discovery_dropped = []

    # ── Progressive discovery (CONTINUE an active session) ──
    # Runs before the intent/small-talk checks so answers to our own discovery
    # questions ("it's for my parents") are never hijacked as small talk.
    _disc = _discovery_step(user_id, user_query, history_text, start_ok=False)
    if _disc is not None:
        if _disc["mode"] == "ask":
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(_disc["question"])
            return _discovery_ask_response(_disc)
        discovery_filters = _disc["filters"]
        discovery_search_query = _disc["search_query"]
        discovery_dropped = _disc["dropped"]

    # ── Mortgage slot filling: continue active mortgage conversation ──
    slot_result = _mortgage_handler.handle_slot_filling(user_id, user_query)
    if slot_result:
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(slot_result["response"])
        return slot_result

    # ── Pending clarification: merge stored query with user's answer ──
    already_clarified = False
    if user_id in user_pending_queries:
        pending = user_pending_queries.pop(user_id)
        original = pending["original_query"]
        user_query = f"{original}. {user_query}"
        already_clarified = True
        print(f">> Merged pending query: {user_query!r}")

    # ── Investment intent pre-detection (keyword, no LLM) ──
    # Must run BEFORE mortgage regex so "invest 1 crore" never hits REVERSE_MORTGAGE.
    if discovery_filters is None and detect_investment_intent(user_query):
        result = _handle_investment_intent(user_query, user_id, search_history, channel)
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(result["response"])
        return result

    # ── Mortgage intent pre-detection (regex, no LLM) ──
    mortgage_intent = detect_mortgage_intent(user_query)
    if discovery_filters is None and mortgage_intent:
        result = _handle_mortgage_intent(mortgage_intent, user_id, user_query, search_history)
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(result["response"])
        return result

    # Quick small talk check before full classification (skipped on a discovery
    # reveal — the gathered filters already tell us this is a property search).
    if discovery_filters is None:
        # Deterministic property signal wins outright — the small fast model is
        # unreliable and was tagging clear property/follow-up turns as SMALLTALK.
        if _has_property_signal(user_query):
            is_smalltalk = False
            print(">> Property signal detected — bypassing small-talk gate")
        else:
            quick_check = llm_fast.invoke([HumanMessage(content=f"""Is this a greeting or small talk unrelated to property search?
    Message: "{user_query}"
    Reply with only: SMALLTALK or PROPERTY""")])
            print(f">> Quick check: {quick_check.content.strip()!r}")
            is_smalltalk = quick_check.content.strip().upper() == "SMALLTALK"
        if is_smalltalk:
            response = llm.invoke([
                SystemMessage(content="You are a friendly real estate assistant. Respond warmly to greetings, then ask what property they're looking for. Max 2 sentences. Match the user's language."),
                HumanMessage(content=user_query)
            ])
            ai_response = response.content
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(ai_response)
            return {"response": ai_response, "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False, "action": None}}

        # ── Progressive discovery (START a session on the first property query) ──
        # Reached only for a genuine property message (small talk returned above).
        _disc = _discovery_step(user_id, user_query, history_text, start_ok=True)
        if _disc is not None:
            if _disc["mode"] == "ask":
                memory.chat_memory.add_user_message(user_query)
                memory.chat_memory.add_ai_message(_disc["question"])
                return _discovery_ask_response(_disc)
            discovery_filters = _disc["filters"]
            discovery_search_query = _disc["search_query"]
            discovery_dropped = _disc["dropped"]

    stripped = user_query.strip()
    matched_listings = None
    context = ""
    filters = {}
    action_label = None
    highlight_id = None
    classification_type = "NEWSEARCH"
    requested_am, satisfied_am, missing_am = [], [], []

    # Handle numeric shortcuts (4=cheaper, 5=larger, 6=contact)
    if discovery_filters is None and stripped in ("4", "5", "6") and search_history:
        last = search_history[-1]
        prev_filters = last.get("filters", {})
        prev_listings = last["listings"]
        locations = prev_filters.get("locations", [])

        if stripped == "4":
            action_label = "cheaper"
            classification_type = "CHEAPER"
            shown = prev_listings[:10]
            prices = [int(l["metadata"].get("price_numeric") or 0) for l in shown if l["metadata"].get("price_numeric")]
            if prices:
                max_price_lacs = int(min(prices)) - 1
                filters = {k: v for k, v in prev_filters.items()}
                filters["max_price"] = lacs_to_price(max_price_lacs)
                results = fetch_cheaper(locations, max_price_lacs, prev_filters, k=10)
                matched_listings, context = _results_to_listings(results, filters)
                search_history.append({"topic": f"cheaper · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
                user_query = "show cheaper property options"
            else:
                matched_listings = []

        elif stripped == "5":
            action_label = "larger"
            classification_type = "LARGER"
            filters = {k: v for k, v in prev_filters.items()}
            shown_beds = [l["metadata"].get("bedrooms", 0) for l in prev_listings[:10] if l["metadata"].get("bedrooms")]
            base_beds = prev_filters.get("bedrooms") or (max(shown_beds) if shown_beds else 3)
            new_beds = base_beds + 1
            filters["bedrooms"] = new_beds
            loc_str = " ".join(locations) if locations else "properties"
            results = search_properties(f"{new_beds} bed {loc_str}", filters, k=10)
            matched_listings, context = _results_to_listings(results, filters)
            search_history.append({"topic": f"larger · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
            user_query = f"show {new_beds} bedroom properties"

        else:  # "6"
            matched_listings = prev_listings
            context = last["context"]
            filters = prev_filters
            user_query = "Share the full agent name, phone number and contact details for each property shown"
            classification_type = "FOLLOWUP"

    if matched_listings is None:
        # Discovery reveal: filters already accumulated — skip classification and
        # go straight to the NEWSEARCH path with the gathered filters.
        if discovery_filters is not None:
            classification = {"type": "NEWSEARCH", "index": None}
            classification_type = "NEWSEARCH"
        # Digit follow-up shortcut
        elif stripped.isdigit() and int(stripped) <= 9 and search_history:
            classification = {"type": "FOLLOWUP", "index": len(search_history) - 1}
            classification_type = "FOLLOWUP"
            print(f">> Classification: digit shortcut → FOLLOWUP")
        else:
            classification = classify_query(user_query, search_history)
            classification_type = classification["type"]

        # SMALLTALK
        if classification_type == "SMALLTALK":
            response = llm.invoke([
                SystemMessage(content="You are a friendly real estate assistant. Respond warmly and briefly to greetings or small talk, then naturally ask what property they are looking for. Max 2 sentences. Respond in the user's language (Urdu or English)."),
                HumanMessage(content=user_query)
            ])
            ai_response = response.content
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(ai_response)
            return {"response": ai_response, "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False, "action": None}}

        # MORTGAGE intents (LLM fallback path — regex pre-check above handles most cases)
        if classification_type in ("MORTGAGE_EXPLICIT", "AFFORDABILITY_HINT", "REVERSE_MORTGAGE"):
            result = _handle_mortgage_intent(classification_type, user_id, user_query, search_history)
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(result["response"])
            return result

        # INVESTMENT (LLM fallback — keyword pre-check above handles most cases)
        if classification_type == "INVESTMENT":
            result = _handle_investment_intent(user_query, user_id, search_history, channel)
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(result["response"])
            return result

        # LARGER
        if classification_type == "LARGER" and search_history:
            action_label = "larger"
            idx = min(classification.get("index") or len(search_history) - 1, len(search_history) - 1)
            last = search_history[idx]
            prev_filters = last.get("filters", {})
            prev_listings = last["listings"]
            locations = prev_filters.get("locations", [])
            shown_beds = [l["metadata"].get("bedrooms", 0) for l in prev_listings[:10] if l["metadata"].get("bedrooms")]
            base_beds = prev_filters.get("bedrooms") or (max(shown_beds) if shown_beds else 3)
            new_beds = base_beds + 1
            filters = {k: v for k, v in prev_filters.items()}
            filters["bedrooms"] = new_beds
            loc_str = " ".join(locations) if locations else "properties"
            results = search_properties(f"{new_beds} bed {loc_str}", filters, k=10)
            matched_listings, context = _results_to_listings(results, filters)
            search_history.append({"topic": f"larger · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
            user_query = f"show {new_beds} bedroom properties"

        # CHEAPER
        elif classification_type == "CHEAPER" and search_history:
            action_label = "cheaper"
            idx = min(classification.get("index") or len(search_history) - 1, len(search_history) - 1)
            last = search_history[idx]
            prev_filters = last.get("filters", {})
            prev_listings = last["listings"]
            locations = prev_filters.get("locations", [])
            shown = prev_listings[:10]
            prices = [int(l["metadata"].get("price_numeric") or 0) for l in shown if l["metadata"].get("price_numeric")]
            if prices:
                max_price_lacs = int(min(prices)) - 1
                filters = {k: v for k, v in prev_filters.items()}
                filters["max_price"] = lacs_to_price(max_price_lacs)
                results = fetch_cheaper(locations, max_price_lacs, prev_filters, k=10)
                matched_listings, context = _results_to_listings(results, filters)
                search_history.append({"topic": f"cheaper · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
                user_query = "show cheaper property options"
            else:
                matched_listings = []

        # FOLLOWUP
        elif classification_type == "FOLLOWUP" and classification.get("index") is not None:
            idx = classification["index"]
            if idx >= len(search_history):
                idx = len(search_history) - 1
            matched_listings = search_history[idx]["listings"]
            context = search_history[idx]["context"]
            filters = {}
            if stripped.isdigit():
                selected = int(stripped) - 1
                if 0 <= selected < len(matched_listings):
                    matched_listings = [matched_listings[selected]]
                    context = matched_listings[0]["metadata"].get("description", context)
                    highlight_id = matched_listings[0]["metadata"].get("id")

        # IMAGES
        elif classification_type == "IMAGES" and search_history:
            action_label = "images"
            idx = min((classification.get("index") or len(search_history) - 1), len(search_history) - 1)
            # Fall back to most recent search that actually had results
            while idx > 0 and not search_history[idx]["listings"]:
                idx -= 1
            prev_listings = search_history[idx]["listings"]

            # Resolve which property the user wants
            prop_num = classification.get("property_num")
            if not prop_num:
                num_match = re.search(r'\b([1-9])\b', user_query)
                prop_num = int(num_match.group(1)) if num_match else 1
            prop_idx = max(0, min(int(prop_num) - 1, len(prev_listings) - 1))
            target = prev_listings[prop_idx] if prev_listings else None

            if target:
                images_raw = target["metadata"].get("images", "[]")
                images_list = json.loads(images_raw) if isinstance(images_raw, str) else images_raw
                prop_title = target["metadata"].get("title", "Property").title()
                ai_response = f"Here are all the photos for *{prop_title}* 📸"
                memory.chat_memory.add_user_message(user_query)
                memory.chat_memory.add_ai_message(ai_response)
                return {
                    "response": ai_response,
                    "listings": [target],
                    "filters": {},
                    "follow_up": None,
                    "actions": [],
                    "meta": {
                        "no_results": False,
                        "action": "images",
                        "images_to_send": images_list,
                        "images_title": prop_title,
                    }
                }
            else:
                matched_listings = []

        # REFINE — adjust the current search, inheriting every dimension the user
        # didn't change (budget + area especially). This is what stops a follow-up
        # like "2 bed would be better, within the budget" from dropping the budget.
        elif classification_type == "REFINE" and search_history:
            idx = min(classification.get("index") or len(search_history) - 1, len(search_history) - 1)
            last = search_history[idx]
            prev_filters = last.get("filters", {})
            new_filters = extract_filters(user_query)
            # New explicit values win; everything else (budget, area, type) carries over.
            filters = {**prev_filters, **new_filters}
            print(f">> REFINE merge: prev={prev_filters}, new={new_filters}, merged={filters}")
            search_query = _filters_to_query(filters, fallback=user_query)
            results = search_properties(search_query, filters)
            matched_listings, context = _results_to_listings(results, filters)
            topic_parts = []
            if filters.get("locations"):
                topic_parts.append(" / ".join(filters["locations"]))
            if filters.get("types"):
                topic_parts.append("/".join(filters["types"]))
            if filters.get("bedrooms"):
                topic_parts.append(f"{filters['bedrooms']} bed")
            if filters.get("max_price"):
                topic_parts.append(f"under {filters['max_price']}")
            topic = ", ".join(topic_parts) if topic_parts else user_query[:60]
            search_history.append({"topic": topic, "listings": matched_listings, "context": context, "filters": filters})
            requested_am, satisfied_am, missing_am = _amenity_gap_summary(matched_listings, filters)
            # Treat as a fresh result set downstream: overview copy + grid repaint.
            classification_type = "NEWSEARCH"

        # NEWSEARCH
        if matched_listings is None:
            if discovery_filters is not None:
                # Discovery already gathered + validated the filters; the
                # clarification gate is redundant here.
                filters = discovery_filters
                search_query = discovery_search_query or user_query
                print(f">> Discovery reveal with filters: {filters}")
            else:
                filters = extract_filters(user_query)
                search_query = user_query
                print(f">> Extracted filters: {filters}")

                # Budget-continuity safety net: if the classifier mislabelled a refinement
                # as NEWSEARCH, the user references their existing budget ("within the
                # budget") but states no figure, inherit it from the last search so the
                # ceiling isn't silently dropped.
                if not filters.get("max_price") and search_history:
                    q_low = user_query.lower()
                    if any(kw in q_low for kw in _BUDGET_CONTINUITY_KW):
                        prev_budget = search_history[-1].get("filters", {}).get("max_price")
                        if prev_budget:
                            filters["max_price"] = prev_budget
                            if not filters.get("locations"):
                                prev_locs = search_history[-1].get("filters", {}).get("locations")
                                if prev_locs:
                                    filters["locations"] = prev_locs
                            print(f">> Inherited budget from previous search: {prev_budget}")

                # ── Clarification gate: ask once only — skip if already asked ──
                if not already_clarified and _needs_clarification(filters):
                    clarifying_q = _generate_clarifying_question(user_query, filters, history_text)
                    user_pending_queries[user_id] = {"original_query": user_query, "filters": filters}
                    print(f">> Asking clarification for user {user_id!r}: {clarifying_q!r}")
                    memory.chat_memory.add_user_message(user_query)
                    memory.chat_memory.add_ai_message(clarifying_q)
                    return {
                        "response": clarifying_q,
                        "listings": [],
                        "filters": filters,
                        "follow_up": None,
                        "actions": [],
                        "meta": {"no_results": False, "action": "clarifying"},
                    }

            results = search_properties(search_query, filters)
            matched_listings, context = _results_to_listings(results, filters)

            # Discovery reveal: if hard filters narrowed to too few, widen to a
            # proper shortlist (a real agent shows 4-5, not 1) and note what eased.
            if discovery_filters is not None and len(matched_listings) < 3:
                matched_listings, context, widened = _widen_discovery_search(filters, search_query)
                if widened:
                    discovery_dropped = list(discovery_dropped) + widened

            topic_parts = []
            if filters.get("locations"):
                topic_parts.append(" / ".join(filters["locations"]))
            if filters.get("types"):
                topic_parts.append("/".join(filters["types"]))
            if filters.get("bedrooms"):
                topic_parts.append(f"{filters['bedrooms']} bed")
            if filters.get("max_price"):
                topic_parts.append(f"under {filters['max_price']}")
            topic = ", ".join(topic_parts) if topic_parts else user_query[:60]
            search_history.append({"topic": topic, "listings": matched_listings, "context": context, "filters": filters})
            print(f">> New search saved as: '{topic}'")
            requested_am, satisfied_am, missing_am = _amenity_gap_summary(matched_listings, filters)

    no_results = len(matched_listings) == 0

    # New result sets get an overview of the whole grid; FOLLOWUP / single-pick
    # responses still describe the one property the user landed on.
    is_overview = classification_type in ("NEWSEARCH", "CHEAPER", "LARGER")
    overview = "" if no_results else _results_overview(matched_listings)
    # CHEAPER / LARGER carry over the amenity filter but skip the NEWSEARCH gap calc
    if is_overview and not no_results and not requested_am and filters.get("amenities_wanted"):
        requested_am, satisfied_am, missing_am = _amenity_gap_summary(matched_listings, filters)

    # Which search dimensions has the user NOT pinned down yet? These power a
    # genuinely contextual closing question instead of a generic "what matters most".
    _open_dims = []
    if not filters.get("locations"):
        _open_dims.append("area/neighbourhood")
    if not filters.get("bedrooms") and not filters.get("bedrooms_floor"):
        _open_dims.append("bedroom count")
    if not filters.get("types"):
        _open_dims.append("property type (house vs apartment vs portion)")
    if not filters.get("max_price"):
        _open_dims.append("budget")
    open_dims_hint = ", ".join(_open_dims) if _open_dims else "none — they've specified area, beds, type and budget"

    # Conversation depth — the opening overview should stay tight; once the user
    # has engaged and we're narrowing, a fuller advisory answer is welcome. A
    # discovery reveal has already had a full consultation, so treat it as engaged.
    is_opening = len(search_history) <= 1 and discovery_filters is None

    # Discovery reveal: area-level price expectations + prioritization advisory,
    # computed from real listings (matches the human agent's "expected pricing in
    # Gulshan vs Johar … prioritize Gulshan because …" guidance).
    advisory_block = ""
    if discovery_filters is not None and filters.get("locations") and not no_results:
        bands = _area_price_bands(filters)
        if bands:
            advisory_block = (
                f"\nArea pricing (typical band for the requested type/bedrooms in our listings):\n{bands}\n"
                "Use this to: (a) set realistic price expectations per area; (b) PRIORITIZE the areas with a "
                "one-line reason each, grounded in what the buyer told you (routine, kids' schools, space for "
                "budget); (c) suggest a focused shortlist of about 4-5 split across these areas. "
                "Do NOT mention scheduling visits or price negotiation."
            )

    # ── CALL 1: Generate conversational response ──
    if channel == "whatsapp" and not no_results:
        _wa_missing = f" If no result has {', '.join(missing_am)}, add a brief honest clause like '— none have {', '.join(missing_am)}'." if missing_am else ""
        system_prompt = f"""You are a sharp, friendly real estate advisor on WhatsApp — not a search engine. Property cards are sent right after your message, so do NOT list individual properties or repeat their price, beds, or size — the cards carry those.
Write a warm, advisory overview in 2-3 short sentences:
- First, frame the set: how many options were found and their price range, with a light human touch on the main areas (e.g. "budget-friendly areas like New Karachi and Landhi") rather than a flat list.
- Then give a small, honest POINT OF VIEW — a gentle recommendation with the reasoning, grounded in what the user has and hasn't told you (e.g. "rather than the cheapest 1-bed, the 2-bed is worth a look first for real family space within the same budget"). When you name a property, write its title EXACTLY as it appears in the overview, never an ID number.{_wa_missing}
Do NOT end with a question — a follow-up with tappable options is sent separately right after. If more than 5 results were found, mention the top 5 are shown below. Friendly tone, at most one emoji. Only use facts from the "Results overview" below. Match the user's language (Urdu or English)."""
    elif no_results:
        system_prompt = """You are a helpful real estate assistant. No properties were found.
Rules:
- ONE sentence saying nothing matched — friendly, not apologetic
- ONE sentence suggesting a specific alternative (different area, higher budget, or different type)
- Never say "I'm sorry", "I apologize", or anything formal
- Speak like a helpful friend
- Match the user's language (Urdu or English)"""
    elif is_overview:
        if is_opening:
            length_guidance = """This is the user's FIRST search — keep it SHORT, ~55-70 words total, easy to skim. Write ONE compact 2-sentence body, then the closing question on its own line. Sentence 1: how many options + the price band + a SHORT area characterization (name at most 2-3 areas, or just say "budget-friendly areas" — do NOT list every neighbourhood). Sentence 2: ONE recommendation with a brief why — no second pick, no amenity list. Then the question. If in doubt, cut."""
        else:
            length_guidance = """The conversation is already going, so a brief advisory answer is welcome — but keep it tight: at most two short paragraphs, and shorter is better. You may compare two options only if it genuinely helps the user decide. Cut anything that isn't pulling its weight. Always put the closing question on its own line, separated from the paragraph above by a blank line."""

        system_prompt = f"""You are a sharp, friendly real estate advisor for Karachi — not a search engine. The user just searched and a grid of property cards is shown beside this chat. Talk like a knowledgeable friend who is helping them think, not a bot reading out a list.

{length_guidance}

Cover these three beats (no headings, no bullets, plain text only) — scale how much you write to the length guidance above:

1) FRAME THE SET: how many options, the price range, and the main areas — with a light human touch on those areas (e.g. "budget-friendly areas like New Karachi and Landhi"). Don't just dump a comma list.

2) GIVE A POINT OF VIEW — this is what makes you useful, not robotic. Don't just name the cheapest and the top match neutrally. Give a small, honest recommendation WITH the reasoning, grounded in what the user has and hasn't told you. Example of the *kind* of judgement (adapt to the actual data, never invent): if they gave a budget but no bedrooms, gently steer toward the choice that's better practical value rather than just the cheapest ("rather than the cheapest 1-bed, the 2-bed in X is worth a look first because it gives real family space within the same budget"). When you name a property, write its title EXACTLY as it appears in the overview so it can be linked, and NEVER use an ID number. Mention a standout amenity only if it genuinely strengthens the pick. If the user asked for an amenity that NO result has, say so honestly in one short clause — never pretend it's there.

3) CLOSE WITH ONE CONTEXTUAL QUESTION (always 1 sentence) — the most important line. Put it on its OWN line, separated from the paragraph above by a blank line. This is what makes the chat feel alive. Ask ONE specific, natural follow-up that moves the user forward by targeting something they HAVEN'T pinned down yet. Dimensions still open for this user: {open_dims_hint}. Pick the one or two that matter most and ask about them concretely — e.g. "Want me to narrow these by area, or by how many bedrooms you need?" NEVER ask a generic "what matters most to you: price, size, or location?" or "anything else?" — it must feel like it was written for THIS search.

Hard rules:
- Never formal, never "I'm sorry"/"I apologize"/"database". No markdown (**bold**, *italics*, backticks).
- Do NOT restate the user's own criteria back to them as if informing them. Do NOT describe every property or repeat per-card details — the grid shows those.
- Only use facts from the "Results overview" below — never invent prices, areas, or amenities.
- Match the user's language (Urdu or English).

User's query (already-known context — do NOT repeat it back): "{user_query}" """

    else:
        system_prompt = f"""You are a sharp, friendly real estate advisor for Karachi. Keep it SHORT and NATURAL — talk like a knowledgeable friend, not customer service.
Rules:
- 2-3 short sentences total.
- Never say "I'm sorry", "I apologize", "database", or anything formal.
- NEVER repeat back criteria the user already stated — they know what they searched for. Instead ADD what they couldn't already know: standout amenities, size (sq yd), value vs other options, notable features (sea view, gated community, pool, etc.).
- If the query already specifies location + type + bedrooms, skip restating those — lead with price and then a differentiating detail.
- Describe ONE property only — the FIRST in the Available properties list. Do not mention other options.
- End with ONE short, contextual follow-up question that moves them forward — e.g. offer to line it up against a cheaper option, pull the agent's contact, or show more photos. Make it specific to THIS property, never a generic "anything else?". Put this question on its own line, separated from the text above by a blank line.
- NEVER reference any property by ID number. Never invent details not in 'Available properties' below.
- If the user asked for specific amenities not available in any shown property, acknowledge it honestly in one short clause (e.g. "though it doesn't have on-site security"). Never pretend a missing amenity is present.
- Match the user's language (Urdu or English).

User's query (treat this as already-known context — do NOT repeat it back): "{user_query}" """

    # On a zero-result discovery reveal, find where the requested type/beds/budget
    # DOES exist so we can suggest concrete alternative areas instead of a dead-end.
    alt_block = ""
    if no_results and filters.get("locations"):
        alt = [a for a in _areas_for_budget({k: v for k, v in filters.items() if k != "locations"})
               if a.lower() not in " ".join(filters["locations"]).lower()]
        if alt:
            beds = filters.get("bedrooms")
            what = f"{beds}-bed " if beds else ""
            what += "/".join(filters.get("types") or [""])
            alt_block = (f"Nothing matched in {', '.join(filters['locations'])}, but {what} options in budget "
                         f"DO exist in: {', '.join(alt[:4])}. Suggest one or two of these concretely, and offer "
                         "to broaden the area or adjust bedrooms.")

    if no_results:
        results_block = "No matching properties found."
    elif channel == "whatsapp" or is_overview:
        results_block = f"Results overview:\n{overview}"
    else:
        results_block = f"Available properties:{context}"

    # Turn the relaxed-field names into one honest, human clause for the reveal copy.
    relax_note = ""
    if discovery_dropped and not no_results:
        _RELAX_PHRASE = {
            "bedrooms": "included options one bedroom off from what you asked",
            "budget":   "added a few options a little above your budget",
            "area":     "looked just beyond your preferred area",
            "floor_band": "eased the floor preference",
            "has_lift": "eased the lift requirement",
        }
        phrases = []
        for f in discovery_dropped:
            phrases.append(_RELAX_PHRASE.get(f, "eased a couple of the softer preferences"))
        # de-dupe while preserving order
        seen = set(); phrases = [p for p in phrases if not (p in seen or seen.add(p))]
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        relax_note = (f"NOTE: nothing matched every criterion exactly, so to give a real shortlist I {joined}. "
                      "Mention this honestly in ONE short, warm clause — don't apologise or over-explain.")

    user_prompt = f"""Conversation history:
{history_text}

User query: {user_query}

{results_block}

{f"User requested amenities: {requested_am or 'none'}. Available in results: {satisfied_am or 'none'}. NOT available in any result: {missing_am or 'none'}." if not no_results else ""}
{relax_note}
{advisory_block}
{alt_block}
{"Tell the user nothing matched and suggest what to try." if no_results else "Answer naturally based on the information above."}"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    ai_response = response.content

    # Web shows the follow-up question inline; keep it on its own line. (On
    # WhatsApp CALL 1 has no trailing question, so this is a no-op there.)
    if channel != "whatsapp":
        ai_response = _split_closing_question(ai_response)

    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(ai_response)

    # ── CALL 2: Decide follow-up actions ──
    action_decision = decide_actions(
        user_query=user_query,
        classification_type=classification_type,
        matched_listings=matched_listings,
        filters=filters,
        search_history=search_history,
        channel=channel,
        history_text=history_text,
        open_dims_hint=open_dims_hint
    )

    return {
        "response": ai_response,
        "stage": "results",
        "discovery_complete": True,
        "match_count": len(matched_listings),
        "accumulated_filters": filters,
        "next_question": None,
        "listings": matched_listings,
        "filters": filters,
        "follow_up": action_decision.get("follow_up"),
        "actions": action_decision.get("actions", []),
        "meta": {
            "no_results": no_results,
            "action": action_label,
            "top_pick_id": matched_listings[0]["metadata"]["id"] if matched_listings else None,
            "new_results": classification_type in ("NEWSEARCH", "CHEAPER", "LARGER"),
            "highlight_id": highlight_id,
            "discovery_dropped": discovery_dropped,
        }
    }