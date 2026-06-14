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

AMENITY_SYNONYMS = {
    "Nearby Schools":                  ["school", "schools", "education", "study ke liye", "bachon ka school"],
    "Nearby Shopping Malls":           ["market", "markets", "mall", "malls", "shopping", "bazaar", "bazar"],
    "Nearby Hospitals":                ["hospital", "hospitals", "clinic", "medical nearby", "doctor"],
    "Nearby Restaurants":              ["restaurant", "restaurants", "food", "dining", "khane"],
    "Nearby Public Transport Service": ["transport", "bus", "metro", "public transport", "commute"],
    "Security Staff":                  ["security", "guard", "secure", "mehfooz", "safe area", "safety"],
    "CCTV Security":                   ["cctv", "cameras", "surveillance"],
    "Maintenance Staff":               ["maintenance", "upkeep", "maintained"],
    "Community Gym":                   ["gym", "fitness", "workout"],
    "Swimming Pool":                   ["pool", "swimming"],
    "Kids Play Area":                  ["play area", "kids area", "playground", "bachon ke liye"],
    "Community Lawn or Garden":        ["lawn", "garden", "green", "park", "outdoor space"],
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
    "near_masjid": 0.05, "gated_community": 0.06, "west_open": 0.06,
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
    (("park", "garden", "jogging", "playground"),
     {"near_park": True}),
]

_SITUATIONAL_FIELDS = ("floor", "near_hospital", "near_school", "near_park",
                       "near_masjid", "gated_community")

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
_LIFT_KW = ("lift", "elevator")
_WEST_OPEN_KW = ("west open", "west-open", "west khula", "westopen")

def infer_floor_pref(query: str) -> dict:
    q = query.lower()
    out = {}
    for triggers, fields in _FLOOR_PREF_RULES:
        if any(t in q for t in triggers):
            out.update(fields)
    if any(t in q for t in _LIFT_KW):
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


def _merge_discovery_filters(state: dict, user_query: str) -> list:
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
    if explicit.get("types") and not _mentions_type(user_query):
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
    det_budget = _parse_budget_ceiling(user_query)
    if det_budget:
        explicit["max_price"] = det_budget
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
    if not merged.get("max_price"):
        ceiling = _parse_budget_ceiling(user_query)
        if ceiling:
            merged["max_price"] = ceiling

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

    state["filters"] = merged
    return dropped


MAX_ASK_PER_DIM = 2  # re-ask an unanswered spec dimension at most twice

def _next_discovery_dim(state: dict):
    """Deterministic, human-shaped order: purpose → budget → bedrooms → area →
    type → floor/lift → extras → recap. (A measured A/B showed this beats letting
    an LLM pick the order — it aligns better with how real agents consult and is
    cheaper + stable.) A dim is skipped once KNOWN; re-asked up to MAX_ASK_PER_DIM
    if still unknown (extraction is flaky); never reveals half-blind nor loops."""
    f = state["filters"]
    asked = state["asked"]

    def need(dim, known):
        return not known and asked.get(dim, 0) < MAX_ASK_PER_DIM

    if not f.get("purpose") and asked.get("purpose", 0) < 1:  # opener — ask at most once
        return "purpose"
    if need("budget", f.get("max_price")):
        return "budget"
    # Possession is a top concern for overseas / document-cautious buyers — ask it
    # early (right after budget), the way a real agent does, not buried at the end.
    if state.get("possession_relevant") and need("possession", f.get("possession")):
        return "possession"
    if need("household", f.get("bedrooms") or f.get("bedrooms_floor") or f.get("bedrooms_min")):
        return "household"
    if need("area", f.get("locations")):
        return "area"
    if need("type", f.get("types")):
        return "type"
    if need("floor", f.get("floor_band") or f.get("has_lift")):
        return "floor"
    if asked.get("extras", 0) < 1:
        return "extras"
    if asked.get("recap", 0) < 1:
        return "recap"
    return None


def _discovery_ready(state: dict) -> bool:
    return state["turns"] >= DISCOVERY_MAX_QUESTIONS or _next_discovery_dim(state) is None


_DISCOVERY_DIM_PROMPT = {
    "purpose": "Open by asking, naturally, whether this is for their own living or for investment — it "
               "shapes everything that follows. One short, warm question.",
    "household": "Ask how many family members will live there and who (kids, parents), so you can judge "
                 "the right number of bedrooms. Warm and brief.",
    "type": "Ask whether they prefer a house, apartment, or portion (and note if they're open to portions).",
    "budget": "Ask warmly what budget you should keep in mind (they'll answer in lac or crore).",
    "area": "Ask which area(s) of Karachi they prefer, or whether they're open to your suggestions.",
    "floor": "Ask their floor preference (ground / middle / top) and whether a lift is required — this "
             "matters for elderly family members visiting.",
    "possession": "Ask whether they need ready possession / ready-to-move only (with clear documents), "
                  "or are open to under-construction — important for overseas or document-cautious buyers.",
    "extras": "Ask if there are any must-have features or nice-to-haves — e.g. dedicated parking, "
              "generator/electricity backup, drawing room, west-open, good maintenance, schools nearby. "
              "Ask it naturally, not as a long checklist.",
}


def _discovery_recap(filters: dict, lang: str = "English") -> str:
    """One-sentence recap of gathered requirements + a 'did I get that right?'."""
    parts = []
    if filters.get("bedrooms"):
        parts.append(f"{filters['bedrooms']} bedrooms")
    if filters.get("types"):
        parts.append("/".join(filters["types"]))
    if filters.get("floor_band"):
        parts.append(f"{filters['floor_band']} floor")
    if filters.get("has_lift"):
        parts.append("lift")
    if filters.get("max_price"):
        parts.append(f"budget up to {filters['max_price']}")
    if filters.get("locations"):
        parts.append("areas: " + ", ".join(filters["locations"]))
    if filters.get("possession") == "ready":
        parts.append("ready possession with clear documents")
    extras = [str(x) for x in (filters.get("features") or [])]
    if filters.get("west_open"):
        extras.append("west open")
    summary = "; ".join(parts)
    extra_str = (" Also: " + ", ".join(extras)) if extras else ""
    prompt = f"""You are a sharp, warm Karachi real-estate advisor. RECAP the buyer's requirements back naturally in ONE flowing sentence to confirm you understood — like a real agent summarising before pulling options — then ask "Did I get that right?". Requirements: {summary}.{extra_str}
Reply ONLY in {lang}. No markdown, no bullet points. Max 2 sentences. Sound human, not like a checklist read-out."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


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
    """Data-grounded area suggestions: parent areas that actually have stock
    matching the buyer's type + bedrooms + budget. Lets the bot advise on areas
    like a human agent ("in this budget, Gulshan/Johar/Scheme 33 are practical")."""
    conds = []
    types = [t for t in (filters.get("types") or []) if isinstance(t, str)]
    if len(types) == 1:
        conds.append({"type": {"$eq": types[0]}})
    elif len(types) > 1:
        conds.append({"type": {"$in": types}})
    if filters.get("bedrooms") is not None:
        conds.append({"bedrooms": {"$eq": filters["bedrooms"]}})
    if filters.get("max_price"):
        ceil = int(price_to_lacs(str(filters["max_price"])) * 1.1)
        if ceil > 0:
            conds.append({"price_numeric": {"$lte": ceil}})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
    try:
        metas = vectorstore._collection.get(where=where, include=["metadatas"]).get("metadatas", []) or []
    except Exception:
        return []
    from collections import Counter

    def parent(loc):
        a = loc.split(",")[0].strip()
        return re.sub(r'\s+(?:phase|block|sector|extension|scheme)\s+\S+.*$', '', a, flags=re.IGNORECASE).strip()

    counts = Counter(parent(m.get("location", "")) for m in metas if m.get("location"))
    return [a for a, _ in counts.most_common(limit) if a]


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
    if count_matches(filters) > 0:
        return None  # actually feasible — no pushback
    alt = {k: v for k, v in filters.items() if k != "locations"}
    areas = _areas_for_budget(alt, limit=5)
    areas_str = ", ".join(areas) if areas else "areas like PECHS, Gulshan, or Gulistan-e-Johar"
    beds = filters.get("bedrooms")
    size = f"{beds}-bed " if beds else ""
    typ = "/".join(filters.get("types") or ["property"])
    prompt = f"""You are a candid, helpful Karachi real-estate agent. The buyer wants a {size}{typ} in {', '.join(filters.get('locations'))} at {filters.get('max_price')}, but at that budget there is essentially nothing there without big compromises (old building, very small unit, no parking, poor block).
In about 2 natural sentences: honestly but warmly tell them this budget is very tight for that size in those premium areas, name the kind of compromise it would force, and ask whether they're open to better-value areas ({areas_str}) or whether those areas are a must. Sound like a real advisor giving honest guidance, not a bot. Keep it tight. Reply ONLY in {lang}. No markdown."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _widen_discovery_search(filters: dict, search_query: str, target: int = 4):
    """A real agent shortlists ~4-5 options. Our hard filters (floor/lift/possession
    + exact bedrooms) often AND down to ~1, so on a discovery reveal, progressively
    relax the least-essential ones until there are enough options. Never relaxes
    budget/area/type. Returns (listings, context, relaxed_field_names)."""
    # Only relax INFERRED/soft structural prefs. NEVER relax explicitly-stated
    # requirements (possession, bedrooms, budget, area, type) — padding the
    # shortlist with properties that violate what the buyer insisted on is worse
    # than showing fewer. (This was surfacing non-possession/wrong-bed options to
    # the overseas buyer who required ready-possession 3-bed.)
    relax_order = ["floor_band", "has_lift"]
    cur = dict(filters)
    relaxed = []
    results = search_properties(search_query, cur)
    ml, ctx = _results_to_listings(results, cur)
    for fld in relax_order:
        if len(ml) >= target:
            break
        if fld in cur:
            cur = {k: v for k, v in cur.items() if k != fld}
            relaxed.append(fld)
            results = search_properties(search_query, cur)
            ml, ctx = _results_to_listings(results, cur)
    return ml, ctx, relaxed


def _discovery_question(dimension: str, filters: dict, history_text: str, count: int,
                        lang: str = "English", user_query: str = "") -> str:
    known = []
    if filters.get("max_price"):
        known.append(f"budget ~{filters['max_price']}")
    if filters.get("locations"):
        known.append("areas: " + ", ".join(filters["locations"]))
    if filters.get("types"):
        known.append("type: " + ", ".join(filters["types"]))
    if filters.get("bedrooms"):
        known.append(f"{filters['bedrooms']} bed")
    known_str = "; ".join(known) if known else "nothing specific yet"
    extra = ""
    if dimension == "area":
        areas = _areas_for_budget(filters)
        if areas:
            extra = (f"\nAreas that actually have matching stock in their budget right now: {', '.join(areas)}. "
                     "Proactively suggest 3-4 of these as practical fits and say a word on why; if they'd want "
                     "pricier areas like DHA or Clifton, gently note those are hard for this size/budget.")
    prompt = f"""You are a sharp, warm Karachi real-estate advisor mid-consultation (before showing listings) — talk like an experienced human agent, NEVER a form or slot-filler.
What you already know about this buyer: {known_str}.
The customer just said: "{user_query}"

Respond the way a real agent would:
1. If they ASKED a question or raised a concern, ANSWER it directly and helpfully FIRST — you're the expert (e.g. actually explain the compromises, give honest guidance). Never dodge their question by asking another one.
2. Briefly acknowledge what they told you, in fresh wording.
3. Then, only if it flows naturally, move forward by learning this next: {_DISCOVERY_DIM_PROMPT[dimension]}{extra}

Hard rules: NEVER ask a question you've already asked in the conversation below — read it first; if you already asked something, do not ask it again. Do NOT interrogate — at most ONE new question, and it's fine to ask none if answering them is what matters this turn. Keep it brief and natural: no lists/bullets/markdown, 1-2 short sentences. Reply ONLY in {lang}.
Conversation so far:
{history_text}
Return ONLY your reply."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _discovery_ask_response(disc: dict) -> dict:
    """Build the API response for a discovery question turn (no results yet)."""
    return {
        "response": disc["question"],
        "stage": "discovery",
        "discovery_complete": False,
        "match_count": disc["count"],
        "accumulated_filters": disc["filters"],
        "next_question": disc["question"],
        "listings": [],
        "filters": disc["filters"],
        "follow_up": None,
        "actions": [],
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

    dropped = _merge_discovery_filters(state, user_query)
    count = count_matches(state["filters"])
    wants_results = any(kw in user_query.lower() for kw in _SHOW_RESULTS_KW)

    # Budget reality check (once): premium-only areas at a budget with no stock for
    # the size — be honest and redirect, like a real agent, instead of forcing it.
    if not state.get("pushed_back"):
        reality = _budget_reality_message(state["filters"], _detect_language(user_query + " " + history_text))
        if reality:
            state["pushed_back"] = True
            state["turns"] += 1
            return {"mode": "ask", "question": reality, "count": count,
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
    state["turns"] += 1
    lang = _detect_language(user_query + " " + history_text)
    if dim == "recap":
        question = _discovery_recap(state["filters"], lang)
    else:
        question = _discovery_question(dim, state["filters"], history_text, count, lang, user_query)
    return {
        "mode": "ask",
        "question": question,
        "count": count,
        "filters": dict(state["filters"]),
        "dropped": dropped,
    }


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
            length_guidance = """The conversation is already going, so a brief advisory answer is welcome — but keep it tight: at most two short paragraphs, and shorter is better. You may compare two options only if it genuinely helps the user decide. Cut anything that isn't pulling its weight."""

        system_prompt = f"""You are a sharp, friendly real estate advisor for Karachi — not a search engine. The user just searched and a grid of property cards is shown beside this chat. Talk like a knowledgeable friend who is helping them think, not a bot reading out a list.

{length_guidance}

Cover these three beats (no headings, no bullets, plain text only) — scale how much you write to the length guidance above:

1) FRAME THE SET: how many options, the price range, and the main areas — with a light human touch on those areas (e.g. "budget-friendly areas like New Karachi and Landhi"). Don't just dump a comma list.

2) GIVE A POINT OF VIEW — this is what makes you useful, not robotic. Don't just name the cheapest and the top match neutrally. Give a small, honest recommendation WITH the reasoning, grounded in what the user has and hasn't told you. Example of the *kind* of judgement (adapt to the actual data, never invent): if they gave a budget but no bedrooms, gently steer toward the choice that's better practical value rather than just the cheapest ("rather than the cheapest 1-bed, the 2-bed in X is worth a look first because it gives real family space within the same budget"). When you name a property, write its title EXACTLY as it appears in the overview so it can be linked, and NEVER use an ID number. Mention a standout amenity only if it genuinely strengthens the pick. If the user asked for an amenity that NO result has, say so honestly in one short clause — never pretend it's there.

3) CLOSE WITH ONE CONTEXTUAL QUESTION (always 1 sentence) — the most important line. This is what makes the chat feel alive. Ask ONE specific, natural follow-up that moves the user forward by targeting something they HAVEN'T pinned down yet. Dimensions still open for this user: {open_dims_hint}. Pick the one or two that matter most and ask about them concretely — e.g. "Want me to narrow these by area, or by how many bedrooms you need?" NEVER ask a generic "what matters most to you: price, size, or location?" or "anything else?" — it must feel like it was written for THIS search.

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
- End with ONE short, contextual follow-up question that moves them forward — e.g. offer to line it up against a cheaper option, pull the agent's contact, or show more photos. Make it specific to THIS property, never a generic "anything else?".
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

    user_prompt = f"""Conversation history:
{history_text}

User query: {user_query}

{results_block}

{f"User requested amenities: {requested_am or 'none'}. Available in results: {satisfied_am or 'none'}. NOT available in any result: {missing_am or 'none'}." if not no_results else ""}
{f"NOTE: no property matched the inferred requirement(s) {discovery_dropped}, so that was relaxed — gently mention this in one short clause." if discovery_dropped else ""}
{advisory_block}
{alt_block}
{"Tell the user nothing matched and suggest what to try." if no_results else "Answer naturally based on the information above."}"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    ai_response = response.content

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