"""Spot-check rent estimates across areas and property types."""
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rental_yield import estimate_rent

QUERIES = [
    # (query string, k, label)
    ("DHA Phase 8 luxury house",              2, "DHA Phase 8 – house"),
    ("Clifton apartment for sale",            2, "Clifton – apartment"),
    ("PECHS house Karachi",                   1, "PECHS – house"),
    ("Gulshan-e-Iqbal apartment",             2, "Gulshan-e-Iqbal – apartment"),
    ("Gulistan-e-Johar house",                1, "Gulistan-e-Johar – house"),
    ("FB Area apartment",                     1, "FB Area – apartment"),
    ("Nazimabad house",                       1, "Nazimabad – house"),
    ("Bahria Town apartment",                 1, "Bahria Town – apartment"),
    ("North Karachi apartment",               1, "North Karachi – apartment"),
    ("Korangi house",                         1, "Korangi – house"),
    ("Landhi house Karachi",                  1, "Landhi – house"),
    ("Bin Qasim Super Highway property",      1, "Outer area – house"),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

seen_ids: set = set()
rows: list[dict] = []

for query, k, label in QUERIES:
    if len(rows) >= 14:
        break
    docs = db.similarity_search(query, k=k)
    for doc in docs:
        m = doc.metadata
        pid = m.get("id")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        r = estimate_rent(
            price_numeric=m.get("price_numeric", 0),
            location=m.get("location", ""),
            property_type=m.get("type", "house"),
        )
        rows.append({
            "label":    label,
            "title":    m.get("title", "")[:38],
            "location": m.get("location", "")[:32],
            "price":    m.get("price", ""),
            "type":     m.get("type", "house"),
            "rate":     r["rate"],
            "y_lo":     r["yield_low"],
            "y_hi":     r["yield_high"],
            "rent_lo":  r["monthly_rent_low"],
            "rent_hi":  r["monthly_rent_high"],
        })
        break  # one per query to keep variety

# ── print table ──────────────────────────────────────────────────────────────
FMT = "{:<38}  {:<32}  {:>18}  {:>11}  {:>6}  {:>9}  {:>26}  {}"
header = FMT.format("Title", "Location", "Price", "Type", "Rate", "Yield%", "Monthly Rent (PKR)", "Band")
print(header)
print("-" * len(header))

for r in rows:
    band = "HIGH" if r["y_lo"] >= 6.5 else ("MID" if r["y_lo"] >= 5.5 else "PREMIUM")
    rent_range  = f"{r['rent_lo']:,} – {r['rent_hi']:,}"
    yield_range = f"{r['y_lo']}–{r['y_hi']}%"
    print(FMT.format(
        r["title"], r["location"], r["price"], r["type"],
        f"{r['rate']:,}", yield_range, rent_range, band,
    ))
