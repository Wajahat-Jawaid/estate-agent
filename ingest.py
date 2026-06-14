import json
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def price_to_lacs(price_str: str) -> int:
    """Convert Pakistani price string to numeric lacs value"""
    price_str = price_str.lower().strip()
    crore = 0
    lac = 0

    crore_match = re.search(r'(\d+)\s*crore', price_str)
    lac_match = re.search(r'(\d+)\s*lac', price_str)

    if crore_match:
        crore = int(crore_match.group(1))
    if lac_match:
        lac = int(lac_match.group(1))

    return (crore * 100) + lac

def listing_to_text(listing):
    amenities = ", ".join(listing.get("amenities", []))
    floor = listing.get("floor", 0)
    floor_desc = "ground floor" if floor == 0 else f"floor {floor}"
    band = listing.get("floor_band", "")
    total_floors = listing.get("total_floors", "")
    lift_desc = "yes" if listing.get("has_lift") else "no"
    west_desc = "yes, west open" if listing.get("west_open") else "no"
    nearby = [label for key, label in (
        ("near_hospital", "hospital"), ("near_school", "school"),
        ("near_park", "park"), ("near_masjid", "masjid/mosque"),
    ) if listing.get(key)]
    nearby_desc = ", ".join(nearby) if nearby else "none noted"
    gated_desc = "yes, gated/secure community" if listing.get("gated_community") else "no"
    return f"""
Property ID: {listing['id']}
Title: {listing['title']}
Type: {listing['type']}
Bedrooms: {listing['bedrooms']}
Bathrooms: {listing['bathrooms']}
Area: {listing['area_sqyd']} square yards
Price: {listing['price']}
Location: {listing['location']}
Floor: {floor_desc} ({band} floor of {total_floors})
Lift: {lift_desc}
West open: {west_desc}
Possession: {listing.get('possession', 'ready')}
Nearby amenities: {nearby_desc}
Gated community: {gated_desc}
Amenities: {amenities}
Agent: {listing['agent']}
Contact: {listing['contact']}
Map: {listing.get('map_url', '')}
""".strip()

# Load listings
print("Loading listings...")
with open("data/listings.json", "r") as f:
    listings = json.load(f)

texts = [listing_to_text(l) for l in listings]
ids = [str(l["id"]) for l in listings]
metadatas = [
    {
        "id": l["id"],
        "title": l["title"],
        "price": l["price"],
        "price_numeric": price_to_lacs(l["price"]),
        "location": l["location"],
        "type": l["type"],
        "bedrooms": l["bedrooms"],
        "contact": l["contact"],
        "agent": l["agent"],
        "image_url": l.get("image_url", ""),
        "images": json.dumps(l.get("images", [])),
        "bathrooms": l.get("bathrooms", 0),
        "area_sqyd": l.get("area_sqyd", 0),
        "area_sqft": l.get("area_sqft", 0),
        "map_url": l.get("map_url", ""),
        "amenities": json.dumps(l.get("amenities", [])),
        # Discovery inference fields (added Phase 0) — scalar so Chroma can filter on them
        "floor": l.get("floor", 0),
        "near_hospital": bool(l.get("near_hospital", False)),
        "near_school": bool(l.get("near_school", False)),
        "near_park": bool(l.get("near_park", False)),
        "near_masjid": bool(l.get("near_masjid", False)),
        "gated_community": bool(l.get("gated_community", False)),
        # Data enrichment round 2 (for deep family discovery)
        "total_floors": l.get("total_floors", 0),
        "floor_band": l.get("floor_band", ""),
        "has_lift": bool(l.get("has_lift", False)),
        "west_open": bool(l.get("west_open", False)),
        "possession": l.get("possession", "ready"),
    }
    for l in listings
]

# Quick sanity check
print("Sample price conversions:")
for l in listings[:5]:
    print(f"  {l['price']} → {price_to_lacs(l['price'])} lacs")

# Load embeddings
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Delete existing ChromaDB and rebuild cleanly
import shutil
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Cleared old ChromaDB")

print("Embedding and storing in ChromaDB...")
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    ids=ids,
    metadatas=metadatas,
    persist_directory="chroma_db"
)

print(f"\nDone! Total vectors stored: {vectorstore._collection.count()}")