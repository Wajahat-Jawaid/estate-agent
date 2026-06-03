import json
import random

random.seed(42)

# ── Image pool (50 total) ──────────────────────────────────────────────────────
POOL_EXTERIOR = [f"https://picsum.photos/seed/realestate{i}/800/600" for i in range(1, 16)]  # 15
POOL_BUILDING = [f"https://picsum.photos/seed/building{i}/800/600"   for i in range(1, 11)]  # 10
POOL_INTERIOR = [f"https://picsum.photos/seed/interior{i}/800/600"   for i in range(1, 11)]  # 10
POOL_LUXURY   = [f"https://picsum.photos/seed/luxury{i}/800/600"     for i in range(1,  9)]  #  8
POOL_OUTDOOR  = [f"https://picsum.photos/seed/outdoor{i}/800/600"    for i in range(1,  8)]  #  7

ALL_IMAGES = POOL_EXTERIOR + POOL_BUILDING + POOL_INTERIOR + POOL_LUXURY + POOL_OUTDOOR  # 50

TYPE_PRIMARY = {
    "house":         POOL_EXTERIOR,
    "apartment":     POOL_BUILDING + POOL_INTERIOR[:5],
    "upper portion": POOL_EXTERIOR[:8],
    "lower portion": POOL_EXTERIOR[:8],
    "penthouse":     POOL_LUXURY + POOL_BUILDING[:2],
    "farmhouse":     POOL_OUTDOOR,
}

def primary_image(pid, ptype):
    pool = TYPE_PRIMARY[ptype]
    return pool[pid % len(pool)]

def property_images(pid, count=10):
    return [ALL_IMAGES[(pid * 7 + i * 11) % 50] for i in range(count)]

# ── Agents ────────────────────────────────────────────────────────────────────
AGENTS = [
    "Ahmed Siddiqui","Maria Hassan","Junaid Khan","Saima Akhtar","Rizwan Ahmed",
    "Farah Naz","Asim Butt","Heba Shaikh","Waqas Mirza","Aisha Qureshi",
    "Mushtaq Ahmed","Layla Hussain","Shahzad Ali","Robia Khan","Imran Siddiqui",
    "Nabila Rehman","Zaheer Abbas","Sadia Ahmed","Owais Malik","Farida Begum",
    "Hamza Sheikh","Naila Iqbal","Bilal Awan","Zubaida Khatoon","Mohsin Khan",
    "Aneela Rizvi","Saleem Butt","Nasim Akhtar","Ghazala Begum","Yasir Mehmood",
    "Mehreen Syed","Arshad Nawaz","Sumbal Wahab","Khalil Ahmed","Ruba Siddiqui",
    "Tariq Farooq","Neelam Baig","Zohaib Malik","Sheeba Naz","Rashid Butt",
    "Hajra Qureshi","Nasir Ali","Kausar Begum","Danish Raza","Shireen Akhtar",
    "Bashir Ahmed","Rumana Syed","Akbar Khan","Parveen Zaman","Nadeem Hussain",
    "Gulzar Ahmed","Fateh Ali","Samreen Baig","Irshad Malik","Tabassum Begum",
    "Haroon Qureshi","Zobia Khan","Feroze Ali","Mehwish Siddiqui","Kashif Butt",
    "Ambreena Malik","Wajid Hussain","Surriya Begum","Qasim Iqbal","Asifa Naz",
    "Babar Qureshi","Nusrat Perveen","Shahzaib Khan","Nida Baig","Faizan Raza",
    "Gul Nawaz","Tasleem Ahmed","Mumtaz Hussain","Farhat Begum","Ejaz Ali",
    "Rehana Bibi","Zahid Raza","Munira Siddiqui","Khawaja Ali","Sabina Iqbal",
    "Imtiaz Butt","Shehnaaz Hussain","Yaseen Khan","Fazeelat Naz","Ansar Mahmood",
]

def get_agent(idx):
    name = AGENTS[idx % len(AGENTS)]
    contact = f"03{random.randint(10,99)}-{random.randint(1000000,9999999)}"
    return name, contact

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = {
    "house": [
        "parking","near school","near hospital","near mosque","near market",
        "near park","corner plot","generator backup","servant quarters",
        "swimming pool","renovated","quiet street","park facing","24hr security",
        "boundary wall","rooftop terrace","water pump","CCTV","huge lawn",
    ],
    "apartment": [
        "parking","gym","swimming pool","sea view","rooftop access",
        "24hr security","gated community","near market","near school",
        "near hospital","generator backup","CCTV","concierge","city view",
        "rooftop terrace","near mosque","earthquake resistant",
    ],
    "upper portion": [
        "separate entrance","near school","near market","near mosque",
        "near hospital","generator backup","water pump","quiet street",
        "parking available","garden access","rooftop access","near park",
    ],
    "lower portion": [
        "separate entrance","near school","near market","near mosque",
        "near hospital","garden","water pump","parking available",
        "quiet street","near park","CCTV","separate kitchen",
    ],
    "penthouse": [
        "parking","sea view","rooftop terrace","gym","swimming pool",
        "24hr security","private lift","panoramic view","generator backup",
        "near school","jacuzzi","smart home",
    ],
    "farmhouse": [
        "huge lawn","orchard","borehole water","generator backup",
        "caretaker quarters","farm animals space","boundary wall",
        "fruit trees","swimming pool","servant quarters",
        "sea proximity","mountain view","solar panels",
    ],
}

# ── Location configs: (location, price_min_lacs, price_max_lacs, area_sqyd_range) ──

HOUSE_LOCS = [
    # DHA
    ("DHA Phase 1, Karachi", 220,320,(180,300)),
    ("DHA Phase 2, Karachi", 240,380,(180,320)),
    ("DHA Phase 3, Karachi", 230,360,(180,300)),
    ("DHA Phase 4, Karachi", 200,350,(180,300)),
    ("DHA Phase 5, Karachi", 250,400,(200,320)),
    ("DHA Phase 6, Karachi", 220,380,(200,320)),
    ("DHA Phase 7, Karachi", 300,500,(220,400)),
    ("DHA Phase 8, Karachi", 320,600,(240,500)),
    ("DHA Phase 8 Extension, Karachi", 250,400,(200,320)),
    ("DHA Phase 6 Extension, Karachi", 300,450,(220,360)),
    # Clifton
    ("Clifton Block 1, Karachi", 350,550,(250,450)),
    ("Clifton Block 2, Karachi", 350,600,(250,500)),
    ("Clifton Block 3, Karachi", 380,650,(280,500)),
    ("Clifton Block 4, Karachi", 350,600,(250,480)),
    ("Clifton Block 5, Karachi", 400,700,(280,550)),
    ("Clifton Block 6, Karachi", 380,680,(280,520)),
    ("Clifton Block 7, Karachi", 500,900,(350,650)),
    ("Clifton Block 8, Karachi", 450,800,(300,600)),
    ("Clifton Block 9, Karachi", 480,850,(320,620)),
    # Bahria Town
    ("Bahria Town Precinct 3, Karachi",  180,280,(220,350)),
    ("Bahria Town Precinct 5, Karachi",  190,300,(235,380)),
    ("Bahria Town Precinct 7, Karachi",  200,310,(235,380)),
    ("Bahria Town Precinct 9, Karachi",  195,305,(235,370)),
    ("Bahria Town Precinct 13, Karachi", 185,295,(220,360)),
    ("Bahria Town Precinct 14, Karachi", 190,305,(235,370)),
    ("Bahria Town Precinct 17, Karachi", 185,300,(235,360)),
    ("Bahria Town Precinct 18, Karachi", 195,310,(235,370)),
    ("Bahria Town Precinct 20, Karachi", 200,320,(235,380)),
    ("Bahria Town Precinct 23, Karachi", 190,300,(235,360)),
    ("Bahria Town Precinct 28, Karachi", 185,295,(220,350)),
    ("Bahria Town Precinct 32, Karachi", 180,290,(220,340)),
    # Gulshan-e-Iqbal
    ("Gulshan-e-Iqbal Block 4, Karachi",  160,270,(180,280)),
    ("Gulshan-e-Iqbal Block 5, Karachi",  155,260,(180,280)),
    ("Gulshan-e-Iqbal Block 8, Karachi",  165,275,(180,290)),
    ("Gulshan-e-Iqbal Block 10A, Karachi",158,265,(180,280)),
    ("Gulshan-e-Iqbal Block 12, Karachi", 162,270,(180,285)),
    ("Gulshan-e-Iqbal Block 14, Karachi", 155,260,(175,280)),
    # North Nazimabad
    ("North Nazimabad Block A, Karachi", 130,220,(180,270)),
    ("North Nazimabad Block B, Karachi", 128,215,(175,265)),
    ("North Nazimabad Block D, Karachi", 132,225,(180,270)),
    ("North Nazimabad Block E, Karachi", 125,210,(175,260)),
    ("North Nazimabad Block G, Karachi", 130,220,(180,265)),
    ("North Nazimabad Block J, Karachi", 128,218,(175,265)),
    ("North Nazimabad Block K, Karachi", 130,222,(180,268)),
    ("North Nazimabad Block M, Karachi", 125,215,(175,260)),
    ("North Nazimabad Block N, Karachi", 128,218,(178,265)),
    # PECHS
    ("PECHS Block 1, Karachi", 220,360,(200,320)),
    ("PECHS Block 4, Karachi", 225,370,(200,330)),
    ("PECHS Block 5, Karachi", 230,380,(200,340)),
    # Gulistan-e-Johar
    ("Gulistan-e-Johar Block 1, Karachi",  120,200,(180,250)),
    ("Gulistan-e-Johar Block 2, Karachi",  118,195,(175,250)),
    ("Gulistan-e-Johar Block 4, Karachi",  122,205,(180,255)),
    ("Gulistan-e-Johar Block 5, Karachi",  120,200,(180,250)),
    ("Gulistan-e-Johar Block 6, Karachi",  118,198,(175,248)),
    ("Gulistan-e-Johar Block 8, Karachi",  120,200,(180,250)),
    ("Gulistan-e-Johar Block 9, Karachi",  118,196,(175,248)),
    ("Gulistan-e-Johar Block 10, Karachi", 120,200,(180,250)),
    ("Gulistan-e-Johar Block 12, Karachi", 118,196,(175,248)),
    ("Gulistan-e-Johar Block 13, Karachi", 120,200,(180,250)),
    # FB Area
    ("FB Area Block 1, Karachi",  110,180,(175,240)),
    ("FB Area Block 5, Karachi",  108,175,(170,240)),
    ("FB Area Block 8, Karachi",  110,180,(175,242)),
    ("FB Area Block 12, Karachi", 105,175,(170,235)),
    ("FB Area Block 16, Karachi", 108,178,(172,238)),
    # Askari
    ("Askari 2, Karachi", 185,290,(190,290)),
    ("Askari 6, Karachi", 190,300,(195,295)),
    # Gulshan-e-Maymar
    ("Gulshan-e-Maymar, Karachi",          150,250,(190,310)),
    ("Gulshan-e-Maymar Sector S, Karachi", 145,245,(185,305)),
    # Malir
    ("Malir Town, Karachi", 80,150,(150,250)),
    ("Malir Halt, Karachi", 82,148,(150,248)),
    # Mid-range
    ("Model Colony, Karachi",       140,230,(180,280)),
    ("Shah Faisal Colony, Karachi",  85,145,(160,250)),
    ("KDA Scheme 1, Karachi",       130,210,(175,275)),
    ("Safoora Goth, Karachi",        95,165,(165,260)),
    ("Scheme 33, Karachi",           90,155,(165,260)),
    ("Scheme 45, Karachi",          130,215,(180,270)),
    # Budget
    ("Orangi Town, Karachi",              45, 90,(80,160)),
    ("North Karachi, Karachi",           105,175,(165,260)),
    ("New Karachi, Karachi",              85,150,(155,250)),
    ("Nazimabad, Karachi",               100,175,(170,265)),
    ("Liaquatabad, Karachi",              90,150,(155,240)),
    ("Korangi, Karachi",                  80,140,(155,240)),
    ("Landhi, Karachi",                   60,110,(130,210)),
    ("Surjani Town, Karachi",             40, 85,(100,180)),
    ("Baldia Town, Karachi",              35, 75,(90,170)),
    ("Gulshan-e-Hadeed Phase 2, Karachi",150,240,(180,280)),
    ("Steel Town, Karachi",               90,160,(160,250)),
    ("Airport Area, Karachi",            115,195,(170,265)),
    ("Tariq Road Area, Karachi",         180,300,(190,300)),
]

APARTMENT_LOCS = [
    ("Clifton Block 1, Karachi",          180,400,(130,280)),
    ("Clifton Block 2, Karachi",          180,380,(120,270)),
    ("Clifton Block 4, Karachi",           90,250,(70,200)),
    ("Clifton Block 5, Karachi",          150,350,(100,260)),
    ("Clifton Block 8, Karachi",          200,450,(140,300)),
    ("Clifton Block 9, Karachi",          190,420,(130,290)),
    ("DHA Phase 2, Karachi",             140,250,(110,200)),
    ("DHA Phase 4, Karachi",             130,230,(100,190)),
    ("DHA Phase 5, Karachi",             150,280,(110,210)),
    ("DHA Phase 6, Karachi",             140,260,(105,200)),
    ("DHA Phase 8, Karachi",              80,180,(65,150)),
    ("Bahria Town Precinct 19, Karachi",  50,110,(65,120)),
    ("Bahria Town Precinct 20, Karachi",  52,112,(67,122)),
    ("Bahria Town Precinct 2, Karachi",  120,200,(100,170)),
    ("Bahria Heights, Karachi",           45,100,(60,120)),
    ("Gulshan-e-Iqbal Block 7, Karachi",  70,130,(85,140)),
    ("Gulshan-e-Iqbal Block 10, Karachi", 72,132,(85,142)),
    ("PECHS Block 2, Karachi",           120,200,(100,180)),
    ("PECHS Block 6, Karachi",           130,220,(105,185)),
    ("Askari 3, Karachi",                110,190,(100,170)),
    ("Askari 5, Karachi",                115,195,(105,175)),
    ("Saima Presidency, Karachi",         75,130,(90,145)),
    ("North Nazimabad, Karachi",          65,115,(80,135)),
    ("Gulistan-e-Johar Block 3, Karachi", 70,115,(85,135)),
    ("Emaar Coral Towers, Karachi",      380,650,(250,400)),
    ("Emaar Oceanfront, Karachi",        400,700,(260,420)),
    ("Ocean Tower, Clifton, Karachi",    350,600,(200,380)),
    ("Marina Tower, Clifton, Karachi",   320,580,(190,360)),
    ("Gulshan-e-Maymar, Karachi",         55,110,(80,140)),
    ("FB Area, Karachi",                  58,105,(80,130)),
    ("North Karachi, Karachi",            55,100,(75,130)),
    ("Scheme 33, Karachi",                60,110,(82,138)),
]

UL_LOCS = [
    ("Gulshan-e-Iqbal Block 3, Karachi",   38,68,(90,130)),
    ("Gulshan-e-Iqbal Block 9, Karachi",   36,65,(88,128)),
    ("North Nazimabad Block A, Karachi",   35,60,(85,125)),
    ("North Nazimabad Block D, Karachi",   33,58,(82,122)),
    ("North Nazimabad Block G, Karachi",   34,60,(84,124)),
    ("FB Area Block 5, Karachi",           28,50,(80,120)),
    ("FB Area Block 12, Karachi",          27,48,(78,118)),
    ("Gulistan-e-Johar Block 4, Karachi",  30,52,(85,125)),
    ("Gulistan-e-Johar Block 10, Karachi", 29,50,(82,122)),
    ("Nazimabad Block 3, Karachi",         30,52,(80,120)),
    ("Nazimabad Block 5, Karachi",         28,50,(78,118)),
    ("PECHS Block 1, Karachi",             52,90,(100,145)),
    ("Liaquatabad Block 5, Karachi",       28,48,(78,115)),
    ("New Karachi Sector 5, Karachi",      25,45,(75,110)),
    ("Surjani Town, Karachi",              20,38,(65,105)),
    ("Orangi Town, Karachi",               18,35,(60,100)),
    ("Korangi, Karachi",                   20,38,(70,110)),
    ("Baldia Town, Karachi",               16,32,(55,95)),
    ("Landhi, Karachi",                    18,35,(60,105)),
    ("Malir City, Karachi",                22,42,(70,115)),
    ("Shah Faisal Colony, Karachi",        24,44,(75,118)),
    ("Model Colony, Karachi",              32,55,(85,128)),
    ("KDA Scheme 1, Karachi",              38,68,(88,132)),
    ("Scheme 33, Karachi",                 30,52,(82,125)),
    ("DHA Phase 4, Karachi",               58,95,(98,145)),
    ("Defence View, Karachi",              45,78,(90,135)),
    ("Saima Arabian Villas, Karachi",      38,68,(85,130)),
    ("Gulshan-e-Maymar, Karachi",          28,50,(80,125)),
    ("Gulshan-e-Hadeed, Karachi",          32,56,(85,130)),
    ("Steel Town, Karachi",                25,45,(75,118)),
]

PENTHOUSE_LOCS = [
    ("Clifton Block 2, Karachi",         380,680,(240,380)),
    ("Clifton Block 5, Karachi",         400,750,(260,400)),
    ("Clifton Block 8, Karachi",         450,800,(280,420)),
    ("DHA Phase 6, Karachi",             320,580,(220,360)),
    ("DHA Phase 8, Karachi",             400,700,(250,400)),
    ("Bahria Town Precinct 2, Karachi",  280,500,(200,340)),
    ("PECHS Block 2, Karachi",           300,520,(210,350)),
    ("Emaar Coral Towers, Karachi",      600,1200,(350,600)),
    ("Ocean Tower, Clifton, Karachi",    500,900,(300,500)),
]

FARMHOUSE_LOCS = [
    ("Gadap Town, Karachi",                220,450,(3000,15000)),
    ("Super Highway, Karachi",             180,380,(5000,20000)),
    ("Hawksbay Road, Karachi",             300,600,(8000,25000)),
    ("Hub River Road, Karachi",            150,350,(4000,18000)),
    ("Deh Manghopir, Karachi",             120,280,(3000,15000)),
    ("Karachi Western Outskirts, Karachi", 200,420,(5000,22000)),
    ("Bin Qasim, Karachi",                 110,250,(2500,12000)),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def lacs_to_price(n):
    c, r = n // 100, n % 100
    if c and r:  return f"{c} crore {r} lac"
    if c:        return f"{c} crore"
    return f"{n} lac"

BED_W = {
    "house":         [2]*1+[3]*4+[4]*3+[5]*2,
    "apartment":     [1]*2+[2]*5+[3]*3+[4]*1,
    "upper portion": [2]*6+[3]*4,
    "lower portion": [2]*6+[3]*4,
    "penthouse":     [3]*3+[4]*5+[5]*2,
    "farmhouse":     [4]*3+[5]*5+[6]*2,
}

BED_SCALE = {1:0.70, 2:0.85, 3:1.00, 4:1.20, 5:1.45, 6:1.70}

def make_listing(pid, ptype, loc, pmin, pmax, area_range, agent_idx):
    agent, contact = get_agent(agent_idx)
    beds  = random.choice(BED_W[ptype])
    baths = max(1, beds - random.randint(0, 1))
    scale = BED_SCALE.get(beds, 1.0)
    price = max(pmin, min(pmax, int(random.randint(pmin, pmax) * scale)))
    area  = random.randint(*area_range)
    if ptype == "farmhouse":
        area = round(area / 500) * 500
    feats = random.sample(FEATURES[ptype], min(random.randint(3,5), len(FEATURES[ptype])))

    loc_short = loc.replace(", Karachi", "")
    if ptype in ("upper portion", "lower portion", "farmhouse"):
        title = f"{ptype.title()} in {loc_short}"
    else:
        title = f"{beds} bed {ptype} in {loc_short}"

    imgs = property_images(pid)
    return {
        "id": pid, "title": title, "type": ptype,
        "bedrooms": beds, "bathrooms": baths, "area_sqyd": area,
        "price": lacs_to_price(price), "location": loc,
        "features": feats, "agent": agent, "contact": contact,
        "image_url": primary_image(pid, ptype),
        "images": imgs,
    }

# ── Build spec list ───────────────────────────────────────────────────────────

specs = []
for loc, a, b, r in HOUSE_LOCS:
    for _ in range(3):
        specs.append(("house", loc, a, b, r))
for loc, a, b, r in APARTMENT_LOCS:
    for _ in range(3):
        specs.append(("apartment", loc, a, b, r))
for loc, a, b, r in UL_LOCS:
    specs.append(("upper portion", loc, a, b, r))
    specs.append(("lower portion", loc, a, b, r))
for loc, a, b, r in PENTHOUSE_LOCS:
    specs.append(("penthouse", loc, a, b, r))
    specs.append(("penthouse", loc, a, b, r))
for loc, a, b, r in FARMHOUSE_LOCS:
    specs.append(("farmhouse", loc, a, b, r))
    specs.append(("farmhouse", loc, a, b, r))

random.shuffle(specs)
specs = specs[:400]

# ── Generate 400 new ─────────────────────────────────────────────────────────

new_listings = []
for i, (ptype, loc, a, b, r) in enumerate(specs):
    new_listings.append(make_listing(101 + i, ptype, loc, a, b, r, i))

# ── Patch existing 100 with image fields ─────────────────────────────────────

with open("data/listings.json") as f:
    existing = json.load(f)

for p in existing:
    p["image_url"] = primary_image(p["id"], p["type"])
    p["images"]    = property_images(p["id"])

# ── Write combined listings.json ─────────────────────────────────────────────

all_listings = existing + new_listings
with open("data/listings.json", "w") as f:
    json.dump(all_listings, f, indent=2)
print(f"listings.json → {len(all_listings)} properties")

# ── Write property_images.json ───────────────────────────────────────────────

with open("data/property_images.json", "w") as f:
    json.dump({
        "total": len(ALL_IMAGES),
        "by_type": {k: v for k, v in TYPE_PRIMARY.items()},
        "all": ALL_IMAGES,
    }, f, indent=2)
print(f"property_images.json → {len(ALL_IMAGES)} images")
