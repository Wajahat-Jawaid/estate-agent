import json
import random

AMENITIES = [
    # Main Features
    "Central Air Conditioning",
    "Electricity Backup",
    "Double Glazed Windows",
    "Central Heating",
    "Flooring",
    "Parking Spaces",
    # Rooms
    "Drawing Room",
    "Study Room",
    "Store Room",
    "Servant Quarters",
    "Dining Room",
    "Prayer Room",
    "Lounge or Sitting Room",
    "Powder Room",
    # Business and Communication
    "Broadband Internet Access",
    "Satellite or Cable TV Ready",
    "Intercom",
    # Community Features
    "Community Lawn or Garden",
    "Community Gym",
    "First Aid or Medical Centre",
    "Day Care Centre",
    "Kids Play Area",
    "Barbeque Area",
    "Mosque",
    "Community Centre",
    # Nearby Locations and Other Facilities
    "Nearby Schools",
    "Nearby Hospitals",
    "Nearby Shopping Malls",
    "Nearby Restaurants",
    "Nearby Public Transport Service",
    # Other Facilities
    "Maintenance Staff",
    "Security Staff",
]

random.seed(42)

with open("data/listings.json") as f:
    listings = json.load(f)

for listing in listings:
    listing.pop("features", None)
    count = random.randint(4, 9)
    listing["amenities"] = sorted(random.sample(AMENITIES, count))

with open("data/listings.json", "w") as f:
    json.dump(listings, f, indent=2)

print(f"Updated {len(listings)} listings — features removed, amenities randomised.")
