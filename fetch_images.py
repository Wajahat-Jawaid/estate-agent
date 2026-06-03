import requests
import json

API_KEY = "8DEFKv3K9mISH7Lrlh8Yfy6td7Txmu1UamiPyRWgUqtc7ET3KGS6dL1L"

CATEGORIES = {
    "house": {"query": "house exterior modern", "count": 20},
    "apartment": {"query": "modern apartment building", "count": 15},
    "upper_portion": {"query": "residential house pakistan", "count": 5},
    "lower_portion": {"query": "residential building exterior", "count": 5},
    "penthouse": {"query": "luxury penthouse rooftop", "count": 3},
    "farmhouse": {"query": "farmhouse countryside", "count": 2},
}

def fetch_images(query, count):
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_KEY}
    params = {"query": query, "per_page": count, "orientation": "landscape"}
    response = requests.get(url, headers=headers, params=params)
    print(f"  Status: {response.status_code}")
    data = response.json()
    return [photo["src"]["large"] for photo in data.get("photos", [])]

all_images = {}

for category, config in CATEGORIES.items():
    print(f"Fetching {config['count']} images for: {category}")
    urls = fetch_images(config["query"], config["count"])
    all_images[category] = urls
    print(f"  Got {len(urls)} URLs")

with open("property_images.json", "w") as f:
    json.dump(all_images, f, indent=2)

print("\nDone! Saved to property_images.json")
