import chromadb
import json

IMAGE_BANK = {
    "house": [
        "https://images.pexels.com/photos/19516616/pexels-photo-19516616.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/30580640/pexels-photo-30580640.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/323780/pexels-photo-323780.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/6342356/pexels-photo-6342356.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7031581/pexels-photo-7031581.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/13729358/pexels-photo-13729358.png?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/33162457/pexels-photo-33162457.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/27953061/pexels-photo-27953061.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/323781/pexels-photo-323781.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7031406/pexels-photo-7031406.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/8482510/pexels-photo-8482510.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7031604/pexels-photo-7031604.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/358636/pexels-photo-358636.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/16804979/pexels-photo-16804979.png?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/32115995/pexels-photo-32115995.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/24524484/pexels-photo-24524484.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7746940/pexels-photo-7746940.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/3958954/pexels-photo-3958954.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7031405/pexels-photo-7031405.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ],
    "apartment": [
        "https://images.pexels.com/photos/33244441/pexels-photo-33244441.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/16110999/pexels-photo-16110999.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/15951714/pexels-photo-15951714.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/9308434/pexels-photo-9308434.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/11631278/pexels-photo-11631278.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/21071043/pexels-photo-21071043.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/5674684/pexels-photo-5674684.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/31656173/pexels-photo-31656173.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/562199/pexels-photo-562199.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/9170385/pexels-photo-9170385.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/14998334/pexels-photo-14998334.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/26690862/pexels-photo-26690862.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/31640021/pexels-photo-31640021.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/36079488/pexels-photo-36079488.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/31656143/pexels-photo-31656143.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ],
    "upper_portion": [
        "https://images.pexels.com/photos/16240077/pexels-photo-16240077.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/18024491/pexels-photo-18024491.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/27822509/pexels-photo-27822509.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/12923395/pexels-photo-12923395.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/27822512/pexels-photo-27822512.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ],
    "lower_portion": [
        "https://images.pexels.com/photos/36279748/pexels-photo-36279748.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/5065808/pexels-photo-5065808.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/37488703/pexels-photo-37488703.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/34940630/pexels-photo-34940630.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/33095097/pexels-photo-33095097.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ],
    "penthouse": [
        "https://images.pexels.com/photos/36362/pexels-photo.jpg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/7045918/pexels-photo-7045918.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/27273422/pexels-photo-27273422.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ],
    "farmhouse": [
        "https://images.pexels.com/photos/11496921/pexels-photo-11496921.jpeg?auto=compress&cs=tinysrgb&h=650&w=940",
        "https://images.pexels.com/photos/3466361/pexels-photo-3466361.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
    ]
}

FALLBACK_IMAGES = IMAGE_BANK["house"]

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("langchain")

result = collection.get(include=["metadatas"])
ids = result["ids"]
metadatas = result["metadatas"]

total = 0
for doc_id, metadata in zip(ids, metadatas):
    property_type = metadata.get("type", "").lower().replace(" ", "_")
    bank = IMAGE_BANK.get(property_type, FALLBACK_IMAGES)
    start = hash(doc_id) % len(bank)
    # Pick 4 distinct images cycling through the bank
    selected = [bank[(start + i) % len(bank)] for i in range(min(4, len(bank)))]
    image_url = selected[0]

    updated_metadata = {**metadata, "image_url": image_url, "images": json.dumps(selected)}
    collection.update(ids=[doc_id], metadatas=[updated_metadata])

    print(f"Updated {doc_id} ({property_type}) → {image_url}")
    total += 1

print(f"\nDone! Total updated: {total}")
