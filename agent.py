import os
import json
import re
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

memory = ConversationBufferMemory(return_messages=True)

# Full search history — never overwritten, only appended
search_history = []

def price_to_lacs(price_str: str) -> int:
    price_str = price_str.lower().strip()
    crore, lac = 0, 0
    crore_match = re.search(r'(\d+)\s*crore', price_str)
    lac_match = re.search(r'(\d+)\s*lac', price_str)
    if crore_match:
        crore = int(crore_match.group(1))
    if lac_match:
        lac = int(lac_match.group(1))
    return (crore * 100) + lac

def extract_filters(query: str) -> dict:
    prompt = f"""
Extract property search filters from this query as JSON. Only include filters explicitly mentioned.

Query: "{query}"

Return ONLY a valid JSON object with these possible keys (omit any not mentioned):
- location (string)
- type (string: house/apartment/upper portion/lower portion/penthouse/farmhouse)
- bedrooms (integer)
- max_price (string, in Pakistani format e.g. "2 crore 50 lac")
- features (array of strings)

Return only raw JSON. No explanation. No markdown. No backticks.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return json.loads(response.content.strip())
    except:
        return {}

def classify_query(query: str) -> dict:
    if not search_history:
        return {"type": "NEWSEARCH", "index": None}

    history_summary = ""
    for i, entry in enumerate(search_history):
        history_summary += f"Search {i}: {entry['topic']}\n"

    prompt = f"""
You are helping a real estate chatbot classify a user query.

Previous searches in this session:
{history_summary}

Current user query: "{query}"

RULES — read carefully:

NEWSEARCH if the query:
- Mentions a NEW location not in any previous search
- Asks for a different property type not previously searched
- Changes the budget significantly
- Uses phrases like "what about X", "do you have in X", "show me X", "any in X"
- Is asking to explore a new area or category

FOLLOWUP if the query:
- Asks to compare, rank, or pick from already shown properties
- Asks about details of already shown properties (area, agent, street, features)
- Uses words like "which one", "the biggest", "the cheapest", "among these", "from those"
- Asks about a specific property ID or title already shown
- Refers to "those properties", "the ones you showed", "from the results"

CRITICAL: If the query mentions ANY new location, area, or neighbourhood — it is ALWAYS a NEWSEARCH regardless of phrasing.

If FOLLOWUP, which search index is being referred to?

Return ONLY valid JSON:
{{"type": "FOLLOWUP", "index": 0}}
or
{{"type": "NEWSEARCH", "index": null}}

No explanation. No markdown. No backticks.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(response.content.strip())
        print(f">> Raw classification: {result}")
        return result
    except:
        return {"type": "NEWSEARCH", "index": None}

def build_chroma_filter(filters: dict):
    conditions = []

    if "bedrooms" in filters:
        conditions.append({"bedrooms": {"$eq": filters["bedrooms"]}})

    if "type" in filters:
        conditions.append({"type": {"$eq": filters["type"]}})

    if "max_price" in filters:
        max_lacs = price_to_lacs(filters["max_price"])
        if max_lacs > 0:
            conditions.append({"price_numeric": {"$lte": max_lacs}})

    # Location intentionally excluded — handled by soft filter in search_properties

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

def search_properties(query: str, filters: dict, k: int = 10) -> list:
    chroma_filter = build_chroma_filter(filters)

    # Fetch more than needed to account for location filtering
    fetch_k = 20 if "location" in filters else k

    try:
        if chroma_filter:
            results = vectorstore.similarity_search_with_score(
                query, k=fetch_k, filter=chroma_filter
            )
        else:
            results = vectorstore.similarity_search_with_score(
                query, k=fetch_k
            )
    except Exception as e:
        print(f"Filter failed, falling back: {e}")
        results = vectorstore.similarity_search_with_score(query, k=fetch_k)

    # Apply soft location filter in Python if location was specified
    if "location" in filters and filters["location"]:
        location_keyword = filters["location"].lower()
        filtered = [
            (doc, score) for doc, score in results
            if location_keyword in doc.metadata.get("location", "").lower()
        ]
        # Only apply filter if it actually found something
        # If nothing matches, fall back to unfiltered results
        if filtered:
            results = filtered

    # Return top k after filtering
    return results[:k]

def get_response(user_query: str) -> dict:
    global search_history

    # ── Small talk check — skip ChromaDB entirely ──
    small_talk_prompt = f"""Is this message small talk, a greeting, or general conversation unrelated to property search?
Message: "{user_query}"
Reply with only: SMALLTALK or PROPERTY"""

    st_check = llm.invoke([HumanMessage(content=small_talk_prompt)])
    if st_check.content.strip().upper() == "SMALLTALK":
        response = llm.invoke([
            SystemMessage(content="You are a friendly real estate assistant. Respond warmly to greetings and small talk, then ask what property the user is looking for. Be brief. Respond in the same language as the user."),
            HumanMessage(content=user_query)
        ])
        ai_response = response.content
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(ai_response)
        return {
            "response": ai_response,
            "listings": [],
            "filters": {}
        }

    # Conversation history
    history = memory.chat_memory.messages
    history_text = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # Classify query
    classification = classify_query(user_query)
    print(f">> Classification: {classification}")

    if classification["type"] == "FOLLOWUP" and classification["index"] is not None:
        # Retrieve the specific previous search being referred to
        index = classification["index"]
        if index < len(search_history):
            matched_listings = search_history[index]["listings"]
            context = search_history[index]["context"]
            filters = {}
            print(f">> Follow up on search {index}: '{search_history[index]['topic']}'")
        else:
            # Index out of range safety fallback
            matched_listings = search_history[-1]["listings"]
            context = search_history[-1]["context"]
            filters = {}
    else:
        # New search
        filters = extract_filters(user_query)
        results = search_properties(user_query, filters)

        context = ""
        matched_listings = []
        raw_scores = []

        for doc, score in results:
            context += f"\n---\n{doc.page_content}\n"
            raw_scores.append((doc, score))

        if raw_scores:
            best_distance = raw_scores[0][1]
            worst_distance = raw_scores[-1][1]
            score_range = worst_distance - best_distance

            for doc, score in raw_scores:
                if score_range == 0:
                    normalized = 95
                else:
                    # Best gets 95, worst gets 72, others scale in between
                    normalized = 95 - ((score - best_distance) / score_range) * 23
                matched_listings.append({
                    "metadata": doc.metadata,
                    "score": round(normalized)
                })

        # Build a readable topic label for this search
        topic_parts = []
        if "location" in filters:
            topic_parts.append(filters["location"])
        if "type" in filters:
            topic_parts.append(filters["type"])
        if "bedrooms" in filters:
            topic_parts.append(f"{filters['bedrooms']} bed")
        if "max_price" in filters:
            topic_parts.append(f"under {filters['max_price']}")
        topic = ", ".join(topic_parts) if topic_parts else user_query[:60]

        search_history.append({
            "topic": topic,
            "listings": matched_listings,
            "context": context
        })
        print(f">> New search saved as: '{topic}'")

    system_prompt = """You are a helpful real estate assistant for a Karachi property search platform.
Be conversational, concise and helpful.

If the user is making small talk (greetings, asking how you are, general conversation) — respond warmly and briefly, then gently ask what property they are looking for. Do NOT show property listings for small talk.

Only search and show properties when the user is clearly asking about real estate.
Answer strictly from the provided property listings — do not make up information.
If the user asks about streets, surroundings, or neighbourhood details not in the listing data,
honestly say that specific detail isn't in the listing but suggest they ask the agent directly.
Respond in the same language the user is using (Urdu or English)."""

    no_results = len(matched_listings) == 0

    user_prompt = f"""
    Conversation history:
    {history_text}

    User query: {user_query}

    {"No matching properties found in the database." if no_results else f"Available properties:{context}"}

    {"Politely tell the user no properties matched their criteria. Suggest they broaden their search — different area, higher budget, or different property type. Be helpful and specific with suggestions based on their query." if no_results else "Answer the user's question based strictly on the available properties above. If comparing or selecting, reason through the properties and give a clear answer. Mention agent name and contact where relevant."}
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    ai_response = response.content
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(ai_response)

    return {
        "response": ai_response,
        "listings": matched_listings,
        "filters": filters
    }


# Test
if __name__ == "__main__":
    print("Test 1 — DHA search")
    r1 = get_response("I want a 3 bed house in DHA Karachi under 2 crore 50 lac with parking")
    print(r1["response"])
    for l in r1["listings"]:
        print(f"  - {l['metadata']['title']} | {l['metadata']['price']}")

    print("\n" + "="*50)
    print("Test 2 — Clifton search")
    r2 = get_response("show me apartments in Clifton with sea view")
    print(r2["response"])
    for l in r2["listings"]:
        print(f"  - {l['metadata']['title']} | {l['metadata']['price']}")

    print("\n" + "="*50)
    print("Test 3 — Nazimabad search")
    r3 = get_response("what options do you have in North Nazimabad")
    print(r3["response"])

    print("\n" + "="*50)
    print("Test 4 — follow up on DHA specifically")
    r4 = get_response("btw what's the street situation in those DHA properties?")
    print(r4["response"])
    print("Listings used:", [l['metadata']['title'] for l in r4["listings"]])

    print("\n" + "="*50)
    print("Test 5 — follow up on Clifton specifically")
    r5 = get_response("which of the Clifton apartments has the best sea view?")
    print(r5["response"])
    print("Listings used:", [l['metadata']['title'] for l in r5["listings"]])