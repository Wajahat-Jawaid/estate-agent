"""Replay harness — drives a scripted conversation through the live agent and
grades each in-scope turn against the expected reply.

  • Structured facts (HARD): listings returned, all prices within budget, stage.
  • Semantic (LLM judge): does the actual reply convey the same meaning/intent
    as the expected one? Verdict MATCH / PARTIAL / MISMATCH + a one-line reason.

Run:  venv/bin/python tests/replay.py            # conversation #1
      venv/bin/python tests/replay.py tests/conv01.py
"""
import importlib.util
import json
import re
import sys
import uuid
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import get_response, price_to_lacs  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# Dedicated judge at temperature 0 so verdicts are stable run-to-run (the app's
# own llm runs at 0.7, which made the scoreboard too noisy to compare).
judge_llm = ChatOpenAI(model="gpt-5.4-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

GREEN, RED, YEL, GRY, RST = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"


def load_conversations(path):
    spec = importlib.util.spec_from_file_location("conv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "CONVERSATIONS"):
        return list(mod.CONVERSATIONS)
    return [mod.CONVERSATION]


def structured_facts(result, filters):
    """Hard checks that don't depend on prose."""
    facts = {"stage": result.get("stage"), "listings": len(result.get("listings", []))}
    ceil = None
    max_price = (filters or {}).get("max_price")
    if max_price:
        ceil = int(price_to_lacs(str(max_price)) * 1.1)
    over = []
    if ceil:
        for l in result.get("listings", []):
            p = int(l["metadata"].get("price_numeric") or 0)
            if p > ceil:
                over.append(p)
    facts["budget_ceiling_lacs"] = ceil
    facts["over_budget_count"] = len(over)
    return facts


def judge(customer, expected, actual, facts):
    prompt = f"""You are grading a real-estate chatbot against a gold-standard human-agent reply.
The bot is a DISCOVERY + SEARCH assistant; exact wording and the exact order of discovery
questions will differ. Grade on MEANING and INTENT, not phrasing or order.

Customer said: "{customer}"
Gold-standard agent reply: "{expected}"
Bot's actual reply: "{actual}"
Structured facts about the bot's turn: {json.dumps(facts)}

Does the bot's reply serve the SAME purpose as the gold reply (e.g. both ask about the same
kind of information, or both give the same kind of recommendation/result)? Over-budget
listings (over_budget_count > 0) is an automatic MISMATCH.

Return ONLY JSON: {{"verdict": "MATCH|PARTIAL|MISMATCH", "reason": "<one line>"}}"""
    raw = judge_llm.invoke([SystemMessage(content="You are a precise grader. Output only JSON."),
                            HumanMessage(content=prompt)]).content.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "PARTIAL", "reason": f"judge parse failed: {raw[:80]}"}


def run(conv):
    uid = f"{conv['user_id']}_{uuid.uuid4().hex[:6]}"
    print(f"\n{'='*70}\nREPLAY: {conv['name']}  (user_id={uid})\n{'='*70}")
    tally = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0, "skipped": 0}
    for i, turn in enumerate(conv["turns"], 1):
        result = get_response(turn["customer"], user_id=uid, channel="web")
        actual = result.get("response", "")
        print(f"\n{'-'*70}\nT{i}  C: {turn['customer'][:80]}")
        if turn.get("scope") != "in":
            tally["skipped"] += 1
            print(f"{GRY}   [skipped — out of scope: {turn.get('reason','')}]{RST}")
            print(f"{GRY}   BOT: {actual[:120]}{RST}")
            continue
        facts = structured_facts(result, result.get("accumulated_filters"))
        v = judge(turn["customer"], turn["expected_agent"], actual, facts)
        verdict = v.get("verdict", "PARTIAL").upper()
        tally[verdict] = tally.get(verdict, 0) + 1
        color = {"MATCH": GREEN, "PARTIAL": YEL, "MISMATCH": RED}.get(verdict, YEL)
        print(f"   exp: {turn['expected_agent'][:90]}")
        print(f"   BOT: {actual[:90]}")
        print(f"   facts: {facts}")
        print(f"   {color}{verdict}{RST} — {v.get('reason','')}")
    print(f"\n{'='*70}\nSUMMARY: {GREEN}{tally['MATCH']} match{RST}, "
          f"{YEL}{tally['PARTIAL']} partial{RST}, {RED}{tally['MISMATCH']} mismatch{RST}, "
          f"{tally['skipped']} skipped (out of scope)\n{'='*70}")
    return tally


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "conversations.py")
    convs = load_conversations(path)
    grand = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0, "skipped": 0}
    per_conv = []
    for c in convs:
        t = run(c)
        per_conv.append((c["name"], t))
        for k in grand:
            grand[k] += t.get(k, 0)
    print(f"\n\n{'#'*70}\nBATCH SCOREBOARD\n{'#'*70}")
    for name, t in per_conv:
        graded = t["MATCH"] + t["PARTIAL"] + t["MISMATCH"]
        print(f"  {name[:48]:48s}  {GREEN}{t['MATCH']}M{RST} {YEL}{t['PARTIAL']}P{RST} {RED}{t['MISMATCH']}X{RST}  (/{graded} graded)")
    g = grand["MATCH"] + grand["PARTIAL"] + grand["MISMATCH"]
    print(f"\n  TOTAL: {GREEN}{grand['MATCH']} match{RST}, {YEL}{grand['PARTIAL']} partial{RST}, "
          f"{RED}{grand['MISMATCH']} mismatch{RST} out of {g} graded turns; {grand['skipped']} skipped.")
