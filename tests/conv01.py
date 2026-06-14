"""Gold-standard conversation #1 — family buyer with kids, mid-budget.

Each turn:
  customer        – the message we feed the bot
  expected_agent  – the human-agent reply we semantically grade against
  scope           – "in"  : graded (within the bot's purpose: discovery + reveal)
                    "out" : skipped (scheduling / negotiation — agreed out of scope)
  reason          – why a turn is out of scope
"""

CONVERSATION = {
    "name": "Family buyer with kids, mid-budget, Karachi",
    "user_id": "replay_conv01",
    "turns": [
        {"customer": "Assalam o Alaikum. I'm looking for a flat in Karachi for my family, not for investment or rent. We want to live there ourselves.",
         "expected_agent": "Wa Alaikum Assalam. Sure. Let me understand your need properly. How many family members?",
         "scope": "in"},
        {"customer": "Husband, wife, two kids, and sometimes my mother stays with us.",
         "expected_agent": "Then a 3-bed apartment would be suitable. Are you only considering apartments, or portions too?",
         "scope": "in"},
        {"customer": "Apartment is preferred. Portion only if the area and building are really good.",
         "expected_agent": "Got it. What budget should I keep in mind?",
         "scope": "in"},
        {"customer": "Around 2.2 to 2.6 crore. I can stretch slightly only if the option is really worth it.",
         "expected_agent": "Good. In this budget, practical family areas would be Gulshan-e-Iqbal, Gulistan-e-Johar, Scheme 33, FB Area, and some parts of North Nazimabad. DHA and Clifton may be difficult for a proper 3-bed in this range.",
         "scope": "in"},
        {"customer": "I'm currently near Gulshan, so Gulshan, Johar, Scheme 33, and maybe FB Area are okay.",
         "expected_agent": "Understood. For family living, I'll focus on safe surroundings, proper parking, lift, backup, water, building maintenance, and nearby schools/markets.",
         "scope": "in"},
        {"customer": "Exactly. I don't want daily issues with lift, water, or parking. Also dedicated parking is important.",
         "expected_agent": "Noted. Middle floor preferred?",
         "scope": "in"},
        {"customer": "Yes. Not ground, not top floor. My mother visits often, so lift is must.",
         "expected_agent": "Perfect. So your main requirements are: 3 bedrooms, family building, lift, parking, middle floor, good maintenance, budget up to 2.6 crore, and priority areas are Gulshan and Johar, with Scheme 33 or FB Area only if the option is strong.",
         "scope": "in"},
        {"customer": "Yes, exactly.",
         "expected_agent": "Any extra preferences like drawing room, west open, servant quarter, or generator?",
         "scope": "in"},
        {"customer": "Drawing room would be nice. West open is a plus. Generator backup for lift is important. Servant quarter is not necessary.",
         "expected_agent": "Makes sense. I'd suggest we shortlist 4 to 5 serious options only. Maybe 2 in Gulshan, 1 or 2 in Johar, and 1 in Scheme 33 if it's well-connected.",
         "scope": "in"},
        {"customer": "That's better. I don't want to waste time seeing random flats.",
         "expected_agent": "Honestly, for your case, I'd prioritize Gulshan first because your routine and kids' school are already nearby. Johar would be second because it may offer better space in budget. Scheme 33 only if the project is not isolated.",
         "scope": "in"},
        {"customer": "That makes sense. Daily life convenience matters more than just getting a bigger flat.",
         "expected_agent": "Exactly. Expected pricing would be around 2.5 to 2.7 crore in Gulshan, and around 2.3 to 2.5 crore in Johar.",
         "scope": "in"},
        {"customer": "If Gulshan goes above 2.6, it may be difficult.",
         "expected_agent": "Then I'll keep 2.6 as the comfortable upper limit and only show higher options if they are genuinely worth it and negotiable.",
         "scope": "out", "reason": "negotiation / budget-ceiling strategy"},
        {"customer": "Good. My wife will also come for the visit.",
         "expected_agent": "That's better. Since it's for living, both of you should see the flat, building, parking, lift, ventilation, and surroundings.",
         "scope": "out", "reason": "physical visit"},
        {"customer": "Saturday after 4 pm works for us.",
         "expected_agent": "Great. We'll plan Gulshan first, then Johar, and only add Scheme 33 if there's a strong option.",
         "scope": "out", "reason": "scheduling"},
        {"customer": "Okay, let's do Saturday 4 pm.",
         "expected_agent": "Done. We'll meet Saturday at 4 pm near Gulshan and visit the shortlisted properties from there.",
         "scope": "out", "reason": "scheduling"},
    ],
}
