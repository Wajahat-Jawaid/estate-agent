"""Quality-test batch — scenario conversations mirroring docs/agent-quality-tests.md.

Graded the same way as the other replay files: in-scope turns are semantically
judged against an intent-level gold reply; physical scheduling/visit turns are
skipped as out of scope.

Run:  venv/bin/python tests/replay.py tests/conv_quality.py
"""

def _c(customer, agent, scope="in", reason=""):
    return {"customer": customer, "expected_agent": agent, "scope": scope, "reason": reason}

SCHED = "scheduling / physical visit"

CONVERSATIONS = [
    {
        "name": "1. Full journey — family buyer, accessibility, objection",
        "user_id": "q_journey",
        "turns": [
            _c("I'm looking for a 3 bed apartment for my family",
               "Sure. To guide you well — what budget should I keep in mind, or which area suits you?"),
            _c("Around 2.5 crore",
               "Got it. Which areas work for you?"),
            _c("DHA or Clifton, whichever has better options",
               "Noted. A proper 3-bed in DHA/Clifton near 2.5 crore is tight, so I'll focus on the strongest options and set expectations honestly."),
            _c("We have my mother living with us, she can't do stairs",
               "Then lift and reliable backup are must-haves. I'll only consider buildings with a working lift."),
            _c("Show me what you have",
               "Here are family 3-bed options with lift; these are the closest matches within your budget."),
            _c("Which of these has the best building and lift situation?",
               "Comparing the shortlist, this one has the best building management and lift/backup for your mother."),
            _c("This seems a bit expensive",
               "I understand. The price reflects the area and building quality — I can pull cheaper options with clear trade-offs if you'd like."),
            _c("My wife and I can visit Friday after school",
               "Great, you're ready to view. Let's lock a time and I'll arrange the visit.", "out", SCHED),
        ],
    },
    {
        "name": "2. Unrealistic demand — expectation reset",
        "user_id": "q_unreal",
        "turns": [
            _c("I want a 3 bed apartment in DHA under 1 crore",
               "Honestly, a 3-bed in DHA under 1 crore isn't realistic. We could raise the budget, drop to fewer beds, or look at more affordable areas — which works for you?"),
            _c("Okay what would you suggest instead?",
               "For a 3-bed near this budget, areas like Gulshan, Johar, or North Karachi are far more realistic than DHA."),
            _c("Let's expand the area then",
               "Good. I'll focus on 3-bed family options in those practical areas within your budget."),
        ],
    },
    {
        "name": "3. Low-budget trade-off question",
        "user_id": "q_lowbudget",
        "turns": [
            _c("I need a good 3 bed flat but my budget is limited",
               "Understood. What budget should I keep in mind?"),
            _c("Around 1.1 crore",
               "At 1.1 crore a 3-bed is tight. Which compromise is okay: keep 3-bed but an older/weaker building, a stronger 2-bed, a farther area, or a slightly higher budget?"),
            _c("I really want 3 bedrooms though",
               "Then I'll keep 3-bed strict and focus on older buildings or less prime blocks that fit 1.1 crore."),
        ],
    },
    {
        "name": "4. Investor — yield language, no ROI promise",
        "user_id": "q_investor",
        "turns": [
            _c("I want to buy something mainly for rental income",
               "Sure. What's your investment budget?"),
            _c("I have about 1 crore to invest",
               "For 1 crore aimed at rental income, here are area tiers by estimated yield — these are estimates, not guarantees."),
            _c("Which area gives the best return?",
               "The higher-yield tier gives the best monthly cash flow, with the trade-off of weaker resale vs premium areas."),
        ],
    },
    {
        "name": "5. Mortgage / affordability",
        "user_id": "q_mortgage",
        "turns": [
            _c("My monthly budget is around 2 lakh, what can I get?",
               "With ~2 lakh/month you can support a certain price range — here's roughly what that affords and some matching options."),
            _c("What would the EMI be on a 2 crore property?",
               "For a 2 crore property, here's an approximate EMI based on a typical down payment, rate, and tenure."),
        ],
    },
    {
        "name": "6. Seller intent — flow switch",
        "user_id": "q_seller",
        "turns": [
            _c("I actually want to sell my flat",
               "Sure, I can help you list it. Where is it, and what's the size/bedrooms and condition?"),
            _c("2 bed in Gulshan, about 1200 sq ft, decent condition",
               "Noted. What price are you expecting? I'll help with valuation and connect you to list it."),
        ],
    },
    {
        "name": "7. Area confusion (Urdu)",
        "user_id": "q_urdu",
        "turns": [
            _c("Gulshan aur Johar mein se kya behtar hai family ke liye?",
               "Dono family ke liye theek hain — Gulshan zyada central hai, Johar mein budget mein zyada space milti hai. Aap ke liye commute important hai ya naya building?"),
            _c("Newer building zyada important hai",
               "Phir Johar ya Scheme 33 ki newer buildings behtar rahengi. Budget kya rakhun?"),
        ],
    },
    {
        "name": "8. Just browsing — no listing dump",
        "user_id": "q_browse",
        "turns": [
            _c("Show me some flats",
               "Happy to — to point you right, what's your budget, area, or how many bedrooms?"),
        ],
    },
    {
        "name": "9. Follow-up actions — cheaper / larger / refine",
        "user_id": "q_followup",
        "turns": [
            _c("3 bed apartment in Gulshan around 2.5 crore",
               "Here are 3-bed options in Gulshan within 2.5 crore."),
            _c("Show me cheaper ones",
               "Here are more affordable 3-bed options in Gulshan below that price."),
            _c("Actually, bigger ones instead",
               "Here are larger options with more space/rooms in Gulshan."),
            _c("2 bed would be better, within the same budget",
               "Switching to 2-bed in Gulshan within the same 2.5 crore budget — here they are."),
        ],
    },
]
