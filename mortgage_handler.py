"""Mortgage conversation handler — slot filling and intent routing."""

from __future__ import annotations
import re
from mortgage_engine import (
    MortgageEngine, format_mortgage_result, format_reverse_result, format_pkr,
    BANKS, DEFAULT_BANK, DEFAULT_DOWN_PCT, DEFAULT_TENURE_YEARS,
)

# Per-user mortgage conversation state (keyed by user_id)
user_mortgage_states: dict = {}


def get_mortgage_state(user_id: str) -> dict:
    if user_id not in user_mortgage_states:
        user_mortgage_states[user_id] = {
            "active": False,
            "property_price": None,       # rupees
            "down_payment_pct": DEFAULT_DOWN_PCT,
            "tenure_years": DEFAULT_TENURE_YEARS,
            "bank": DEFAULT_BANK,
        }
    return user_mortgage_states[user_id]


def reset_mortgage_state(user_id: str):
    user_mortgage_states.pop(user_id, None)


# ── Slot extractors ──────────────────────────────────────────────────────────

def extract_price_rupees(text: str) -> float | None:
    """Extract a PKR amount from text; returns rupees."""
    t = text.lower()
    crore_m = re.search(r"(\d+(?:\.\d+)?)\s*crore", t)
    lakh_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", t)
    crore_val = float(crore_m.group(1)) if crore_m else 0.0
    lakh_val = float(lakh_m.group(1)) if lakh_m else 0.0
    if crore_val or lakh_val:
        return (crore_val * 100 + lakh_val) * 100_000
    # Plain large number (≥ 10,000 treated as raw rupees)
    m = re.search(r"(?<![.\d])(\d[\d,]{4,})(?!\d)", text)
    if m:
        n = float(m.group(1).replace(",", ""))
        if n >= 10_000:
            return n
    return None


def _extract_percentage(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|pc)\b", text.lower())
    if m:
        return float(m.group(1))
    return None


def _extract_tenure(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:year|yr|sal|saal)", text.lower())
    if m:
        return int(m.group(1))
    # bare number 1–30 (only if no lakh/crore in text, to avoid false matches)
    if not re.search(r"(?:lakh|lac|crore)", text.lower()):
        m = re.search(r"\b(\d+)\b", text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 30:
                return n
    return None


def _extract_bank(text: str) -> str | None:
    t = text.lower()
    if "meezan" in t or "islamic" in t:
        return "Meezan"
    if "hbfc" in t or "ghar sahulat" in t:
        return "HBFC"
    if "hbl" in t:
        return "HBL"
    return None


def _fill_slots_from_message(state: dict, query: str):
    """Try to fill any slots from the user's current message."""
    if state["property_price"] is None:
        price = extract_price_rupees(query)
        if price:
            state["property_price"] = price
    pct = _extract_percentage(query)
    if pct is not None and 5 <= pct <= 50:
        state["down_payment_pct"] = pct
    tenure = _extract_tenure(query)
    if tenure is not None:
        state["tenure_years"] = tenure
    bank = _extract_bank(query)
    if bank:
        state["bank"] = bank


def _build_result(state: dict) -> dict:
    """Calculate and return a mortgage result response dict."""
    calc = MortgageEngine.calculate(
        price=state["property_price"],
        down_pct=state["down_payment_pct"],
        tenure_years=state["tenure_years"],
        bank=state["bank"],
    )
    response = format_mortgage_result(calc)
    max_lacs = int(state["property_price"] / 100_000)
    return {
        "response": response,
        "listings": [],
        "filters": {},
        "follow_up": "Want me to show properties within your budget?",
        "actions": [{"id": "within_budget", "label": "🏠 Show properties"}],
        "meta": {
            "no_results": False,
            "action": "mortgage_result",
            "mortgage_result": calc,
            "max_price_lacs": max_lacs,
        },
    }


# ── Public handler class ─────────────────────────────────────────────────────

class MortgageConversationHandler:

    def handle_slot_filling(self, user_id: str, query: str) -> dict | None:
        """Continue slot filling if user is mid-mortgage-conversation. Returns None if not active."""
        state = get_mortgage_state(user_id)
        if not state.get("active"):
            return None

        _fill_slots_from_message(state, query)

        if state["property_price"] is None:
            return {
                "response": "What's the property price you want to calculate for? (e.g. 1.5 crore or 80 lakh)",
                "listings": [],
                "filters": {},
                "follow_up": None,
                "actions": [],
                "meta": {"no_results": False, "action": "mortgage_slot"},
            }

        result = _build_result(state)
        reset_mortgage_state(user_id)
        return result

    def handle_mortgage_explicit(self, user_id: str, query: str, context_price_rupees: float | None = None) -> dict:
        """Handle direct mortgage/EMI query with slot filling."""
        state = get_mortgage_state(user_id)
        state["active"] = True

        _fill_slots_from_message(state, query)

        # Pre-fill from property in context if price still missing
        if state["property_price"] is None and context_price_rupees:
            state["property_price"] = context_price_rupees

        if state["property_price"] is None:
            return {
                "response": "Sure! What's the property price you want to calculate EMI for? (e.g. 1.5 crore or 80 lakh)",
                "listings": [],
                "filters": {},
                "follow_up": None,
                "actions": [],
                "meta": {"no_results": False, "action": "mortgage_slot"},
            }

        result = _build_result(state)
        reset_mortgage_state(user_id)
        return result

    def handle_affordability_hint(self, user_id: str, query: str) -> dict:
        """Soft offer to calculate EMI — triggered when user hints at price concern."""
        return {
            "response": "Want me to calculate the monthly installment on this property?",
            "listings": [],
            "filters": {},
            "follow_up": None,
            "actions": [{"id": "calculate_emi", "label": "💰 Calculate EMI"}],
            "meta": {"no_results": False, "action": "affordability_hint"},
        }

    def handle_reverse_mortgage(self, user_id: str, query: str) -> dict:
        """Extract monthly budget, calculate max property price, trigger search."""
        monthly_budget = extract_price_rupees(query)

        if not monthly_budget:
            return {
                "response": "What's your monthly budget? (e.g. 1 lakh per month or 80,000 per month)",
                "listings": [],
                "filters": {},
                "follow_up": None,
                "actions": [],
                "meta": {"no_results": False, "action": "mortgage_slot"},
            }

        bank = _extract_bank(query) or DEFAULT_BANK
        max_price = MortgageEngine.reverse_calculate(monthly_budget, DEFAULT_DOWN_PCT, DEFAULT_TENURE_YEARS, bank)
        max_lacs = int(max_price / 100_000)

        response = format_reverse_result(monthly_budget, max_price, DEFAULT_DOWN_PCT, DEFAULT_TENURE_YEARS, bank)

        return {
            "response": response,
            "listings": [],
            "filters": {"max_price": f"{max_lacs} lakh"},
            "follow_up": None,
            "actions": [],
            "meta": {
                "no_results": False,
                "action": "reverse_mortgage",
                "trigger_search": True,
                "max_price_lacs": max_lacs,
            },
        }
