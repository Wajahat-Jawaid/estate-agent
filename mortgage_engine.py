"""Mortgage calculation engine for Pakistani real estate."""

KIBOR = 19.0  # hardcoded percent per annum — disclaimer shown to user

BANKS = {
    "HBFC": {"name": "HBFC Ghar Sahulat", "spread": 3.0},
    "Meezan": {"name": "Meezan Islamic", "spread": 2.5},
    "HBL": {"name": "HBL", "spread": 3.5},
}

DEFAULT_BANK = "HBFC"
DEFAULT_DOWN_PCT = 20
DEFAULT_TENURE_YEARS = 20


def format_pkr(amount_rupees: float) -> str:
    """Format PKR amount in Lakh/Crore notation."""
    lakh = amount_rupees / 100_000
    if lakh >= 100:
        crore = lakh / 100
        if crore == int(crore):
            return f"{int(crore)} crore"
        formatted = f"{crore:.2f}".rstrip("0").rstrip(".")
        return f"{formatted} crore"
    if lakh == int(lakh):
        return f"{int(lakh)} lakh"
    formatted = f"{lakh:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} lakh"


class MortgageEngine:

    @staticmethod
    def annual_rate(bank: str) -> float:
        config = BANKS.get(bank, BANKS[DEFAULT_BANK])
        return KIBOR + config["spread"]

    @staticmethod
    def _emi(principal: float, annual_rate_pct: float, tenure_years: int) -> float:
        r = (annual_rate_pct / 100) / 12
        n = tenure_years * 12
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    @classmethod
    def calculate(cls, price: float, down_pct: float, tenure_years: int, bank: str) -> dict:
        """Calculate mortgage for a given property price (in rupees)."""
        loan = price * (1 - down_pct / 100)
        down_payment = price * down_pct / 100
        annual_rate = cls.annual_rate(bank)
        emi = cls._emi(loan, annual_rate, tenure_years)
        total_payment = emi * tenure_years * 12
        min_salary = emi * 3
        bank_config = BANKS.get(bank, BANKS[DEFAULT_BANK])
        return {
            "bank": bank,
            "bank_name": bank_config["name"],
            "annual_rate": annual_rate,
            "loan_amount": loan,
            "down_payment": down_payment,
            "monthly_emi": emi,
            "total_payment": total_payment,
            "min_salary": min_salary,
            "price": price,
            "down_pct": down_pct,
            "tenure_years": tenure_years,
        }

    @classmethod
    def reverse_calculate(cls, monthly_budget: float, down_pct: float, tenure_years: int, bank: str) -> float:
        """Return max affordable property price (rupees) given a monthly EMI budget."""
        annual_rate = cls.annual_rate(bank)
        r = (annual_rate / 100) / 12
        n = tenure_years * 12
        if r == 0:
            max_loan = monthly_budget * n
        else:
            max_loan = monthly_budget * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        return max_loan / (1 - down_pct / 100)


def format_mortgage_result(result: dict) -> str:
    """Format mortgage calc result for WhatsApp (no markdown, emoji structure)."""
    rate_line = f"{result['annual_rate']:.1f}% p.a. — {result['bank_name']}"
    disclaimer = f"(KIBOR {KIBOR:.0f}% hardcoded — actual rate may vary)"
    lines = [
        f"🏠 Property Price: PKR {format_pkr(result['price'])}",
        f"💳 Down Payment ({result['down_pct']:.0f}%): PKR {format_pkr(result['down_payment'])}",
        f"🏦 Loan Amount: PKR {format_pkr(result['loan_amount'])}",
        "",
        f"📅 Tenure: {result['tenure_years']} years",
        f"📈 Rate: {rate_line}",
        f"   {disclaimer}",
        "",
        f"💰 Monthly EMI: PKR {format_pkr(result['monthly_emi'])}",
        f"💸 Total Payment: PKR {format_pkr(result['total_payment'])}",
        "",
        f"✅ Recommended Min. Salary: PKR {format_pkr(result['min_salary'])} / month",
    ]
    return "\n".join(lines)


def format_reverse_result(monthly_budget: float, max_price: float, down_pct: float, tenure_years: int, bank: str) -> str:
    """Format reverse mortgage result for WhatsApp."""
    bank_name = BANKS.get(bank, BANKS[DEFAULT_BANK])["name"]
    annual_rate = MortgageEngine.annual_rate(bank)
    down_amount = max_price * down_pct / 100
    lines = [
        f"💡 With PKR {format_pkr(monthly_budget)}/month you can afford:",
        "",
        f"🏠 Max Property Price: PKR {format_pkr(max_price)}",
        f"💳 Down Payment ({down_pct:.0f}%): PKR {format_pkr(down_amount)}",
        f"📅 Tenure: {tenure_years} years at {annual_rate:.1f}% p.a. ({bank_name})",
        "",
        f"(KIBOR {KIBOR:.0f}% hardcoded — actual rate may vary)",
        "",
        "Searching for properties within this budget...",
    ]
    return "\n".join(lines)
