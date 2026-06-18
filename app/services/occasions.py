import time
from datetime import date, timedelta

from ..clients.calendarific_client import fetch_holidays
from ..config import OCCASION_CACHE_TTL_SECONDS

# Calendarific bundles every US state's own proclamation of the same holiday
# (e.g. 30+ separate "Juneteenth" entries, one per state) plus UN/worldwide
# awareness days that aren't useful marketing angles. Keeping only
# nationwide entries of these two types is what's left over: real national
# holidays and the commercial observances (Black Friday, Mother's/Father's
# Day, Valentine's Day...) that are actually worth referencing in a subject.
_RELEVANT_TYPES = {"National holiday", "Observance"}

# These shopping events are commercially observed on the same calendar date
# worldwide (synchronized to the US convention), but Calendarific only ever
# tags them under the "US" country bucket — even for countries, like Brazil,
# that run the same Black Friday. Borrowing the real US-tagged date for
# other countries is more honest than re-deriving the date formula ourselves.
_GLOBAL_SHOPPING_EVENTS = {"Black Friday", "Cyber Monday"}

# Gift-giving occasions need lead time for the gift to actually arrive —
# unlike a shopping event (Black Friday, Cyber Monday), where the day of is
# the peak moment, not a delivery deadline. Matched by keyword, not exact
# name, since Calendarific spells these differently per country (e.g.
# "Christmas Day" vs "Christmas", "Brazilian Valentine's Day" vs
# "Valentine's Day").
_GIFT_OCCASION_KEYWORDS = ("christmas", "mother's day", "father's day", "valentine")


def _is_gift_occasion(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in _GIFT_OCCASION_KEYWORDS)

_cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}


def _get_holidays_cached(country: str, year: int) -> list[dict]:
    cached = _cache.get((country, year))
    now = time.monotonic()
    if cached and now - cached[0] < OCCASION_CACHE_TTL_SECONDS:
        return cached[1]
    holidays = fetch_holidays(country, year)
    _cache[(country, year)] = (now, holidays)
    return holidays


def _collect(country: str, reference_date: date, end_date: date, years_needed: set[int]) -> list[dict]:
    seen_names_by_date: set[tuple[str, str]] = set()
    occasions = []
    for year in years_needed:
        for holiday in _get_holidays_cached(country, year):
            if holiday.get("locations") != "All" or not _RELEVANT_TYPES.intersection(holiday.get("type", [])):
                continue
            holiday_date = date.fromisoformat(holiday["date"]["iso"][:10])
            if not (reference_date <= holiday_date <= end_date):
                continue
            key = (holiday["name"], holiday_date.isoformat())
            if key in seen_names_by_date:
                continue
            seen_names_by_date.add(key)
            occasions.append(
                {
                    "name": holiday["name"],
                    "date": holiday_date.isoformat(),
                    "days_until": (holiday_date - reference_date).days,
                }
            )
    return occasions


def get_upcoming_occasions(
    country: str, reference_date: date, lookahead_days: int, gift_occasion_lead_time_days: int = 2
) -> list[dict]:
    """Real, computed occasions within the window — never invented by the
    LLM. Calendarific covers both official holidays and commercial
    observances (Black Friday, Mother's/Father's Day, Valentine's Day),
    which is what makes it useful here beyond pure public-holiday APIs.

    Gift occasions closer than gift_occasion_lead_time_days are dropped
    entirely rather than left for the LLM to judge — a deterministic filter
    is reliable where a prompt instruction asking the model not to promise
    a too-late gift was not."""
    end_date = reference_date + timedelta(days=lookahead_days)
    years_needed = {reference_date.year, end_date.year}

    occasions = _collect(country, reference_date, end_date, years_needed)

    if country != "US":
        known_names = {o["name"] for o in occasions}
        for shopping_event in _collect("US", reference_date, end_date, years_needed):
            if shopping_event["name"] in _GLOBAL_SHOPPING_EVENTS and shopping_event["name"] not in known_names:
                occasions.append(shopping_event)

    occasions = [
        o
        for o in occasions
        if o["name"] in _GLOBAL_SHOPPING_EVENTS
        or not _is_gift_occasion(o["name"])
        or o["days_until"] >= gift_occasion_lead_time_days
    ]

    return sorted(occasions, key=lambda o: o["days_until"])
