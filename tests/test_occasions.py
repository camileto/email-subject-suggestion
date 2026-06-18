from datetime import date

from app.services import occasions as occasions_module
from app.services.occasions import get_upcoming_occasions


def _holiday(name: str, iso_date: str, locations: str = "All", type_: str = "National holiday") -> dict:
    return {"name": name, "date": {"iso": iso_date}, "type": [type_], "locations": locations}


def test_get_upcoming_occasions_filters_window_and_crosses_year_boundary(monkeypatch):
    occasions_module._cache.clear()
    # Mimics Calendarific's real behavior: holidays are filtered server-side by year.
    holidays_by_year = {
        2026: [_holiday("Christmas", "2026-12-25"), _holiday("Independence Day", "2026-09-07")],
        2027: [_holiday("New Year", "2027-01-01")],
    }
    monkeypatch.setattr(occasions_module, "fetch_holidays", lambda country, year: holidays_by_year[year])

    occasions = get_upcoming_occasions("BR", date(2026, 12, 1), lookahead_days=35)

    assert [o["name"] for o in occasions] == ["Christmas", "New Year"]
    assert occasions[0]["days_until"] == 24
    assert occasions[1]["days_until"] == 31


def test_get_upcoming_occasions_excludes_past_and_out_of_range(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [_holiday("Carnival", "2026-02-14"), _holiday("Christmas", "2026-12-25")],
    )

    occasions = get_upcoming_occasions("BR", date(2026, 12, 1), lookahead_days=10)

    assert occasions == []


def test_get_upcoming_occasions_excludes_state_specific_and_irrelevant_types(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [
            _holiday("Juneteenth", "2026-06-19", locations="All", type_="National holiday"),
            _holiday("Juneteenth Day", "2026-06-19", locations="TX", type_="Local holiday"),
            _holiday("Some UN Day", "2026-06-20", locations="All", type_="United Nations observance"),
        ],
    )

    occasions = get_upcoming_occasions("US", date(2026, 6, 17), lookahead_days=10)

    assert [o["name"] for o in occasions] == ["Juneteenth"]


def test_get_upcoming_occasions_deduplicates_same_name_and_date(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [
            _holiday("Christmas", "2026-12-25"),
            _holiday("Christmas", "2026-12-25"),
        ],
    )

    occasions = get_upcoming_occasions("BR", date(2026, 12, 1), lookahead_days=30)

    assert len(occasions) == 1


def test_get_upcoming_occasions_caches_per_country_year(monkeypatch):
    occasions_module._cache.clear()
    call_count = {"n": 0}

    def fake_fetch(country: str, year: int) -> list[dict]:
        call_count["n"] += 1
        return [_holiday("Christmas", "2026-12-25")]

    monkeypatch.setattr(occasions_module, "fetch_holidays", fake_fetch)

    # Non-US country also fetches "US" once, to check for Black Friday/Cyber
    # Monday — so the first call hits the network twice (BR + US), and the
    # second call should reuse both cache entries (0 new fetches).
    get_upcoming_occasions("BR", date(2026, 12, 1), lookahead_days=10)
    assert call_count["n"] == 2

    get_upcoming_occasions("BR", date(2026, 12, 1), lookahead_days=10)
    assert call_count["n"] == 2


def test_get_upcoming_occasions_does_not_double_fetch_when_country_is_us(monkeypatch):
    occasions_module._cache.clear()
    call_count = {"n": 0}

    def fake_fetch(country: str, year: int) -> list[dict]:
        call_count["n"] += 1
        return [_holiday("Christmas", "2026-12-25")]

    monkeypatch.setattr(occasions_module, "fetch_holidays", fake_fetch)

    get_upcoming_occasions("US", date(2026, 12, 1), lookahead_days=10)

    assert call_count["n"] == 1


def test_get_upcoming_occasions_borrows_black_friday_from_us(monkeypatch):
    occasions_module._cache.clear()
    by_country = {
        "BR": [_holiday("Republic Day", "2026-11-15")],
        "US": [_holiday("Black Friday", "2026-11-27", type_="Observance"), _holiday("Veterans Day", "2026-11-11")],
    }
    monkeypatch.setattr(occasions_module, "fetch_holidays", lambda country, year: by_country[country])

    occasions = get_upcoming_occasions("BR", date(2026, 11, 1), lookahead_days=30)

    names = [o["name"] for o in occasions]
    assert "Black Friday" in names
    assert "Veterans Day" not in names  # only the specific shopping events are borrowed
    assert "Republic Day" in names


def test_get_upcoming_occasions_prefers_country_specific_black_friday_over_us_borrow(monkeypatch):
    occasions_module._cache.clear()
    by_country = {
        "BR": [_holiday("Black Friday", "2026-11-20", type_="Observance")],
        "US": [_holiday("Black Friday", "2026-11-27", type_="Observance")],
    }
    monkeypatch.setattr(occasions_module, "fetch_holidays", lambda country, year: by_country[country])

    occasions = get_upcoming_occasions("BR", date(2026, 11, 1), lookahead_days=30)

    black_fridays = [o for o in occasions if o["name"] == "Black Friday"]
    assert len(black_fridays) == 1
    assert black_fridays[0]["date"] == "2026-11-20"


def test_get_upcoming_occasions_drops_gift_occasion_inside_lead_time(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [_holiday("Mother's Day", "2026-06-18")],
    )

    # reference_date is the day before Mother's Day: days_until=1, below the default lead time of 2.
    occasions = get_upcoming_occasions("BR", date(2026, 6, 17), lookahead_days=10)

    assert occasions == []


def test_get_upcoming_occasions_keeps_gift_occasion_outside_lead_time(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [_holiday("Mother's Day", "2026-06-19")],
    )

    occasions = get_upcoming_occasions("BR", date(2026, 6, 17), lookahead_days=10)

    assert [o["name"] for o in occasions] == ["Mother's Day"]


def test_get_upcoming_occasions_lead_time_is_configurable(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [_holiday("Mother's Day", "2026-06-18")],
    )

    # days_until=1, but caller only requires 0 days of lead time.
    occasions = get_upcoming_occasions("BR", date(2026, 6, 17), lookahead_days=10, gift_occasion_lead_time_days=0)

    assert [o["name"] for o in occasions] == ["Mother's Day"]


def test_get_upcoming_occasions_shopping_event_ignores_lead_time(monkeypatch):
    occasions_module._cache.clear()
    monkeypatch.setattr(
        occasions_module,
        "fetch_holidays",
        lambda country, year: [_holiday("Black Friday", "2026-11-27", type_="Observance")],
    )

    # days_until=0 (the day itself) — would be dropped for a gift occasion, but Black Friday is exempt.
    occasions = get_upcoming_occasions("US", date(2026, 11, 27), lookahead_days=10)

    assert [o["name"] for o in occasions] == ["Black Friday"]
