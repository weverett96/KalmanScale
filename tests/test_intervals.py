import pytest

from kalmanscale.intervals import parse_rides, parse_wellness


def test_parse_wellness_converts_kg_and_drops_zero_body_fat():
    records = [
        {"id": "2026-09-01", "weight": 80.0, "bodyFat": 18.5},
        {"id": "2026-09-02", "weight": 79.8, "bodyFat": 0},      # Garmin quirk
        {"id": "2026-09-03", "weight": 79.9, "bodyFat": None},
        {"id": "2026-09-04", "weight": None, "bodyFat": 18.0},    # no weigh-in
        {"id": "2026-09-05"},
    ]
    out = parse_wellness(records)
    assert out["2026-09-01"] == (pytest.approx(176.4), 18.5)
    assert out["2026-09-02"][1] is None
    assert out["2026-09-03"][1] is None
    assert "2026-09-04" not in out
    assert "2026-09-05" not in out


def test_parse_rides_filters_types_prefers_work_and_sums_per_day():
    activities = [
        {"type": "Ride", "start_date_local": "2026-09-01T07:00:00", "icu_joules": 900_000, "calories": 1500},
        {"type": "VirtualRide", "start_date_local": "2026-09-01T18:00:00", "icu_joules": 400_000},
        {"type": "Run", "start_date_local": "2026-09-01T12:00:00", "calories": 600},
        {"type": "GravelRide", "start_date_local": "2026-09-02T08:00:00", "icu_joules": None, "calories": 1100},
        {"type": "Ride", "start_date_local": "2026-09-03T08:00:00"},
    ]
    out = parse_rides(activities)
    assert out == {"2026-09-01": 1300, "2026-09-02": 1100}
