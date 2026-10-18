from datetime import datetime, timezone

from main import _inside_candle_window, _seconds_until_next_candle_check


def dt(h, m, s):
    return datetime(2026, 1, 1, h, m, s, tzinfo=timezone.utc)


def test_scheduler_waits_for_current_boundary_delay():
    delay = _seconds_until_next_candle_check(dt(10, 0, 1))
    assert 0.9 <= delay <= 1.1


def test_scheduler_does_not_drift_from_hour_boundary():
    delay = _seconds_until_next_candle_check(dt(10, 15, 17))
    assert 2684 <= delay <= 2686  # 11:00:02 - 10:15:17


def test_signal_window_is_immediately_after_candle_close_delay():
    assert not _inside_candle_window(dt(10, 0, 1))
    assert _inside_candle_window(dt(10, 0, 2))
    assert _inside_candle_window(dt(10, 0, 45))
    assert not _inside_candle_window(dt(10, 2, 0))
