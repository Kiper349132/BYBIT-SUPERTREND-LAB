"""Re-used candle files are trimmed to exactly the requested period."""
from datetime import datetime, timedelta, timezone


from lab_data import find_reusable_candles, save_candles, market_to_df
from synthetic import synthetic_market


def test_longer_file_is_trimmed_to_request(tmp_path):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df36 = market_to_df(synthetic_market(36 * 730, "60", end_ms=now_ms - 3_600_000))
    save_candles(df36, "TEST", "60", tmp_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=round(18 * 30.4375))
    got, path = find_reusable_candles("TEST", "60", start, end, tmp_path)
    assert got is not None
    span_days = (got["timestamp"].iloc[-1] - got["timestamp"].iloc[0]) / 86_400_000
    assert abs(span_days - (end - start).days) <= 1.0          # v0.7.1 returned ~1095 days here
    assert got["timestamp"].iloc[0] >= int(start.timestamp() * 1000)


def test_stale_file_is_not_reused(tmp_path):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = market_to_df(synthetic_market(18 * 730, "60", end_ms=now_ms - 3 * 86_400_000))   # 3 days old
    save_candles(df, "TEST", "60", tmp_path)
    end = datetime.now(timezone.utc)
    got, _ = find_reusable_candles("TEST", "60", end - timedelta(days=400), end, tmp_path)
    assert got is None


def test_partial_download_files_are_ignored(tmp_path):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = market_to_df(synthetic_market(500 * 24, "60", end_ms=now_ms - 3_600_000))
    df.to_csv(tmp_path / "TEST_60m_DOWNLOADING.csv", index=False)
    end = datetime.now(timezone.utc)
    got, _ = find_reusable_candles("TEST", "60", end - timedelta(days=300), end, tmp_path)
    assert got is None
