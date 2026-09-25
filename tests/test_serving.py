"""Serving endpoint wiring that does not need a live market-data request."""

from fastapi.testclient import TestClient

import evaluator.serving.app as serving


def test_analog_endpoint_forwards_requested_k(monkeypatch):
    seen = {}

    def fake_payload(ticker, **kwargs):
        seen["ticker"] = ticker
        seen.update(kwargs)
        return {"historical_analogs": {"available": True, "matches": []}}

    monkeypatch.setattr(serving, "build_payload", fake_payload)
    response = TestClient(serving.app).get("/analogs/test?k=7")

    assert response.status_code == 200
    assert seen == {
        "ticker": "TEST",
        "with_sentiment": False,
        "with_analogs": True,
        "analog_k": 7,
    }


def test_dashboard_and_static_assets_are_served():
    client = TestClient(serving.app)

    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text

    for asset in ("/static/app.js", "/static/app.css"):
        assert client.get(asset).status_code == 200


def test_universe_lists_tickers_for_search():
    body = TestClient(serving.app).get("/universe").json()

    assert body["count"] == len(body["tickers"])
    assert {"ticker", "name", "sector"} <= set(body["tickers"][0])


def test_prices_endpoint_returns_trailing_closes(monkeypatch):
    import pandas as pd

    import evaluator.data.sources as sources

    index = pd.bdate_range("2024-01-01", periods=30)
    frame = pd.DataFrame({"close": range(30)}, index=index, dtype="float64")
    seen = {}

    def fake_load_prices(ticker, start, *args, **kwargs):
        seen["args"] = (ticker, start)
        return frame

    monkeypatch.setattr(sources, "load_prices", fake_load_prices)
    body = TestClient(serving.app).get("/prices/test?days=10").json()

    # Same lookback as scoring, so the chart reuses the scoring price cache.
    assert seen["args"] == ("TEST", serving.PRICE_LOOKBACK_START)
    assert body["close"] == [float(v) for v in range(20, 30)]
    assert body["dates"][-1] == index[-1].strftime("%Y-%m-%d")


def test_unknown_ticker_is_a_client_error_not_a_crash(monkeypatch):
    def no_prices(ticker, **kwargs):
        raise ValueError(f"no price data returned for {ticker!r}")

    monkeypatch.setattr(serving, "build_payload", no_prices)
    client = TestClient(serving.app)

    for path in ("/payload/zzzz", "/analogs/zzzz", "/evaluate/zzzz"):
        response = client.get(path)
        assert response.status_code == 400, path
        assert "no price data" in response.json()["detail"]


def test_validation_endpoint_carries_baselines_when_computed(monkeypatch):
    class FakeModel:
        metadata = {
            "target": "magnitude_1d", "schema": "s", "train_start": "a", "train_end": "b",
            "train_rows": 1, "train_tickers": 1, "validation": {}, "data_caveats": [],
        }

    monkeypatch.setattr(serving, "load_model", lambda target: FakeModel())
    client = TestClient(serving.app)

    monkeypatch.setattr(serving, "load_report", lambda target, kind: None)
    body = client.get("/model/magnitude_1d/validation").json()
    assert body["baselines"] is None and body["ablations"] is None and body["vol_benchmarks"] is None

    reports = {"baselines": {"verdict": "v"}, "ablations": {"sets": {}}, "vol_benchmarks": {"benchmarks": {}}}
    monkeypatch.setattr(serving, "load_report", lambda target, kind: reports[kind])
    body = client.get("/model/magnitude_1d/validation").json()
    assert body["baselines"] == {"verdict": "v"} and body["ablations"] == {"sets": {}}
    assert body["vol_benchmarks"] == {"benchmarks": {}}


def test_malformed_tickers_are_refused_at_the_edge(monkeypatch):
    client = TestClient(serving.app)
    called = []
    monkeypatch.setattr(serving, "build_payload", lambda *a, **k: called.append(1) or {})

    for bad in ("A" * 13, "AAPL%20X", "AA;PL", "AA%27PL", "%2E%2E%2Fsecret"):
        assert client.get(f"/payload/{bad}").status_code in (404, 422), bad

    assert called == []


def test_real_world_symbols_pass_validation(monkeypatch):
    seen = []
    monkeypatch.setattr(serving, "build_payload", lambda ticker, **k: seen.append(ticker) or {"ok": True})
    client = TestClient(serving.app)

    for symbol in ("aapl", "BRK.B", "BF-B", "^VIX"):
        assert client.get(f"/payload/{symbol}").status_code == 200, symbol

    assert seen == ["AAPL", "BRK.B", "BF-B", "^VIX"]


def test_dates_that_would_become_file_names_must_be_dates(monkeypatch):
    client = TestClient(serving.app)
    monkeypatch.setattr(serving, "build_payload", lambda *a, **k: {})
    monkeypatch.setattr(serving, "leaderboard", lambda *a, **k: {"available": False, "rows": []})

    for traversal in ("../../etc/passwd", "2026-9-1", "2026-09-01x", "latest"):
        assert client.get("/leaderboard", params={"as_of": traversal}).status_code == 422, traversal
        assert client.get("/payload/AAPL", params={"lookback_start": traversal}).status_code == 422, traversal

    assert client.get("/leaderboard", params={"as_of": "2026-09-18"}).status_code == 200
    assert client.get("/payload/AAPL", params={"lookback_start": "2015-01-01"}).status_code == 200
