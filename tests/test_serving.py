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
