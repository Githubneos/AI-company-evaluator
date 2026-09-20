"""API keys, the local-only default, and rate limits."""

import time

import pytest
from fastapi.testclient import TestClient

import evaluator.serving.app as serving
from evaluator.serving import auth


@pytest.fixture(autouse=True)
def clean_limits(monkeypatch):
    monkeypatch.delenv(auth.API_KEYS_ENV, raising=False)
    auth.limiter.reset()
    yield
    auth.limiter.reset()


@pytest.fixture
def client(monkeypatch):
    # /evaluate and /sentiment are stubbed: this is about the door, not the room.
    monkeypatch.setattr(serving, "build_payload", lambda ticker, **k: {"ticker": ticker, "gbm": {}})
    monkeypatch.setattr(serving, "log_prediction", lambda row: "id")
    monkeypatch.setattr(serving, "_prediction_row", lambda payload: {})
    monkeypatch.setattr(serving, "llm_evaluate", lambda payload: {"evaluation": "text"})
    return TestClient(serving.app)


def test_public_paths_never_need_a_key(client, monkeypatch):
    monkeypatch.setenv(auth.API_KEYS_ENV, "secret")

    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_with_keys_configured_everything_else_needs_one(client, monkeypatch):
    monkeypatch.setenv(auth.API_KEYS_ENV, "secret,second")

    assert client.get("/universe").status_code == 401
    assert client.get("/universe", headers={auth.API_KEY_HEADER: "wrong"}).status_code == 401
    assert client.get("/universe", headers={auth.API_KEY_HEADER: "secret"}).status_code == 200
    assert client.get("/universe", headers={auth.API_KEY_HEADER: "second"}).status_code == 200
    assert client.get("/health").json()["auth"] == "api-key"


def test_without_keys_costly_endpoints_are_local_only(client):
    # TestClient presents as a loopback client, so it is allowed...
    assert client.get("/evaluate/AAPL").status_code == 200
    assert client.get("/health").json()["auth"] == "local-only"

    # ...and the same call from elsewhere is not.
    allowed, reason = auth.authorize("/evaluate/AAPL", None, "203.0.113.9")
    assert not allowed and auth.API_KEYS_ENV in reason
    assert auth.authorize("/payload/AAPL", None, "203.0.113.9")[0]  # cheap endpoints stay open


def test_a_valid_key_opens_costly_endpoints_from_anywhere(monkeypatch):
    monkeypatch.setenv(auth.API_KEYS_ENV, "secret")

    assert auth.authorize("/evaluate/AAPL", "secret", "203.0.113.9")[0]
    assert not auth.authorize("/evaluate/AAPL", None, "203.0.113.9")[0]


def test_keys_are_compared_in_constant_time(monkeypatch):
    import hmac

    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))

    assert auth.key_is_valid("secret", {"secret"})
    assert not auth.key_is_valid("secre", {"secret"})
    assert not auth.key_is_valid(None, {"secret"})
    assert calls, "keys must go through hmac.compare_digest, never =="


def test_rate_limit_refuses_and_then_refills():
    limiter = auth.RateLimiter({"/evaluate": auth.Limit(2, 60), "default": auth.Limit(100, 60)})

    assert limiter.check("k", "/evaluate/AAPL")[0]
    assert limiter.check("k", "/evaluate/AAPL")[0]
    allowed, retry_after = limiter.check("k", "/evaluate/AAPL")
    assert not allowed and retry_after > 0

    # Another caller has their own bucket, and a cheaper route has its own too.
    assert limiter.check("other", "/evaluate/AAPL")[0]
    assert limiter.check("k", "/payload/AAPL")[0]


def test_rate_limit_refills_over_time(monkeypatch):
    limiter = auth.RateLimiter({"default": auth.Limit(1, 10)})
    clock = [1000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock[0])

    assert limiter.check("k", "/x")[0]
    assert not limiter.check("k", "/x")[0]

    clock[0] += 11  # a full refill period later
    assert limiter.check("k", "/x")[0]


def test_the_endpoint_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter({"default": auth.Limit(1, 60)}))
    monkeypatch.setattr(serving, "limiter", auth.limiter)

    assert client.get("/universe").status_code == 200
    refused = client.get("/universe")

    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) > 0
    assert "rate limit" in refused.json()["detail"]


def test_identity_is_the_key_when_present_else_the_caller():
    assert auth.identify("abcdefghijkl", "1.2.3.4") == "key:abcdefgh"
    assert auth.identify(None, "1.2.3.4") == "ip:1.2.3.4"
    assert auth.identify(None, None) == "ip:unknown"


def test_bucket_selection_is_by_path_prefix():
    limiter = auth.RateLimiter()

    assert limiter.bucket_for("/evaluate/AAPL") == "/evaluate"
    assert limiter.bucket_for("/sentiment/AAPL") == "/sentiment"
    assert limiter.bucket_for("/payload/AAPL") == "default"


def test_limits_are_per_process_and_the_buckets_are_real():
    """A documented limitation: these buckets do not survive a restart."""
    limiter = auth.RateLimiter({"default": auth.Limit(1, 3600)})
    assert limiter.check("k", "/x")[0]
    assert not limiter.check("k", "/x")[0]

    assert auth.RateLimiter({"default": auth.Limit(1, 3600)}).check("k", "/x")[0]
    time.sleep(0)  # no wall-clock dependence in this assertion
