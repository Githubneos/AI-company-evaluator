"""Atomic writes: readers never see a partial file, failures never damage the old one."""

import json
import threading

import numpy as np
import pandas as pd
import pytest

from evaluator import io
from evaluator.data import fundamentals
from evaluator.data.edgar import EdgarClient


def _frame(n: int, value: float) -> pd.DataFrame:
    return pd.DataFrame({"a": np.full(n, value), "b": np.arange(n, dtype="int64")})


def test_round_trips(tmp_path):
    frame = _frame(10, 1.5)
    io.atomic_write_parquet(frame, tmp_path / "x.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "x.parquet"), frame)

    io.atomic_write_json({"k": [1, 2]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"k": [1, 2]}

    io.atomic_write_text("hello", tmp_path / "nested" / "x.txt")  # parent created
    assert (tmp_path / "nested" / "x.txt").read_text() == "hello"


def test_npz_lands_at_the_target_not_a_suffixed_twin(tmp_path):
    # np.savez_compressed appends ".npz" to names lacking it; the temp name must
    # already end in .npz or the rename would move an empty file into place.
    target = tmp_path / "oos.npz"
    io.atomic_write_bytes(lambda tmp: np.savez_compressed(tmp, y=np.arange(5)), target)

    assert np.array_equal(np.load(target)["y"], np.arange(5))
    assert [p.name for p in tmp_path.iterdir()] == ["oos.npz"]


def test_failed_write_keeps_the_old_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "x.parquet"
    io.atomic_write_parquet(_frame(3, 1.0), target)

    def boom(tmp):
        with open(tmp, "wb") as fh:
            fh.write(b"PAR1 partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        io.atomic_write_bytes(boom, target)

    pd.testing.assert_frame_equal(pd.read_parquet(target), _frame(3, 1.0))
    assert [p.name for p in tmp_path.iterdir()] == ["x.parquet"]


def test_unreadable_cache_is_a_miss(tmp_path, caplog):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"PAR1 half-written garbage")

    assert io.read_parquet_or_none(bad) is None
    assert io.read_parquet_or_none(tmp_path / "missing.parquet") is None
    assert "discarding unreadable cache" in caplog.text


def test_concurrent_writers_and_readers_never_see_a_partial_file(tmp_path):
    """The production failure, reproduced: many writers, many readers, one path."""
    target = tmp_path / "shared.parquet"
    io.atomic_write_parquet(_frame(50_000, 0.0), target)
    # Frames differ in size so a torn file cannot pass for a complete one.
    frames = {float(v): _frame(20_000 + v * 3_000, float(v)) for v in range(8)}
    errors: list[BaseException] = []
    seen: set[float] = set()
    stop = threading.Event()

    def writer(value: float):
        try:
            for _ in range(15):
                io.atomic_write_parquet(frames[value], target)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        while not stop.is_set():
            try:
                got = pd.read_parquet(target)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            value = float(got["a"].iloc[0])
            # Every read is exactly one complete frame that some writer wrote.
            if value != 0.0:
                pd.testing.assert_frame_equal(got, frames[value])
            seen.add(value)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(v,)) for v in frames]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert not errors, errors[:3]
    assert len(seen) > 1  # readers really did observe different versions
    assert [p.name for p in tmp_path.iterdir()] == ["shared.parquet"]


def test_corrupt_edgar_cache_is_refetched_and_repaired(tmp_path, monkeypatch):
    client = EdgarClient(cache_dir=tmp_path)
    (tmp_path / "TEST.parquet").write_bytes(b"PAR1 garbage")
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2024-01-05"],
                "acceptanceDateTime": ["2024-01-05T16:30:00.000Z"],
                "reportDate": ["2024-01-04"],
                "items": ["2.02"],
                "accessionNumber": ["0000000000-24-000001"],
            }
        }
    }
    calls = []
    monkeypatch.setattr(client, "_get_json", lambda url: calls.append(url) or payload)

    frame = client.filings(1, "TEST")

    assert len(calls) == 1 and list(frame["form"]) == ["8-K"]
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "TEST.parquet"), frame)


def test_corrupt_fundamentals_cache_is_refetched_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(fundamentals, "CACHE_DIR", tmp_path)
    (tmp_path / "fundamentals").mkdir()
    (tmp_path / "fundamentals" / "TEST.parquet").write_bytes(b"PAR1 garbage")

    class Client:
        calls = 0

        def _get_json(self, url):
            Client.calls += 1
            return None  # provider unavailable: an empty frame, not an exception

    frame = fundamentals.load_fundamentals("TEST", 1, client=Client())

    assert Client.calls == 1
    assert frame.empty and "filed" in frame.columns
