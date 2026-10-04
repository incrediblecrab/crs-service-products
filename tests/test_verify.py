"""verify: a synced store passes, each planted defect is named, and the command exits 1 when one is."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crs_products import verify as verify_module
from crs_products.pipeline import MAX_ATTEMPTS
from crs_products.store import SCHEMA, normalize, partition_path, sha256_file, write_parquet
from crs_products.verify import verify
from conftest import ScriptedSource, local_store, run_once, scripted

REPO = Path(__file__).parent.parent
T1 = "2026-09-01T10:00:00Z"


@pytest.fixture
def synced(tmp_path):
    store, state = local_store(tmp_path), scripted(units={"R40001": T1, "R40002": T1, "R40003": T1, "IN12001": T1})
    run_once(store, state)
    return store, state


def edit_manifest(store, change):
    path = store.root / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def rewrite_partition(store, key, ids):
    """Replaces a partition's rows as given (unsorted, unchecked) and updates its manifest entry to match, so only the planted defect is left."""
    path = store.root / partition_path(key)
    pq.write_table(pa.Table.from_pylist([normalize({"id": uid, "title": str(uid)}) for uid in ids], schema=SCHEMA), path)
    edit_manifest(store, lambda m: m["partitions"][key].update(rows=len(ids), sha256=sha256_file(path)))


PLANTS = {
    "missing file": (lambda store: (store.root / "data/IN12.parquet").unlink(), "IN12: data/IN12.parquet is in the manifest but not in the repo"),
    "sha256": (lambda store: edit_manifest(store, lambda m: m["partitions"]["R40"].update(sha256="0" * 64)), "R40: data/R40.parquet has sha256"),
    "row count": (lambda store: edit_manifest(store, lambda m: m["partitions"]["R40"].update(rows=4)), "R40: 3 rows, the manifest says 4"),
    "extra file": (lambda store: write_parquet([{"id": "R41001"}], store.root / "data/R41.parquet"), "data/R41.parquet is not in the manifest"),
    "misplaced id": (lambda store: rewrite_partition(store, "R40", ["R40001", "R40002", "IN12002"]), "R40: ids that belong elsewhere ['IN12002']"),
    "duplicate id": (lambda store: rewrite_partition(store, "R40", ["R40001", "R40002", "R40002"]), "R40: duplicate ids ['R40002']"),
    "null id": (lambda store: rewrite_partition(store, "R40", ["R40001", None, "R40003"]), "R40: null ids"),
    "complete count": (lambda store: edit_manifest(store, lambda m: m["partitions"]["R40"].update(listed=4)), "R40: complete, but 3 rows + 0 failed != 4 listed"),
}


def test_a_synced_store_passes(synced):
    store, state = synced
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == []
    assert report["rows"] == 4 and report["partitions"] == 2 and report["complete"] == 2 and report["text_sources"] == {"pdf": 4}
    assert report["live"] == {"count": 4, "listed": 4, "on_hub": 4, "missing": 0, "missing_recorded_as_failed": 0, "extra": 0, "missing_sample": [], "extra_sample": []}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_each_planted_defect_is_named(synced, plant):
    store, _ = synced
    change, expected = PLANTS[plant]
    change(store)
    problems = verify(store)["problems"]
    assert any(problem.startswith(expected) for problem in problems), problems


def test_the_live_diff_allows_a_few_products_since_the_last_sync_but_not_more(synced):
    store, state = synced
    state.units.update({f"R40{n:03d}": T1 for n in range(100, 110)})
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == [] and report["live"]["missing"] == 10
    state.units["R40110"] = T1
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == ["R40: complete, but 11 ids differ from the live listing (tolerance 10)"]


def test_the_live_diff_counts_products_the_hub_has_and_the_api_dropped(synced, monkeypatch):
    store, state = synced
    monkeypatch.setattr(verify_module, "TOLERANCE_MIN", 0)
    monkeypatch.setattr(verify_module, "TOLERANCE_SHARE", 0)
    del state.units["R40003"]
    report = verify(store, ScriptedSource(state))
    assert report["live"]["extra"] == 1 and report["live"]["extra_sample"] == ["R40003"]
    assert report["problems"] == ["R40: complete, but 1 ids differ from the live listing (tolerance 0)"]


def test_products_recorded_as_failed_do_not_count_against_the_live_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_module, "TOLERANCE_MIN", 0)
    monkeypatch.setattr(verify_module, "TOLERANCE_SHARE", 0)
    store, state = local_store(tmp_path), scripted(units={"R40001": T1, "R40002": T1}, fail={"R40002"})
    for _ in range(MAX_ATTEMPTS):
        run_once(store, state)
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == [] and report["failed"] == 1
    assert report["live"]["missing"] == 1 and report["live"]["missing_recorded_as_failed"] == 1
    state.units["R40003"] = T1
    assert verify(store, ScriptedSource(state))["problems"] == ["R40: complete, but 1 ids differ from the live listing (tolerance 0)"]


def test_an_incomplete_partition_is_not_held_to_the_live_listing(tmp_path):
    store, state = local_store(tmp_path), scripted(units={f"R40{n:03d}": T1 for n in range(20)}, stop_after=2)
    run_once(store, state)
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == [] and report["complete"] == 0 and report["live"]["missing"] == 18


def test_no_manifest_is_a_problem(tmp_path):
    assert verify(local_store(tmp_path))["problems"] == ["no manifest.json"]


def test_short_live_listing_is_degraded_not_a_claim_of_extra_rows(synced):
    store, state = synced
    del state.units["R40003"]
    state.count = 4
    report = verify(store, ScriptedSource(state))
    assert report["problems"] == []
    assert "source listing incomplete" in report["warnings"][0]
    assert report["live"]["status"] == "incomplete"
    assert report["live"]["extra"] is None and report["live"]["missing"] is None
    PLANTS["sha256"][0](store)
    assert verify(store, ScriptedSource(state))["problems"], "a source outage must not suppress integrity failures"


def test_live_extra_rows_are_confirmed_through_detail_before_reporting(synced):
    store, state = synced
    del state.units["R40003"]
    state.exists.add("R40003")
    report = verify(store, ScriptedSource(state))
    assert state.exists_asked == ["R40003"]
    assert report["problems"] == [] and report["live"]["status"] == "incomplete"
    assert "still serves" in report["warnings"][0]


def test_an_unreconciled_source_does_not_replace_the_last_complete_sync(synced):
    store, state = synced
    before = store.read_manifest()
    state.count = 100
    state.units = {"R40001": T1}
    state.fetched.clear()
    result = run_once(store, state)
    after = store.read_manifest()
    assert result["finished"] is False and result["stopped"] == "incomplete listing"
    assert result["warnings"] and not state.fetched and not state.exists_asked
    assert after["listing"] == before["listing"] and after["partitions"] == before["partitions"]
    assert after["source_listing"]["complete"] is False
    assert "source listing incomplete" in store.read_text("README.md")


def test_a_recovered_listing_clears_degraded_status_even_without_data_changes(synced):
    store, state = synced
    state.count = 100
    run_once(store, state)
    assert store.read_manifest()["source_listing"]["complete"] is False
    state.count = None
    run = run_once(store, state)
    assert run["finished"] and run["commits"] > 0
    assert store.read_manifest()["source_listing"]["complete"] is True
    assert "**Degraded:**" not in store.read_text("README.md")


def run_cli(*args):
    env = {key: value for key, value in os.environ.items() if key not in ("GITHUB_ACTIONS", "GITHUB_OUTPUT", "HF_OIDC_RESOURCE")}
    return subprocess.run([sys.executable, "-m", "crs_products", *args], capture_output=True, text=True, cwd=REPO, env=env)


def test_the_verify_command_exits_1_on_a_planted_defect(synced):
    store, _ = synced
    clean = run_cli("verify", "--local", str(store.root))
    assert clean.returncode == 0, clean.stderr
    PLANTS["sha256"][0](store)
    planted = run_cli("verify", "--local", str(store.root))
    assert planted.returncode == 1 and '"R40: data/R40.parquet has sha256' in planted.stdout
