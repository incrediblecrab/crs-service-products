"""Checks that the files on the Hub are what the manifest says, and with a source (--live) that they hold what the API lists."""

import math
from collections import Counter

from .pipeline import MAX_ATTEMPTS
from .source import listing_problem, listing_summary
from .store import partition_of, partition_path

# A complete partition may differ from a live listing by this much: products published or removed since the last run.
TOLERANCE_MIN = 10
TOLERANCE_SHARE = 0.005
SAMPLE = 10


def tolerance(listed):
    return max(TOLERANCE_MIN, math.ceil(TOLERANCE_SHARE * (listed or 0)))


def verify(store, source=None, partition_of=partition_of, tally="text_source", live=None):
    """Returns a report; report["problems"] is empty when every check passed. partition_of, tally (the column whose values are counted, reported as report[tally + "s"]) and live (how --live compares the Hub with the source; None: live_diff) default to the products'."""
    manifest = store.read_manifest()
    if not manifest:
        return {"problems": ["no manifest.json"]}
    problems = []
    entries = manifest.get("partitions") or {}
    files = set(store.list_files("data/"))
    expected = {key: entry.get("file") or partition_path(key) for key, entry in entries.items()}
    sha256s = store.file_sha256s(sorted(path for path in expected.values() if path in files))
    stored, tallies = {}, Counter()
    for key, path in sorted(expected.items()):
        entry = entries[key]
        if path not in files:
            problems.append(f"{key}: {path} is in the manifest but not in the repo")
            continue
        if sha256s.get(path) != entry.get("sha256"):
            problems.append(f"{key}: {path} has sha256 {str(sha256s.get(path))[:12]}, the manifest says {str(entry.get('sha256'))[:12]}")
        columns = store.read_columns(path, ["id", tally])
        ids = columns["id"]
        if len(ids) != entry.get("rows"):
            problems.append(f"{key}: {len(ids)} rows, the manifest says {entry.get('rows')}")
        if any(uid is None for uid in ids):
            problems.append(f"{key}: null ids")
        duplicates = sorted(uid for uid, n in Counter(ids).items() if uid is not None and n > 1)
        if duplicates:
            problems.append(f"{key}: duplicate ids {duplicates[:SAMPLE]}")
        misplaced = sorted(uid for uid in ids if uid is not None and partition_of(uid) != key)
        if misplaced:
            problems.append(f"{key}: ids that belong elsewhere {misplaced[:SAMPLE]}")
        if entry.get("complete") and (entry.get("rows") or 0) + (entry.get("failed") or 0) != entry.get("listed"):
            problems.append(f"{key}: complete, but {entry.get('rows')} rows + {entry.get('failed')} failed != {entry.get('listed')} listed")
        stored[key] = {uid for uid in ids if uid is not None}
        tallies.update(value or "none" for value in columns[tally])
    for path in sorted(files - set(expected.values())):
        problems.append(f"{path} is not in the manifest")
    rows = sum(len(ids) for ids in stored.values())
    listing = manifest.get("listing") or {}
    report = {
        "rows": rows,
        "listed_at_last_full_sync": listing.get("count"),
        "partitions": len(entries),
        "complete": sum(1 for entry in entries.values() if entry.get("complete")),
        "failed": sum(1 for f in (manifest.get("failures") or {}).values() if f["attempts"] >= MAX_ATTEMPTS),
        f"{tally}s": dict(sorted(tallies.items())),
    }
    if source is not None:
        report["live"] = (live or live_diff)(manifest, stored, source, problems, partition_of)
        if report["live"].get("warning"):
            report["warnings"] = [report["live"]["warning"]]
    report["problems"] = problems
    return report


def live_diff(manifest, stored, source, problems, partition_of=partition_of):
    """Exact id sets: what the API lists now against what the Hub holds. Only a complete partition that differs by more than its tolerance is a problem; an incomplete one is still being filled."""
    head, items = source.list_all()
    live = set(items)
    hub = set().union(*stored.values()) if stored else set()
    summary = listing_summary(head, items)
    warning = listing_problem(summary)
    if warning is None:
        present = [uid for uid in sorted(hub - live) if source.exists(uid)]
        if present:
            summary.update(complete=False, unlisted_but_present=len(present))
            warning = f"source listing incomplete: the detail API still serves {len(present)} stored IDs omitted by the listing; exact comparison unavailable"
    if warning:
        return {"status": "incomplete", "count": head["count"], "listed": len(live), "on_hub": len(hub),
                "missing": None, "extra": None, "listing": summary, "warning": warning}
    failed = {uid for uid, f in (manifest.get("failures") or {}).items() if f["attempts"] >= MAX_ATTEMPTS}
    missing, extra = sorted(live - hub), sorted(hub - live)
    by_key = Counter(partition_of(uid) for uid in missing if uid not in failed) + Counter(partition_of(uid) for uid in extra)
    entries = manifest.get("partitions") or {}
    for key, n in sorted(by_key.items()):
        entry = entries.get(key) or {}
        if entry.get("complete") and n > tolerance(entry.get("listed")):
            problems.append(f"{key}: complete, but {n} ids differ from the live listing (tolerance {tolerance(entry.get('listed'))})")
    return {
        "count": head["count"],
        "listed": len(live),
        "on_hub": len(hub),
        "missing": len(missing),
        "missing_recorded_as_failed": sum(1 for uid in missing if uid in failed),
        "extra": len(extra),
        "missing_sample": missing[:SAMPLE],
        "extra_sample": extra[:SAMPLE],
    }
