"""The sync loop.

The Congress.gov API lists every CRS product with its updateDate. A run lists them all, groups them into partitions (store.partition_of), and for each partition loads what the Hub holds, fetches the products that are new or whose updateDate changed, drops products the API no longer has, and stages the partition with the manifest. A fetch that returns what is stored keeps the stored row (same_content), the manifest records the updateDate it was checked at (restamped), and a partition in which nothing changed is not staged again. Staged work is committed every checkpoint_seconds and at the end, so a run that dies loses at most one interval. The Parquet files are the state: apart from restamped, nothing else records which products were fetched.

One writer at a time, because www.congress.gov allows 10 requests a minute in total. The manifest names the last writer and when it wrote (the lease). A run defers while another writer's lease is fresh, and commits before it works on its first partition, which claims the lease. Every commit names its parent (store.HubStore.commit), so when two writers race, the second commit is refused and that run stops (Superseded) before fetching.
"""

import copy
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .http import Blocked, MissingKey, QuotaExhausted
from .source import IncompleteListing, listing_problem, listing_summary
from .store import Superseded, normalize, partition_of

log = logging.getLogger("crs_products")

MANIFEST_VERSION = 2
SOURCE = "https://api.congress.gov/v3/crsreport"
MAX_ATTEMPTS = 3
# A product that failed MAX_ATTEMPTS times is left alone this long, then tried again.
RETRY_AFTER_HOURS = 24
# A stored product without text (no rendition answered with any) is fetched again after this long.
TEXT_RETRY_HOURS = 7 * 24
LEASE_MINUTES = 45
RUNS_KEPT = 20
# A listing that would remove more than this share of a stored partition is treated as an upstream glitch, not a deletion.
MAX_REMOVED_SHARE = 0.5
MIN_REMOVED_GUARD = 10
# Stop the run rather than record a failure against the product: these say the source, the key or the Hub is refusing, not that one product is bad.
FATAL = (Blocked, QuotaExhausted, MissingKey, Superseded)
# Runs that stopped for these reasons did their job; any other stop is a failed run, and the probe backs off after it.
CLEAN_STOPS = (None, "budget")
STAMP = "%Y-%m-%dT%H:%M:%SZ"
# The card's front-matter key for probe_state(), and the shape it has; a card with another version is ignored and the probe reads the manifest.
PROBE_KEY = "crs_products_probe"
PROBE_STATE_VERSION = 1


def utcnow():
    return datetime.now(timezone.utc).strftime(STAMP)


def later(stamp, hours):
    return (datetime.strptime(stamp, STAMP) + timedelta(hours=hours)).strftime(STAMP)


def age_hours(stamp):
    """Hours since one of this pipeline's own utcnow() stamps; infinite for a missing one. The API's updateDate is only ever compared as a string: its zone appears not to be the UTC its Z says."""
    if not stamp:
        return float("inf")
    then = datetime.strptime(stamp, STAMP).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600


def writer_identity():
    return "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"


@dataclass
class Unit:
    id: str
    updated_at: str | None


@dataclass
class Partition:
    key: str
    units: dict

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        for uid in sorted(self.units):
            digest.update(f"{uid}\t{self.units[uid].updated_at}\n".encode())
        return digest.hexdigest()

    @property
    def newest(self):
        return max((unit.updated_at or "" for unit in self.units.values()), default="")


@dataclass
class Context:
    """deadline is a time.monotonic() value. only (partition keys) and max_units (products fetched per partition) bound a smoke run; a capped partition stays incomplete and resumes on the next run. refetch fetches every product of each partition the run reaches, as after a parser change. partition_of, comparable (None: this module's comparable) and source_url default to the products'; another dataset synced by this loop passes its own."""

    store: object
    deadline: float
    checkpoint_seconds: float = 600.0
    only: frozenset | None = None
    max_units: int | None = None
    refetch: bool = False
    writer: str = field(default_factory=writer_identity)
    partition_of: object = partition_of
    comparable: object = None
    source_url: str = SOURCE
    stats: dict = field(default_factory=lambda: defaultdict(int))
    pending: list = field(default_factory=list)
    last_commit: float = field(default_factory=time.monotonic)
    claimed: bool = False

    def out_of_time(self):
        return time.monotonic() > self.deadline


def new_manifest(source=SOURCE):
    return {"version": MANIFEST_VERSION, "source": source, "partitions": {}, "failures": {}, "listing": None, "runs": [], "writer": None, "updated_at": None}


def other_writer(manifest, writer):
    """The lease of a different writer that wrote less than LEASE_MINUTES ago, else None."""
    lease = (manifest or {}).get("writer") or {}
    if lease.get("by") and lease["by"] != writer and age_hours(lease.get("at")) * 60 < LEASE_MINUTES:
        return lease
    return None


def backoff_minutes(runs):
    """0 after a clean run; after n failed runs in a row, 15 minutes doubled n-1 times, at most a day."""
    failed = 0
    for run in reversed(runs or []):
        if run.get("stopped") in CLEAN_STOPS:
            break
        failed += 1
    return min(24 * 60, 15 * 2 ** (failed - 1)) if failed else 0


def probe_state(manifest):
    """What decide() reads from the manifest. The card carries it in its front matter, so the probe gets it with the repo's metadata, an API call, instead of downloading manifest.json, which the Hub counts as a download. Failed runs keep only their exception's name, so no error text reaches the card's YAML."""
    if not manifest:
        return None
    failed = []
    for run in reversed(manifest.get("runs") or []):
        if run.get("stopped") in CLEAN_STOPS:
            break
        failed.insert(0, {"stopped": re.match(r"\w*", run.get("stopped") or "").group(0) or "failed", "ended": run.get("ended")})
    listing = manifest.get("listing")
    stored = manifest.get("partitions") or {}
    return {
        "version": PROBE_STATE_VERSION,
        "writer": manifest.get("writer"),
        "failed_runs": failed,
        "listing": {key: listing.get(key) for key in ("count", "newest", "at")} if listing else None,
        "incomplete": sorted(key for key in (listing or {}).get("partitions") or {} if not (stored.get(key) or {}).get("complete")),
    }


def decide(state, head, writer):
    """Whether a full sync is needed, from probe_state() (as the card carries it, or computed from the manifest) and one request's worth of listing (count and newest). Returns (needed, reason)."""
    if not state:
        return True, "no manifest yet"
    holder = other_writer(state, writer)
    if holder:
        return False, f"deferred: {holder['by']} holds the lease (wrote at {holder['at']})"
    runs = state["failed_runs"]
    wait = backoff_minutes(runs)
    if wait and age_hours(runs[-1].get("ended")) * 60 < wait:
        return False, f"backing off {wait} minutes after a failed run: {runs[-1].get('stopped')}"
    listing = state["listing"]
    if not listing:
        return True, "never listed"
    incomplete = state["incomplete"]
    if incomplete:
        return True, f"{len(incomplete)} partitions incomplete: {', '.join(incomplete[:5])}"
    if head["count"] != listing.get("count"):
        return True, f"count {listing.get('count')} -> {head['count']}"
    if head["newest"] != listing.get("newest"):
        return True, f"newest {listing.get('newest')} -> {head['newest']}"
    if age_hours(listing.get("at")) >= 24:
        return True, f"last full listing at {listing.get('at')}"
    return False, "up to date"


def group(items, partition_of=partition_of):
    partitions = {}
    for uid, item in items.items():
        key = partition_of(uid)
        partitions.setdefault(key, Partition(key, {})).units[uid] = Unit(uid, item.get("updateDate"))
    return partitions


def retry_due(manifest, key, entry, now):
    if entry.get("text_retry_at") and entry["text_retry_at"] <= now:
        return True
    return any(f.get("partition") == key and f["attempts"] >= MAX_ATTEMPTS and later(f["at"], RETRY_AFTER_HOURS) <= now for f in manifest["failures"].values())


def order(manifest, partitions):
    """Changed complete partitions first (new products reach the Hub before the backfill resumes), then incomplete ones with the newest products first, then the rest."""
    now = utcnow()

    def rank(partition):
        entry = manifest["partitions"].get(partition.key) or {}
        if not entry.get("complete"):
            return 1
        return 0 if entry.get("fingerprint") != partition.fingerprint or retry_due(manifest, partition.key, entry, now) else 2

    return sorted(sorted(partitions.values(), key=lambda p: p.newest, reverse=True), key=rank)


def flush(ctx, manifest, message=None):
    """Stamp the lease and commit everything staged, the manifest and card included."""
    manifest["writer"] = {"by": ctx.writer, "at": utcnow()}
    ctx.store.stage_manifest(manifest)
    text = message or "; ".join(ctx.pending) or "manifest"
    if ctx.store.commit(text if len(text) <= 500 else text[:497] + "..."):
        ctx.stats["commits"] += 1
    ctx.pending = []
    ctx.claimed = True
    ctx.last_commit = time.monotonic()


def sync(ctx, source):
    """One run. source lists every product (list_all), fetches one (fetch) and confirms one still exists (exists); see source.CrsSource. Returns the run record."""
    started = utcnow()
    base = ctx.store.read_manifest() or new_manifest(ctx.source_url)
    holder = other_writer(base, ctx.writer)
    if holder:
        log.info("deferring to %s, which wrote at %s", holder["by"], holder["at"])
        return {"started": started, "ended": utcnow(), "writer": ctx.writer, "finished": False, "stopped": "deferred", "holder": holder, "commits": 0}
    manifest = copy.deepcopy(base)
    manifest.setdefault("partitions", {})
    manifest.setdefault("failures", {})
    manifest["version"] = MANIFEST_VERSION
    finished, reason, listing, warnings = True, None, None, []
    try:
        head, items = source.list_all()
        manifest["source_listing"] = dict(listing_summary(head, items), at=started)
        problem = listing_problem(manifest["source_listing"])
        if problem:
            raise IncompleteListing(problem)
        partitions = group(items, ctx.partition_of)
        listing = {"count": head["count"], "newest": head["newest"], "listed": len(items), "at": started,
                   "partitions": {key: len(partitions[key].units) for key in sorted(partitions)}}
        # What this run saw, for the card; the probe reads only the published listing.
        manifest["seen"] = {"count": head["count"], "listed": len(items), "at": started}
        for key in manifest["partitions"]:
            partitions.setdefault(key, Partition(key, {}))
        for partition in order(manifest, partitions):
            if ctx.only is not None and partition.key not in ctx.only:
                continue
            if ctx.out_of_time() or not sync_partition(ctx, source, manifest, partition):
                finished, reason = False, "budget"
                break
    except IncompleteListing as error:
        finished, reason = False, "incomplete listing"
        warnings.append(str(error))
        log.warning("%s; stored data and the last complete listing are retained", error)
    except Superseded as error:
        finished, reason = False, "superseded"
        log.warning("stopped: %s", error)
    except FATAL as error:
        finished, reason = False, f"{type(error).__name__}: {error}"[:300]
        log.warning("stopped: %s", reason)
    except Exception as error:  # noqa: BLE001 - recorded in the run record; the probe backs off
        finished, reason = False, f"{type(error).__name__}: {error}"[:300]
        log.exception("run failed")
    run = {"started": started, "ended": utcnow(), "writer": ctx.writer, "finished": finished, "stopped": reason}
    if warnings:
        run["warnings"] = warnings
    run.update({key: ctx.stats[key] for key in ("fetched", "unchanged", "failed", "removed", "suspect_listings")})
    if reason == "superseded":
        return dict(run, commits=ctx.stats["commits"])
    runs = base.get("runs") or []
    # The listing is what the probe compares against, so it is published only by a run that brought every partition up to date with it; until then the probe keeps asking for a run.
    changed = (base.get("source_listing") or {}).get("complete") is False and not warnings
    if finished and ctx.only is None and listing:
        old = base.get("listing") or {}
        changed = changed or any(listing[key] != old.get(key) for key in ("count", "newest", "listed", "partitions")) or age_hours(old.get("at")) > 23
        manifest["listing"] = listing
    if ctx.store.staged or ctx.stats["commits"] or changed or reason not in CLEAN_STOPS or not runs or age_hours(runs[-1].get("ended")) > 23:
        manifest["runs"] = runs[-(RUNS_KEPT - 1):] + [dict(run, commits=ctx.stats["commits"] + 1)]
        try:
            flush(ctx, manifest)
        except Superseded as error:
            run.update(finished=False, stopped="superseded")
            log.warning("stopped: %s", error)
        except Exception as error:  # noqa: BLE001 - the Hub refused the final commit
            run.update(finished=False, stopped=f"{type(error).__name__}: {error}"[:300])
            log.exception("final commit failed")
    return dict(run, commits=ctx.stats["commits"])


def comparable(row):
    """A row as same_content compares it: no stamps, topics sorted, and the detail record without its updateDate and with its topics sorted."""
    out = normalize(row)
    del out["updated_at"], out["fetched_at"]
    out["topics"] = sorted(out["topics"])
    try:
        metadata = json.loads(out["metadata"]) if out["metadata"] else None
    except ValueError:
        return out
    if isinstance(metadata, dict):
        metadata.pop("updateDate", None)
        if isinstance(metadata.get("topics"), list):
            metadata["topics"] = sorted(metadata["topics"], key=lambda topic: json.dumps(topic, sort_keys=True))
    out["metadata"] = metadata
    return out


def same_content(stored, fetched, comparable=comparable):
    """Whether a fetch returned what is stored, apart from the stamps and the order of topics. Congress.gov re-stamps some products every hour without changing them (measured September 25, 2026: 26 of 26 such re-fetches returned the same record and the same PDF or HTML file, 3 of them with topics reordered), and rewriting their partitions added about 100 MB to the Hub repo's history each time. A stored row without text never counts as the same, so its text retry still advances fetched_at."""
    return bool(stored and stored.get("text")) and comparable(stored) == comparable(fetched)


def sync_partition(ctx, source, manifest, partition):
    """Bring one partition up to date. Returns False when the budget ran out first."""
    entry = manifest["partitions"].get(partition.key) or {}
    failures = manifest["failures"]
    now = utcnow()
    if entry.get("complete") and entry.get("fingerprint") == partition.fingerprint and not ctx.refetch and not retry_due(manifest, partition.key, entry, now):
        return True
    if not ctx.claimed:
        flush(ctx, manifest, f"{ctx.writer} takes the writer lease")
    stored = {row["id"]: row for row in ctx.store.read_partition(partition.key)} if entry else {}
    missing = sorted(uid for uid in stored if uid not in partition.units)
    if len(missing) > max(MIN_REMOVED_GUARD, MAX_REMOVED_SHARE * len(stored)):
        log.warning("%s: the listing lacks %d of %d stored products; skipped as a suspect listing", partition.key, len(missing), len(stored))
        ctx.stats["suspect_listings"] += 1
        return True
    counts = {"fetched": 0, "failed": 0, "removed": 0, "unchanged": 0}
    # {id: updateDate} of products fetched at that updateDate and found unchanged; their rows keep an older updated_at.
    restamped = dict(entry.get("restamped") or {})
    for uid in missing:
        if source.exists(uid):
            # Pages shift while the listing is read; a product the API still serves stays as stored.
            partition.units[uid] = Unit(uid, stored[uid]["updated_at"])
        else:
            del stored[uid]
            failures.pop(uid, None)
            counts["removed"] += 1
    for uid in [uid for uid, failure in failures.items() if failure.get("partition") == partition.key and uid not in partition.units]:
        del failures[uid]

    todo = []
    for uid in sorted(partition.units):
        unit, row, failure = partition.units[uid], stored.get(uid), failures.get(uid)
        if not ctx.refetch and row is not None and (row["updated_at"] == unit.updated_at or (uid in restamped and restamped[uid] == unit.updated_at)):
            if row.get("text") or later(row["fetched_at"], TEXT_RETRY_HOURS) > now:
                continue
        if not ctx.refetch and failure and failure.get("updated_at") == unit.updated_at and failure["attempts"] >= MAX_ATTEMPTS and later(failure["at"], RETRY_AFTER_HOURS) > now:
            continue
        todo.append(unit)
    if not todo and not counts["removed"] and entry.get("complete") and entry.get("fingerprint") == partition.fingerprint:
        return True

    done, finished = set(), True
    for unit in todo if ctx.max_units is None else todo[:ctx.max_units]:
        if ctx.out_of_time():
            finished = False
            break
        try:
            row = source.fetch(unit)
        except FATAL:
            _write(ctx, manifest, partition, stored, restamped, todo, done, counts, final=False)
            raise
        except Exception as error:  # noqa: BLE001 - recorded per product, retried on later runs
            previous = failures.get(unit.id) or {}
            attempts = previous.get("attempts", 0) + 1 if previous.get("updated_at") == unit.updated_at else 1
            failures[unit.id] = {"partition": partition.key, "updated_at": unit.updated_at, "attempts": attempts, "error": f"{type(error).__name__}: {error}"[:300], "at": utcnow()}
            counts["failed"] += 1
            log.info("%s failed (attempt %d): %s", unit.id, attempts, error)
        else:
            if same_content(stored.get(unit.id), row, ctx.comparable or comparable):
                restamped[unit.id] = unit.updated_at
                counts["unchanged"] += 1
            else:
                stored[unit.id] = dict(row, updated_at=unit.updated_at, fetched_at=utcnow())
            failures.pop(unit.id, None)
            counts["fetched"] += 1
        done.add(unit.id)
        if time.monotonic() - ctx.last_commit >= ctx.checkpoint_seconds:
            _write(ctx, manifest, partition, stored, restamped, todo, done, counts, final=False)
    _write(ctx, manifest, partition, stored, restamped, todo, done, counts, final=finished)
    return finished


def _write(ctx, manifest, partition, stored, restamped, todo, done, counts, final):
    failures = manifest["failures"]
    still_open = [unit for unit in todo if unit.id not in done or (unit.id in failures and failures[unit.id]["attempts"] < MAX_ATTEMPTS)]
    complete = final and not still_open
    rows = list(stored.values())
    entry = manifest["partitions"].setdefault(partition.key, {})
    # Otherwise the file on the Hub, or the one an earlier call staged, already holds these rows.
    if counts["fetched"] > counts["unchanged"] or counts["removed"] or not entry.get("sha256"):
        entry.update(ctx.store.stage_partition(partition.key, rows))
    # Only a stamp the listing still shows, other than the row's own, saves a fetch.
    checked = {uid: stamp for uid, stamp in sorted(restamped.items()) if uid in stored and stamp == partition.units[uid].updated_at and stamp != stored[uid]["updated_at"]}
    if checked:
        entry["restamped"] = checked
    else:
        entry.pop("restamped", None)
    textless = sorted(row["fetched_at"] for row in rows if not row.get("text"))
    entry.update({
        "listed": len(partition.units),
        "failed": sum(1 for uid, f in failures.items() if f.get("partition") == partition.key and f["attempts"] >= MAX_ATTEMPTS and uid not in stored),
        "complete": complete,
        "fingerprint": partition.fingerprint if complete else None,
        "text_retry_at": later(textless[0], TEXT_RETRY_HOURS) if textless else None,
        "updated_at": utcnow(),
    })
    manifest["updated_at"] = entry["updated_at"]
    ctx.pending.append(f"{partition.key}: {counts['fetched']} fetched, {counts['failed']} failed, {counts['removed']} removed, {entry['rows']} rows" + ("" if complete else " (partial)") + (f", {counts['unchanged']} unchanged" if counts["unchanged"] else ""))
    log.info(ctx.pending[-1])
    for key, value in counts.items():
        ctx.stats[key] += value
        counts[key] = 0
    if time.monotonic() - ctx.last_commit >= ctx.checkpoint_seconds:
        flush(ctx, manifest)
