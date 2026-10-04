"""python -m crs_products {run,probe,card,verify,squash} [--dataset products|summaries|constitution]: sync a dataset, decide whether a sync is needed, render the card, check the Hub, squash history."""

import argparse
import json
import logging
import os
import shutil
import time

import httpx
from huggingface_hub.errors import HfHubHTTPError

from . import constitution, summaries
from .http import Fetcher, QuotaExhausted, Unavailable
from .pipeline import CLEAN_STOPS, PROBE_KEY, PROBE_STATE_VERSION, Context, decide, probe_state, sync, utcnow, writer_identity
from .source import CrsSource
from .store import CARD, SQUASH_AFTER_COMMITS

DEFAULT_REPO = "incrediblecrab/crs-research-papers"
REPOS = {"products": DEFAULT_REPO, "summaries": summaries.REPO_ID, "constitution": constitution.REPO_ID}
# What the Hub answers a Trusted Publishing exchange for a repo that has no publisher registered (measured September 23, 2026 in workflow run 35935093031).
NO_PUBLISHER = "No trusted publisher configured"

def card_of(dataset):
    if dataset == "summaries":
        from .summaries_card import render
    elif dataset == "constitution":
        from .constitution_card import render
    else:
        from .card import render
    return render


def open_store(args, write=False):
    """Reads are anonymous (token=False): the dataset is public, and a read-only command then never asks for a Trusted Publishing token."""
    from .store import HubStore, LocalStore

    card = card_of(args.dataset)
    writer = {"summaries": summaries.write, "constitution": constitution.write}.get(args.dataset)
    if args.local:
        return LocalStore(args.local, workdir=args.workdir, card=card, write=writer)
    return HubStore(args.repo, workdir=args.workdir, token=None if write else False, card=card, write=writer)


def github_output(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")


def warn(message, level="warning"):
    print(f"::{level}::{message}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"{level}: {message}")


def transient(error):
    """Outages that say nothing about the data: the next scheduled probe tries again."""
    if isinstance(error, (Unavailable, QuotaExhausted, httpx.TransportError)):
        return True
    status = getattr(getattr(error, "response", None), "status_code", None)
    return isinstance(error, HfHubHTTPError) and status is not None and (status >= 500 or status == 429)


def cmd_run(args):
    if args.dataset != "summaries" and not shutil.which("pdftotext"):
        raise SystemExit("pdftotext is missing: install poppler (brew install poppler, or apt-get install poppler-utils)")
    try:
        store = open_store(args, write=True)
    except HfHubHTTPError as error:
        if NO_PUBLISHER in str(error):
            # A failed run is one GitHub can email the owner about; a warning would let a dataset stop updating unnoticed.
            warn(f"{NO_PUBLISHER} for {args.repo}, so nothing was written. Register the workflow under the dataset's Settings > Trusted Publishers.", level="error")
            github_output(commits=0, more="false")
            return 1
        raise
    fetcher = Fetcher()
    started = time.monotonic()
    only = frozenset(key.strip() for key in args.partitions.split(",") if key.strip()) if args.partitions else None
    deadline = started + args.budget_minutes * 60
    if args.dataset == "summaries":
        ctx = Context(store=store, deadline=deadline, only=only, refetch=args.refetch, partition_of=summaries.partition_of, comparable=summaries.comparable, source_url=summaries.SOURCE_URL)
        runner, source = summaries.sync, summaries.SummariesSource(fetcher)
    elif args.dataset == "constitution":
        ctx = Context(store=store, deadline=deadline, only=only, max_units=args.max_units, refetch=args.refetch, partition_of=constitution.partition_of, comparable=constitution.comparable, source_url=constitution.SOURCE_URL)
        runner, source = sync, constitution.ConanSource(fetcher)
    else:
        ctx = Context(store=store, deadline=deadline, only=only, max_units=args.max_units, refetch=args.refetch)
        runner, source = sync, CrsSource(fetcher)
    try:
        run = runner(ctx, source)
    finally:
        store.close()
        fetcher.close()
    run.update(minutes=round((time.monotonic() - started) / 60, 1), peak_scratch_bytes=store.peak_bytes, requests=dict(sorted(fetcher.requests.items())))
    print(json.dumps(run, indent=1))
    # more: the budget ran out while the run was still making progress, so another run should start without waiting for the schedule.
    github_output(commits=run["commits"], more="true" if run["stopped"] == "budget" and run["fetched"] > 0 else "false")
    for message in run.get("warnings", []):
        warn(message)
    if run["stopped"] == "incomplete listing":
        return 0
    if run["stopped"] in CLEAN_STOPS + ("deferred", "superseded"):
        return 0
    warn(f"run stopped: {run['stopped']}")
    return 1


def read_probe_state(store):
    """probe_state() as the card carries it, which costs no file download; from the manifest when the card has none of this version (a card rendered before it carried one). Returns (state, where it came from)."""
    state = store.read_card_data().get(PROBE_KEY)
    if isinstance(state, dict) and state.get("version") == PROBE_STATE_VERSION:
        return state, "card"
    return probe_state(store.read_manifest()), "manifest"


def cmd_probe(args):
    """One API request and one read of the repo's metadata: prints the decision, and writes needed=true|false to $GITHUB_OUTPUT."""
    fetcher = Fetcher()
    try:
        store = open_store(args)
        try:
            state, origin = read_probe_state(store)
        finally:
            store.close()
        head = CrsSource(fetcher).head()
    except Exception as error:  # noqa: BLE001 - transient outages are a skipped probe, not a failed job
        if not transient(error):
            raise
        warn(f"probe skipped: {type(error).__name__}: {error}")
        github_output(needed="false")
        return 0
    finally:
        fetcher.close()
    needed, reason = decide(state, head, writer_identity())
    print(json.dumps({"needed": needed, "reason": reason, "head": head, "state_from": origin, "at": utcnow()}))
    github_output(needed="true" if needed else "false")
    return 0


def cmd_card(args):
    """Re-renders the card from the manifest, for a card change that should not wait for the next sync."""
    store = open_store(args, write=True)
    try:
        text = card_of(args.dataset)(store.read_manifest())
        if store.read_text(CARD) == text:
            print("card unchanged")
        else:
            store.put_text(CARD, text, "Update dataset card")
            print("card updated")
    finally:
        store.close()
    return 0


def cmd_verify(args):
    from .verify import verify

    store = open_store(args)
    fetcher = Fetcher() if args.live else None
    try:
        if args.dataset == "summaries":
            report = verify(store, summaries.SummariesSource(fetcher) if fetcher else None, partition_of=summaries.partition_of, tally="bill_type", live=summaries.live_counts)
        elif args.dataset == "constitution":
            report = verify(store, constitution.ConanSource(fetcher) if fetcher else None, partition_of=constitution.partition_of, tally="kind")
        else:
            report = verify(store, CrsSource(fetcher) if fetcher else None)
    finally:
        store.close()
        if fetcher:
            fetcher.close()
    print(json.dumps(report, indent=1))
    for message in report.get("warnings", []):
        warn(message)
    return 1 if report["problems"] else 0


def cmd_squash(args):
    """Replaces the repo's history with one commit once it is longer than --min-commits: every sync commits, and a rewritten partition's old chunks stay referenced by history until it is squashed."""
    from huggingface_hub import HfApi

    api = HfApi()
    commits = len(api.list_repo_commits(args.repo, repo_type="dataset"))
    if commits <= args.min_commits:
        print(f"{commits} commits; not squashed (threshold {args.min_commits})")
        return 0
    api.super_squash_history(args.repo, repo_type="dataset", commit_message=f"Squash history of {commits} commits ({utcnow()})")
    print(f"squashed {commits} commits in {args.repo}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crs_products", description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text):
        # No abbreviated options: the workflow must spell every option out, so adding an option cannot change what an old abbreviation meant.
        sub = commands.add_parser(name, help=help_text, allow_abbrev=False)
        sub.add_argument("--dataset", choices=sorted(REPOS), default="products", help="which dataset (default products)")
        target = sub.add_mutually_exclusive_group()
        target.add_argument("--repo", help=f"Hugging Face dataset repo (default: the dataset's own, {DEFAULT_REPO} for the products)")
        target.add_argument("--local", help="use a local directory instead of the Hub (tests and dry runs)")
        sub.add_argument("--workdir", help="parent directory for scratch files (default: system temp)")
        sub.set_defaults(handler=handler)
        return sub

    run = add("run", cmd_run, "sync within a time budget")
    run.add_argument("--budget-minutes", type=float, default=320.0)
    run.add_argument("--partitions", help="comma-separated partition keys to sync, for example R49,IN12 or 119-hr; others are skipped (smoke tests)")
    run.add_argument("--max-units", type=int, help="fetch at most this many units per partition (smoke tests; not for summaries)")
    run.add_argument("--refetch", action="store_true", help="fetch every unit of the partitions reached again, as after a parser change; for summaries, read every slice again in full")
    add("probe", cmd_probe, "decide from one API request whether a products sync is needed")
    add("card", cmd_card, "re-render README.md on the Hub from the manifest")
    add("verify", cmd_verify, "check the files against the manifest; --live also against the API's listing").add_argument("--live", action="store_true")
    add("squash", cmd_squash, "squash the Hub repo's history into one commit").add_argument(
        "--min-commits", type=int, default=SQUASH_AFTER_COMMITS, help=f"squash only a longer history (default {SQUASH_AFTER_COMMITS})")

    args = parser.parse_args(argv)
    args.repo = args.repo or REPOS[args.dataset]
    if args.command == "squash" and args.local:
        parser.error("squash works on the Hub only")
    if args.command == "probe" and args.dataset != "products":
        parser.error("only the products dataset has a probe")
    if args.command == "run" and args.dataset == "summaries" and args.max_units is not None:
        parser.error("--max-units does not apply to summaries, which are read a slice at a time")
    if not args.local and os.environ.get("GITHUB_ACTIONS") == "true":
        # Trusted Publishing: huggingface_hub trades the job's OIDC id token for a short-lived token scoped to this repo.
        os.environ.setdefault("HF_OIDC_RESOURCE", f"datasets/{args.repo}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    return args.handler(args)
