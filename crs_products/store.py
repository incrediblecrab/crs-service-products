"""Where the dataset lives: a Hugging Face dataset repo, or a local directory for tests and dry runs.

The repo holds data/{partition}.parquet, manifest.json (what every partition holds, and where the last listing stood) and README.md (the dataset card, rendered from the manifest). The manifest and the card are staged together and committed with the partitions they describe, so the three cannot disagree on the Hub.

Every Hub commit names its parent (parent_commit). If anything else committed since this store last read or wrote the repo, the Hub refuses the commit and the store raises Superseded instead of overwriting the other writer's work.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RemoteEntryNotFoundError

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("content_type", pa.string()),
    ("status", pa.string()),
    ("version", pa.int32()),
    ("title", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("topics", pa.list_(pa.string())),
    ("publish_date", pa.string()),
    ("updated_at", pa.string()),
    ("url", pa.string()),
    ("summary", pa.large_string()),
    ("text", pa.large_string()),
    ("text_source", pa.string()),
    ("text_url", pa.string()),
    ("text_sha256", pa.string()),
    ("metadata", pa.large_string()),
    ("fetched_at", pa.string()),
])
COLUMNS = SCHEMA.names
MANIFEST = "manifest.json"
CARD = "README.md"
ROW_GROUP_BYTES = 64 << 20
COMMIT_ATTEMPTS = 4
RETRYABLE = (408, 429, 500, 502, 503, 504)
# Failures with no HTTP status that are still a busy Hub or network, so they are retried as well: a dropped connection or timeout, and a README check answered with a page that is not JSON. huggingface_hub 1.32.0 parses the validate-yaml body before it checks the status, so a busy Hub raises JSONDecodeError there, before anything is uploaded; the Fed scheduled run of September 27, 2026 stopped on one.
TRANSIENT = (httpx.TransportError, json.JSONDecodeError)
# Measured September 23, 2026 on a scratch dataset: a commit whose parent_commit is no longer the branch head answers 412 Precondition Failed.
CONFLICT = 412
# The Hub's docs say the repo experience degrades after a few thousand commits; history is squashed past this many.
SQUASH_AFTER_COMMITS = 1000
log = logging.getLogger("crs_products")


class Superseded(RuntimeError):
    """Another writer committed to the repo since this store last saw it."""


def partition_of(uid):
    """The series letters plus the thousands of the number (R49 holds R49000 to R49999); ids numbered by year, like 98-684, go to "numeric"."""
    match = re.fullmatch(r"([A-Z]+)(\d+)", uid or "")
    if match:
        return f"{match.group(1)}{int(match.group(2)) // 1000}"
    return "numeric" if (uid or "")[:1].isdigit() else "other"


def partition_path(key):
    return f"data/{key}.parquet"


def normalize(row):
    out = {name: row.get(name) for name in COLUMNS}
    if out["version"] is not None:
        out["version"] = int(out["version"])
    for name in ("authors", "topics"):
        out[name] = list(out[name] or [])
    if isinstance(out["metadata"], (dict, list)):
        out["metadata"] = json.dumps(out["metadata"], ensure_ascii=False, sort_keys=True)
    return out


def row_weight(row):
    """What a product row adds to its row group's size."""
    return len(row["text"] or "") + len(row["summary"] or "") + len(row["metadata"] or "")


def write_parquet(rows, path, schema=SCHEMA, prepare=normalize, weight=row_weight):
    """Rows sorted by id, zstd, content-defined chunking so an updated partition re-uploads only changed chunks. schema, prepare and weight default to the products'; another dataset passes its own (every schema has id and text)."""
    rows = sorted((prepare(row) for row in rows), key=lambda row: row["id"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, schema, compression="zstd", compression_level=9, use_content_defined_chunking=True)
    try:
        batch, size = [], 0
        for row in rows:
            batch.append(row)
            size += weight(row)
            if size >= ROW_GROUP_BYTES:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch, size = [], 0
        if batch or not rows:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()
    return {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256_file(path), "text_rows": sum(1 for row in rows if row["text"])}


def read_parquet(path):
    return pq.read_table(path).to_pylist()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass
    return total


class _Staging:
    """Scratch space for one run. Tracks the largest footprint it ever reached, which is the local-disk bound. card, if given, renders README.md from each staged manifest; write(rows, path) writes a partition (write_parquet for the products)."""

    def __init__(self, workdir=None, card=None, write=None):
        self.dir = Path(tempfile.mkdtemp(prefix="crs-products-", dir=workdir))
        self.card = card
        self.write = write or write_parquet
        self.staged = {}
        self.peak_bytes = 0

    def measure(self):
        self.peak_bytes = max(self.peak_bytes, dir_bytes(self.dir))

    def _stage(self, repo_path, text):
        local = self.dir / "stage" / repo_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text)
        self.staged[repo_path] = local
        return local

    def stage_partition(self, key, rows):
        repo_path = partition_path(key)
        local = self.dir / "stage" / repo_path
        stats = self.write(rows, local)
        self.staged[repo_path] = local
        self.measure()
        return dict(stats, file=repo_path)

    def stage_manifest(self, manifest):
        self._stage(MANIFEST, json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        if self.card:
            self._stage(CARD, self.card(manifest))

    def put_text(self, repo_path, text, message):
        self._stage(repo_path, text)
        return self.commit(message)

    def clear(self):
        for local in self.staged.values():
            local.unlink(missing_ok=True)
        self.staged = {}

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class LocalStore(_Staging):
    def __init__(self, root, workdir=None, card=None, write=None):
        super().__init__(workdir, card, write)
        self.root = Path(root)
        self.commits = []

    def read_manifest(self):
        text = self.read_text(MANIFEST)
        return json.loads(text) if text else None

    def read_card_data(self):
        text = self.read_text(CARD)
        return DatasetCard(text).data.to_dict() if text else {}

    def read_partition(self, key):
        path = self.root / partition_path(key)
        return read_parquet(path) if path.exists() else []

    def read_columns(self, repo_path, columns):
        return pq.read_table(self.root / repo_path, columns=columns).to_pydict()

    def file_sha256s(self, repo_paths):
        return {repo_path: sha256_file(self.root / repo_path) for repo_path in repo_paths if (self.root / repo_path).exists()}

    def list_files(self, prefix=""):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and str(p.relative_to(self.root)).startswith(prefix))

    def read_text(self, repo_path):
        path = self.root / repo_path
        return path.read_text() if path.exists() else None

    def commit(self, message):
        if not self.staged:
            return None
        for repo_path, local in self.staged.items():
            target = self.root / repo_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)
        self.commits.append({"message": message, "files": sorted(self.staged)})
        self.clear()
        return str(len(self.commits))


class HubStore(_Staging):
    """token=False reads anonymously (the dataset is public), which also keeps Trusted Publishing out of read-only commands."""

    def __init__(self, repo_id, workdir=None, token=None, card=None, api=None, write=None):
        self.repo_id = repo_id
        self.api = api or HfApi(token=token)
        # Before the scratch directory exists, so a Hub that cannot be reached leaves nothing behind.
        info = self.api.dataset_info(repo_id)
        self.revision = info.sha
        self.card_data = info.card_data.to_dict() if info.card_data else {}
        super().__init__(workdir, card, write)
        self.superseded = None

    def _download(self, repo_path):
        try:
            local = hf_hub_download(self.repo_id, repo_path, repo_type="dataset", revision=self.revision, local_dir=self.dir / "download", token=self.api.token)
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            return None
        self.measure()
        return Path(local)

    def read_manifest(self):
        text = self.read_text(MANIFEST)
        return json.loads(text) if text else None

    def read_card_data(self):
        """The card's front matter as the Hub parsed it, from the dataset_info call that found the revision: no file is downloaded."""
        return self.card_data

    def read_partition(self, key):
        local = self._download(partition_path(key))
        if local is None:
            return []
        try:
            return read_parquet(local)
        finally:
            local.unlink(missing_ok=True)

    def read_columns(self, repo_path, columns):
        """Reads only the named columns, by HTTP range requests, so verification never downloads the text."""
        fs = HfFileSystem(token=self.api.token)
        with fs.open(f"datasets/{self.repo_id}@{self.revision}/{repo_path}", "rb") as handle:
            return pq.read_table(handle, columns=columns).to_pydict()

    def file_sha256s(self, repo_paths):
        out = {}
        paths = list(repo_paths)
        for start in range(0, len(paths), 100):
            for info in self.api.get_paths_info(self.repo_id, paths[start:start + 100], repo_type="dataset", revision=self.revision):
                lfs = getattr(info, "lfs", None)
                if lfs is not None:
                    out[info.path] = lfs.sha256
        return out

    def list_files(self, prefix=""):
        return sorted(path for path in self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision) if path.startswith(prefix))

    def read_text(self, repo_path):
        local = self._download(repo_path)
        if local is None:
            return None
        try:
            return local.read_text()
        finally:
            local.unlink(missing_ok=True)

    def commit(self, message):
        """One atomic commit of everything staged, on top of the last commit this store saw. Retries rate limits, server errors and the TRANSIENT failures; a retry that finds its own manifest already at the head counts as landed."""
        if not self.staged:
            return None
        if self.superseded:
            raise self.superseded
        for attempt in range(COMMIT_ATTEMPTS):
            operations = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local)) for repo_path, local in sorted(self.staged.items())]
            try:
                oid = self.api.create_commit(self.repo_id, operations=operations, commit_message=message, repo_type="dataset", parent_commit=self.revision).oid
                break
            except (HfHubHTTPError, *TRANSIENT) as error:
                http = isinstance(error, HfHubHTTPError)
                status = getattr(error.response, "status_code", None) if http else None
                if status == CONFLICT:
                    oid = self._landed()
                    if oid:
                        log.warning("commit %r had already landed as %s", message[:60], oid[:12])
                        break
                    self.superseded = Superseded(f"{self.repo_id} has a commit this run did not write (its last commit was {self.revision[:12]})")
                    raise self.superseded from None
                if attempt == COMMIT_ATTEMPTS - 1 or (http and status not in RETRYABLE):
                    raise
                log.warning("commit attempt %d failed with %s; retrying", attempt + 1, f"HTTP {status}" if http else type(error).__name__)
                time.sleep(60 * (attempt + 1))
        self.revision = oid
        self.clear()
        return oid

    def _landed(self):
        """The head commit if it already holds exactly the manifest staged here, else None."""
        staged = self.staged.get(MANIFEST)
        if staged is None:
            return None
        head = self.api.dataset_info(self.repo_id).sha
        data = staged.read_bytes()
        for info in self.api.get_paths_info(self.repo_id, [MANIFEST], repo_type="dataset", revision=head):
            lfs = getattr(info, "lfs", None)
            same = lfs.sha256 == hashlib.sha256(data).hexdigest() if lfs else getattr(info, "blob_id", None) == git_blob_sha1(data)
            if same:
                return head
        return None
