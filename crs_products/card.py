"""Renders the dataset card (README.md on the Hub) from the manifest alone, so it is staged with every manifest commit and never disagrees with it."""

import json
import re
from collections import defaultdict

from .pipeline import MAX_ATTEMPTS, PROBE_KEY, RETRY_AFTER_HOURS, TEXT_RETRY_HOURS, probe_state
from .store import SCHEMA
from .source import listing_problem
from .verify import TOLERANCE_MIN

GITHUB = "https://github.com/incrediblecrab/crs-service-products"
REPO_ID = "incrediblecrab/crs-research-papers"
COLUMN_DOCS = {
    "id": "CRS product number, for example R49359, IN12740 or 98-684",
    "content_type": "Reports, Posts, Resources, Testimony or Infographics, as the API names them",
    "status": "Active or Archived",
    "version": "The API's currentVersion; only the current version is kept",
    "title": "Title",
    "authors": "Distinct author names, in the API's order",
    "topics": "CRS topic names, in no fixed order (the API's order varies between requests); often empty",
    "publish_date": "Publication date (YYYY-MM-DD)",
    "updated_at": "The listing's updateDate when this row was written. A changed value triggers a re-fetch; one that finds the product unchanged keeps the row and records the new value in `manifest.json` instead, so this can be older than the listing's",
    "url": "The product's page on congress.gov",
    "summary": "The API's summary, as plain text",
    "text": "Full text of the current version: the PDF's text layer, else the HTML rendition's text; null when neither gave text",
    "text_source": "pdf or html; null when text is null",
    "text_url": "The URL the text was extracted from; for a record that lists only HTML, possibly the PDF at the path the HTML implies",
    "text_sha256": "SHA-256 of the bytes fetched from text_url",
    "metadata": "The API's detail record as JSON: formats, related bills and laws, authors as listed",
    "fetched_at": "When the fetch that wrote this row ran (UTC); a later fetch that found the product unchanged is not recorded",
}


def series_of(key):
    match = re.match(r"[A-Z]+", key)
    return match.group(0) if match else key


def size_category(rows):
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")):
        if rows < limit:
            return label
    return "1M<n<10M"


def render(manifest):
    manifest = manifest or {}
    entries = manifest.get("partitions") or {}
    failures = manifest.get("failures") or {}
    listing = manifest.get("listing") or {}
    seen = manifest.get("seen") or listing
    rows = sum(entry.get("rows") or 0 for entry in entries.values())
    with_text = sum(entry.get("text_rows") or 0 for entry in entries.values())
    exhausted = sum(1 for f in failures.values() if f["attempts"] >= MAX_ATTEMPTS)
    count = seen.get("count")
    lines = ["---", "pretty_name: US Congressional Research Service Products", "license: other", "license_name: us-government-works",
             "license_link: https://www.copyright.gov/title17/92chap1.html#105", "language:", "- en",
             "task_categories:", "- text-generation", "- summarization", "- text-classification",
             "tags:", "- legal", "- government", "- congress", "- public-policy", "- united-states", "- crs-reports",
             "size_categories:", f"- {size_category(rows)}"]
    if entries:
        lines += ["configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/*.parquet"]
    state = probe_state(manifest)
    if state:
        lines.append(f"{PROBE_KEY}: {json.dumps(state, sort_keys=True)}")
    lines += ["---", "", "# US Congressional Research Service Products", ""]
    lines += [
        "Every product of the Congressional Research Service (CRS), the research service of the United States Congress, that the [Congress.gov API](https://api.congress.gov) lists, active and archived, with full text and metadata: Reports, Posts, Resources, Testimony and Infographics, as the API names them. CRS writes them for Members of Congress; Congress.gov publishes them.",
        "",
        f"Nothing here is edited by hand. The pipeline, its tests and its schedule are in [{GITHUB.removeprefix('https://')}]({GITHUB}), and this card is rendered from `manifest.json` in the same commit.",
        "",
        "## Status",
        "",
    ]
    if count:
        lines.append(f"**{rows:,} of {count:,} products** ({rows / count:.1%}) as of {seen.get('at')} UTC, when the API last listed {count:,}. {with_text:,} rows have text. {exhausted:,} products failed {MAX_ATTEMPTS} times and are retried every {RETRY_AFTER_HOURS} hours; their errors are in `manifest.json`.")
    else:
        lines.append("The first sync has not listed the API yet.")
    if listing.get("at"):
        lines += ["", f"Last complete sync: {listing['at']} UTC."]
    source_listing = manifest.get("source_listing")
    if source_listing and (problem := listing_problem(source_listing)):
        lines += ["", f"**Degraded:** {problem}. No product rows were changed by this attempt; the last complete sync is retained."]
    lines += ["", "| Series | Rows | With text | Listed | Partitions complete |", "|---|---:|---:|---:|---:|"]
    series = defaultdict(lambda: [0, 0, 0, 0, 0])
    for key, entry in entries.items():
        total = series[series_of(key)]
        total[0] += entry.get("rows") or 0
        total[1] += entry.get("text_rows") or 0
        total[2] += entry.get("listed") or 0
        total[3] += 1 if entry.get("complete") else 0
        total[4] += 1
    for name in sorted(series, key=lambda s: -series[s][2]):
        r, t, n, c, p = series[name]
        lines.append(f"| {name} | {r:,} | {t:,} | {n:,} | {c} of {p} |")
    if not series:
        lines.append("| (none yet) | 0 | 0 | 0 | 0 of 0 |")
    lines += [
        "",
        "*Listed* counts the products in partitions a sync has reached; partitions not reached yet are not in the table.",
        "",
        "## Use",
        "",
        "```python",
        "from datasets import load_dataset",
        f'crs = load_dataset("{REPO_ID}", split="train")',
        "```",
        "",
        "```sql",
        f"-- DuckDB, straight from the Hub",
        f"SELECT id, title, publish_date FROM 'hf://datasets/{REPO_ID}/data/*.parquet' WHERE status = 'Active' ORDER BY publish_date DESC LIMIT 10;",
        "```",
        "",
        "## Files",
        "",
        "- `data/{partition}.parquet`: one row per product. A partition is the series letters plus the thousands of the number (`R49` holds R49000 to R49999); ids numbered by year, like 98-684, are in `numeric`. Rows are sorted by id.",
        "- `manifest.json`: per partition, the row count, SHA-256, rows with text and whether it is complete; failed products with their errors; the last listing; the last 20 runs.",
        "",
        "## Schema",
        "",
        "| Column | Type | Description |",
        "|---|---|---|",
    ]
    for column in SCHEMA:
        lines.append(f"| `{column.name}` | {column.type} | {COLUMN_DOCS[column.name]} |")
    lines += [
        "",
        "## How it stays current",
        "",
        "A GitHub Actions job is scheduled at 00:00 and 12:00 UTC. It asks the API for its product count and its most recently updated product: one request. When either changed, or the last full listing is a day old, the job lists every product and fetches the ones that are new or whose `updateDate` changed, removes the ones the API no longer has, and commits the changed partitions with this card. A job that runs out of time while still fetching starts the next one itself.",
        "",
        "A product whose `updateDate` changed while its record and text did not keeps its row, its partition is not rewritten, and `manifest.json` records the new `updateDate`, so the product is not fetched again until its `updateDate` next changes. Congress.gov re-stamps some products about every hour without changing them: on September 25, 2026, 26 of 26 such re-fetches of 13 products returned the same record and the same PDF or HTML file, 3 of them with topics in another order, and each rewrite of their 10 partitions added about 100 MB to this repository's history.",
        "",
        f"The check reads who wrote last and which partitions are incomplete from `{PROBE_KEY}` in this card's metadata rather than downloading `manifest.json`. The Hub counts file downloads, so checks stay out of the download count; syncs, which download files, are in it.",
        "",
        "GitHub starts scheduled jobs late, or drops them, when it is busy, and its documentation names the start of every hour, when this job is scheduled, as a busy time. On September 24, 2026, from 06:01 to 15:26 UTC, it started 1 of the 113 jobs an earlier 5-minute schedule asked for. So a new product can take more than 12 hours to appear, and the delay is not fixed. A dropped job loses nothing, because the next one reads whatever changed.",
        "",
        "The job writes with Hugging Face Trusted Publishing, so no write token is stored anywhere. Only one writer fetches at a time, because www.congress.gov asks for at most 10 requests a minute in total: the manifest records who wrote last and when, a writer waits while another's record is under 45 minutes old, and every commit names its parent commit, so a second writer's commit is refused instead of merged. `manifest.json` names each run's writer: `github-actions` for this job, `local` for the same pipeline run from a computer.",
        "",
        "## Known gaps",
        "",
        "- Only the current version of each product. Earlier versions are not kept.",
        "- `text` is the PDF's text layer, so it keeps line breaks, page headers and footers, and has no figures or table structure. When the API lists only an HTML rendition, the PDF is tried at the path the HTML implies before falling back to the HTML.",
        f"- www.congress.gov sometimes answers CRS HTML with a Cloudflare bot challenge, which this pipeline does not try to get past. A product whose only rendition was challenged has null `text`, and products without text are fetched again every {TEXT_RETRY_HOURS // 24} days.",
        f"- The API's `updateDate` values end in Z but appear to be US Eastern time. `updated_at` keeps them as given; they are compared, not converted.",
        f"- A partition may lag a live listing by up to {TOLERANCE_MIN} products between syncs.",
        "",
        "## License",
        "",
        "CRS products are works of the United States Government and are not subject to copyright in the United States ([17 U.S.C. § 105](https://www.copyright.gov/title17/92chap1.html#105)). The notice in the products themselves adds: \"However, as a CRS Report may include copyrighted images or material from a third party, you may need to obtain the permission of the copyright holder if you wish to copy or otherwise use copyrighted material.\"",
        "",
    ]
    return "\n".join(lines)
