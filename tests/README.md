# tests

Offline tests: `pip install '.[test]'`, then `python -m pytest -q`. No network or key; PDF tests are skipped without `pdftotext`.

**Objective:** pin down the behavior an unattended pipeline depends on. Each load-bearing check was also tested by planting the defect it should catch and confirming that a test fails.

**Inputs:** real API responses in [`fixtures/`](fixtures/README.md), and scripted fakes.

**Files:**

- `conftest.py`: a scripted CRS source, and a fetcher that serves fixtures by URL.
- `test_pipeline.py`: the sync loop: first sync, idle runs, changes, re-stamps, removals, retries, the suspect-listing guard, resumption, the writer lease, the probe's decision.
- `test_source.py`: rows from real API records, text renditions, bot challenges, listing pages.
- `test_summaries.py`: summaries against a fake API that reorders ties: complete reads where offset paging skips, the daily check, removals.
- `test_constitution.py`: Constitution Annotated ids, listing, rows, a sync.
- `test_http.py`: pacing, challenges, quota exhaustion, missing keys, outages.
- `test_store.py`: partition keys, Parquet round trips, an empty partition written without a row group, the commit fence and which failures it retries, no scratch left behind.
- `test_verify.py`: each planted data defect is named; exit 1.
- `test_card.py`: the products card's front matter and numbers.
- `test_cli.py`: exit codes, `$GITHUB_OUTPUT`, Trusted Publishing, dataset routing, and every workflow's commands, options, outputs and schedule.
