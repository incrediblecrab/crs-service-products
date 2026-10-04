"""The CRS adapter on real API samples: rows, text renditions in order, challenges, listing pages, the probe's request."""

import json
import shutil

import pytest

from crs_products import source as source_module
from crs_products.http import Blocked, Unavailable
from crs_products.pipeline import Unit
from crs_products.source import API, CrsSource, mark, pdf_beside, product_row
from crs_products.text import pdf_text, tidy
from conftest import FakeFetcher, fixture_json

IN12740 = fixture_json("detail-crsreport-IN12740.json")
RL34480 = fixture_json("detail-crsreport-RL34480.json")
PDF = "https://www.congress.gov/crs_external_products/IN/PDF/IN12740/IN12740.3.pdf"
HTML = "https://www.congress.gov/crs_external_products/IN/HTML/IN12740.html"
RL_HTML = "https://www.congress.gov/crs_external_products/RL/HTML/RL34480.html"
needs_pdftotext = pytest.mark.skipif(not shutil.which("pdftotext"), reason="pdftotext is not installed")


def tiny_pdf(text):
    """A one-page PDF with a real text layer, built by hand so the test needs no PDF library."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def source_with(get_map, details=None):
    json_map = {f"{API}/crsreport/IN12740": IN12740, f"{API}/crsreport/RL34480": RL34480, **(details or {})}
    fetcher = FakeFetcher(json_map=json_map, get_map=get_map)
    return CrsSource(fetcher), fetcher


def test_product_row_from_a_post_with_pdf_and_html():
    row = product_row(IN12740["CRSReport"])
    assert row["id"] == "IN12740" and row["content_type"] == "Posts" and row["status"] == "Active" and row["version"] == 3
    assert row["authors"] == ["R. Corinne Blackford", "Anthony A. Cilluffo"] and row["topics"] == []
    assert row["publish_date"] == "2026-09-21" and row["url"] == "https://www.congress.gov/crs-report/IN12740"
    assert row["summary"] == tidy(IN12740["CRSReport"]["summary"]) and row["summary"].startswith("Overview\nIn a proposed rule")
    assert json.loads(row["metadata"]) == IN12740["CRSReport"]


def test_product_row_from_an_html_only_report_with_a_repeated_author():
    row = product_row(RL34480["CRSReport"])
    assert row["version"] == 11 and row["content_type"] == "Reports"
    assert row["authors"] == ["Valerie Heitshusen"], "the API lists this author twice"
    assert row["topics"] == ["Resolving Bicameral Differences & Presidential Action"]


@needs_pdftotext
def test_pdf_text_reads_a_real_text_layer_and_rejects_garbage():
    assert pdf_text(tiny_pdf("Hello CRS")) == "Hello CRS"
    assert pdf_text(b"%PDF-1.4 not really") is None


@needs_pdftotext
def test_text_comes_from_the_pdf_first():
    source, fetcher = source_with({PDF: tiny_pdf("Insight text"), HTML: b"<html><body><p>html text</p></body></html>"})
    row = source.fetch(Unit("IN12740", "x"))
    assert row["text"] == "Insight text" and row["text_source"] == "pdf" and row["text_url"] == PDF and len(row["text_sha256"]) == 64
    assert HTML not in fetcher.requests


@needs_pdftotext
@pytest.mark.parametrize("pdf", [404, b"%PDF-1.4 no text layer", b"<html>not a pdf</html>"])
def test_html_is_the_fallback_when_the_pdf_gives_no_text(pdf):
    source, _ = source_with({PDF: pdf, HTML: b"<html><body><p>First.</p><p>Second.</p></body></html>"})
    row = source.fetch(Unit("IN12740", "x"))
    assert row["text"] == "First.\n\nSecond." and row["text_source"] == "html" and row["text_url"] == HTML


def test_html_holding_characters_xml_does_not_allow_still_gives_text_and_summary():
    # lxml refuses to set a text node that holds one, so such a character in a block element failed the product, and in an inline element stayed in the text. Each becomes a space.
    source, _ = source_with({PDF: 404, HTML: b"<html><body><p>First.\x0c</p><p>Sec\x00ond <b>a\x1fb</b></p></body></html>"})
    row = source.fetch(Unit("IN12740", "x"))
    assert row["text"] == "First.\n\nSec ond a b" and row["text_source"] == "html"
    report = dict(IN12740["CRSReport"], summary="<p>Overview\x0c</p><p>In a proposed rule</p>")
    assert product_row(report)["summary"] == "Overview\n\nIn a proposed rule"


def test_a_challenged_html_rendition_gives_null_text_and_is_not_asked_again():
    source, fetcher = source_with({RL_HTML: Blocked("bot challenge"), PDF: 404, HTML: b"<p>never fetched</p>"})
    row = source.fetch(Unit("RL34480", "x"))
    assert row["text"] is None and row["text_source"] is None and row["title"]
    assert source.html_blocked
    row = source.fetch(Unit("IN12740", "x"))
    assert row["text"] is None and HTML not in fetcher.requests, "after one challenge, the run skips HTML"


def test_a_challenged_pdf_stops_the_run():
    source, _ = source_with({PDF: Blocked("bot challenge")})
    with pytest.raises(Blocked):
        source.fetch(Unit("IN12740", "x"))


RL_GUESS = "https://www.congress.gov/crs_external_products/RL/PDF/RL34480/RL34480.11.pdf"


def test_pdf_beside_takes_the_directory_from_the_html_rendition():
    # Both checked live on September 25, 2026: the first answered 200 with text though R43797's record listed only HTML; the second is the PDF 95-1135's record lists.
    assert pdf_beside("https://www.congress.gov/crs_external_products/R/HTML/R43797.html", 8) == "https://www.congress.gov/crs_external_products/R/PDF/R43797/R43797.8.pdf"
    assert pdf_beside("https://www.congress.gov/crs_external_products/RL/HTML/95-1135.html", "13") == "https://www.congress.gov/crs_external_products/RL/PDF/95-1135/95-1135.13.pdf"
    assert pdf_beside(RL_HTML, RL34480["CRSReport"]["currentVersion"]) == RL_GUESS
    for url, version in ((None, 8), (RL_HTML, None), (RL_HTML, "8a"), ("https://www.congress.gov/crs-report/R43797", 8)):
        assert pdf_beside(url, version) is None


@needs_pdftotext
def test_a_record_listing_only_html_gets_the_pdf_beside_it_first():
    source, fetcher = source_with({RL_GUESS: tiny_pdf("Enrollment text"), RL_HTML: b"<p>html text</p>"})
    row = source.fetch(Unit("RL34480", "x"))
    assert row["text"] == "Enrollment text" and row["text_source"] == "pdf" and row["text_url"] == RL_GUESS
    assert RL_HTML not in fetcher.requests


@needs_pdftotext
@pytest.mark.parametrize("guess", [404, 403, b"%PDF-1.4 no text layer", b"<html>not a pdf</html>"])
def test_a_refused_or_textless_guess_falls_back_to_the_html(guess):
    source, _ = source_with({RL_GUESS: guess, RL_HTML: b"<p>First.</p>"})
    row = source.fetch(Unit("RL34480", "x"))
    assert row["text"] == "First." and row["text_source"] == "html" and row["text_url"] == RL_HTML


def test_a_challenged_guess_stops_the_run_like_a_listed_pdf():
    source, _ = source_with({RL_GUESS: Blocked("bot challenge"), RL_HTML: b"<p>First.</p>"})
    with pytest.raises(Blocked):
        source.fetch(Unit("RL34480", "x"))


def test_no_guess_when_a_pdf_is_listed_or_the_version_is_unknown():
    source, fetcher = source_with({PDF: 404, HTML: b"<p>First.</p>"})
    source.fetch(Unit("IN12740", "x"))
    assert [url for url in fetcher.requests if "www." in url] == [PDF, HTML]
    record = {"CRSReport": dict(RL34480["CRSReport"], id="R40003", currentVersion=None)}
    source, fetcher = source_with({RL_HTML: b"<p>First.</p>"}, details={f"{API}/crsreport/R40003": record})
    source.fetch(Unit("R40003", "x"))
    assert [url for url in fetcher.requests if "www." in url] == [RL_HTML]


def test_a_server_error_fails_the_product():
    source, _ = source_with({PDF: Unavailable("HTTP 503")})
    with pytest.raises(Unavailable):
        source.fetch(Unit("IN12740", "x"))


def test_a_detail_for_another_id_or_no_detail_fails_the_product():
    source, _ = source_with({}, details={f"{API}/crsreport/R40001": RL34480, f"{API}/crsreport/R40002": None})
    with pytest.raises(RuntimeError, match="answered RL34480"):
        source.fetch(Unit("R40001", "x"))
    with pytest.raises(RuntimeError, match="no detail"):
        source.fetch(Unit("R40002", "x"))


def test_exists():
    source, _ = source_with({}, details={f"{API}/crsreport/R40002": None})
    assert source.exists("IN12740") and not source.exists("R40002")


def test_list_all_pages_by_count_and_dedups_shifted_pages(monkeypatch):
    monkeypatch.setattr(source_module, "PAGE", 2)
    items = [{"id": f"R4000{n}", "updateDate": f"2026-09-0{9 - n}T00:00:00Z"} for n in range(5)]
    # The second page repeats R40001, as when an update moves a product to the top between page reads.
    pages = {0: items[0:2], 2: [items[1], items[2]], 4: items[3:5]}

    def listing(params):
        return {"CRSReports": pages.get(params["offset"], [])[: params["limit"]], "pagination": {"count": 5}}

    source = CrsSource(FakeFetcher(json_map={f"{API}/crsreport": listing}))
    head, found = source.list_all()
    assert head["count"] == 5 and head["newest"] == "R40000@2026-09-09T00:00:00Z"
    assert head["listing"]["complete"] is True and head["listing"]["duplicates"] == 1
    assert sorted(found) == ["R40000", "R40001", "R40002", "R40003", "R40004"]
    assert source.head() == {key: head[key] for key in ("count", "newest")}


def test_the_probe_sends_exactly_the_listing_first_request():
    """Measured September 25, 2026: at 08:35Z limit=1 answered IN12689@2026-09-25T04:24:41Z while limit=250 answered IN12689@2026-09-25T04:09:54Z, and at 08:20Z the other way round (03:25:27Z against 04:09:54Z). A probe asking for one item saw a change that no listing published, so the local loop relisted every product every 51 seconds."""
    stale = [{"id": "IN12689", "updateDate": "2026-09-25T04:09:54Z"}, {"id": "R48859", "updateDate": "2026-09-25T04:09:51Z"}]
    fresh = [{"id": "IN12689", "updateDate": "2026-09-25T04:24:41Z"}, stale[1]]
    asked = []

    def listing(params):
        asked.append(dict(params))
        snapshot = stale if params["limit"] == source_module.PAGE else fresh
        return {"CRSReports": snapshot[params["offset"]:][: params["limit"]], "pagination": {"count": 2}}

    source = CrsSource(FakeFetcher(json_map={f"{API}/crsreport": listing}))
    head, _ = source.list_all()
    assert source.head() == {key: head[key] for key in ("count", "newest")} == {"count": 2, "newest": "IN12689@2026-09-25T04:09:54Z"}
    assert asked[-1] == asked[0]


def test_short_listings_retry_without_unioning_different_passes(monkeypatch):
    monkeypatch.setattr(source_module, "PAGE", 2)
    passes = 0

    def listing(params):
        nonlocal passes
        if params["offset"] == 0:
            passes += 1
        ids = ["R40001", "R40001"] if passes == 1 else ["R40002", "R40003"]
        return {"CRSReports": [{"id": uid, "updateDate": "same"} for uid in ids], "pagination": {"count": 2}}

    source = CrsSource(FakeFetcher(json_map={f"{API}/crsreport": listing}))
    head, found = source.list_all()
    assert set(found) == {"R40002", "R40003"}
    assert head["listing"]["complete"] is True and head["listing"]["attempts"] == 2


def test_persistently_short_listing_is_bounded_and_reported():
    fetcher = FakeFetcher(json_map={f"{API}/crsreport": {"CRSReports": [{"id": "R40001"}, {"id": "R40001"}], "pagination": {"count": 2}}})
    head, found = CrsSource(fetcher).list_all()
    assert set(found) == {"R40001"}
    assert head["listing"]["complete"] is False and head["listing"]["duplicates"] == 1
    assert head["listing"]["attempts"] == source_module.LISTING_ATTEMPTS
    assert len(fetcher.requests) <= 2 * source_module.LISTING_ATTEMPTS


def test_a_count_change_during_pagination_never_claims_completeness(monkeypatch):
    monkeypatch.setattr(source_module, "PAGE", 2)

    def listing(params):
        ids = ["R40001", "R40002"] if params["offset"] == 0 else ["R40003"]
        return {"CRSReports": [{"id": uid} for uid in ids], "pagination": {"count": 3 if params["offset"] == 0 else 4}}

    head, found = CrsSource(FakeFetcher(json_map={f"{API}/crsreport": listing})).list_all()
    assert len(found) == head["count"] == 3
    assert head["listing"]["complete"] is False and head["listing"]["counts_stable"] is False


def test_invalid_listing_is_not_an_empty_success():
    with pytest.raises(ValueError, match="listing"):
        CrsSource(FakeFetcher(json_map={f"{API}/crsreport": {}})).list_all()


def test_exists_rejects_a_detail_for_a_different_product():
    source, _ = source_with({}, details={f"{API}/crsreport/R40001": RL34480})
    with pytest.raises(RuntimeError, match="answered RL34480"):
        source.exists("R40001")


def test_exists_does_not_treat_a_malformed_detail_as_a_removed_product():
    source, _ = source_with({}, details={f"{API}/crsreport/R40001": {}})
    with pytest.raises(ValueError, match="detail response"):
        source.exists("R40001")
