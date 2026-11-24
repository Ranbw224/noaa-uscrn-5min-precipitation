"""Inventory NOAA products and verify local caches or public downloads."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit
from uuid import uuid4
import requests

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = 1

PRODUCT_PATTERNS = {
    "hourly02": r"^CRNH02\d{2}-{year}-(?P<station>.+)\.txt$",
    "subhourly01": r"^CRNS01\d{2}-05-{year}-(?P<station>.+)\.txt$",
}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_json(path, value):
    """Only the uniquely named temporary file created here may be removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows readers or antivirus can briefly hold the destination without
        # FILE_SHARE_DELETE. Keep the complete temporary payload and retry only
        # this atomic replacement; persistent permission failures still surface.
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(0.05 * (2 ** attempt), 0.5))
    finally:
        temporary.unlink(missing_ok=True)

def digest_file(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            total += len(chunk)
            hasher.update(chunk)
    return total, hasher.hexdigest()

class DirectoryTableParser(HTMLParser):
    """Parse NOAA's directory table without dependence on column HTML styling."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cells = None
        self.cell = None
        self.href = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells, self.href = [], None
        elif tag == "td" and self.cells is not None:
            self.cell = []
        elif tag == "a" and self.cells is not None:
            candidate = dict(attrs).get("href")
            if self.href is None:
                self.href = candidate

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.cells.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.cells is not None:
            if self.href and self.cells:
                self.rows.append((self.href, self.cells))
            self.cells, self.cell, self.href = None, None, None

def request_settings(config):
    settings = config["download"]
    return {
        "timeout": (settings["connect_timeout_seconds"], settings["read_timeout_seconds"]),
        "headers": {"User-Agent": settings["user_agent"], "Accept-Encoding": "identity"},
    }

def raw_request_url(source_url, attempt, previous_failed=False):
    """Avoid reusing a potentially truncated CDN response, retaining source ID."""
    if attempt == 1 and not previous_failed:
        return source_url
    parts = urlsplit(source_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("uscrn_retry", uuid4().hex))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

def fetch_directory(config, product, year):
    url = urljoin(config["source_base_url"], f"{product}/{year}/")
    last_error = None
    for attempt in range(1, config["download"]["attempts"] + 1):
        try:
            with requests.get(url, **request_settings(config)) as response:
                response.raise_for_status()
                body = response.content
                parser = DirectoryTableParser()
                parser.feed(body.decode(response.encoding or "utf-8"))
                pattern = re.compile(PRODUCT_PATTERNS[product].replace("{year}", str(year)))
                files = {}
                for href, cells in parser.rows:
                    # Only simple filenames in this exact product/year directory.
                    name = unquote(unescape(href))
                    if Path(name).name != name or "/" in name or "\\" in name:
                        continue
                    match = pattern.fullmatch(name)
                    if not match:
                        continue
                    station = match["station"]
                    if station in files:
                        raise ValueError(f"duplicate {product}/{year} station suffix: {station}")
                    size = int(cells[2]) if len(cells) > 2 and cells[2].isdigit() else None
                    relative = f"{product}/{year}/{name}"
                    files[station] = {
                        "station": station, "product": product, "year": year,
                        "filename": name, "relative_source_path": relative,
                        "source_url": urljoin(config["source_base_url"], relative),
                        "expected_bytes_from_inventory": size,
                        "directory_last_modified": cells[1] if len(cells) > 1 else None,
                    }
                if not files:
                    raise ValueError(f"no valid station filenames in {product}/{year}")
                return {
                    "source_url": url, "retrieved_at_utc": utc_now(),
                    "http_etag": response.headers.get("ETag"),
                    "http_last_modified": response.headers.get("Last-Modified"),
                    "listing_sha256": hashlib.sha256(body).hexdigest(),
                    "listing_bytes": len(body), "files": files,
                }
        except (requests.RequestException, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < config["download"]["attempts"]:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"inventory failed: {url}: {last_error}")

def inventory_year(config, year, refresh=False):
    path = ROOT / config["inventory_directory"] / f"inventory_{year}.json"
    if path.exists() and not refresh:
        cached = read_json(path)
        if (cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("source_base_url") == config["source_base_url"]
                and cached.get("year") == year
                and cached.get("products") == config["products"]):
            return cached
    # Two small directory listings per year; station data are not fetched here.
    listings = {p: fetch_directory(config, p, year) for p in config["products"]}
    sets = {p: set(listings[p]["files"]) for p in config["products"]}
    paired = sorted(set.intersection(*sets.values()))
    if not paired:
        raise ValueError(f"no exact station pairs found in {year}")
    tasks = [listings[p]["files"][s] for s in paired for p in config["products"]]
    unknown_sizes = sum(t["expected_bytes_from_inventory"] is None for t in tasks)
    record = {
        "schema_version": SCHEMA_VERSION, "year": year,
        "source_base_url": config["source_base_url"], "products": config["products"],
        "created_at_utc": utc_now(), "selection": config["station_selection"],
        "paired_station_count": len(paired), "paired_stations": paired,
        "file_count": len(tasks),
        "expected_total_bytes": sum(t["expected_bytes_from_inventory"] or 0 for t in tasks),
        "files_with_unknown_size": unknown_sizes,
        "unpaired_stations_by_product": {p: sorted(sets[p] - set(paired)) for p in sets},
        "listings": listings, "download_tasks": tasks,
    }
    atomic_json(path, record)
    return record

def manifest_index(path):
    if not path.exists():
        return {}
    value = read_json(path)
    records = value if isinstance(value, list) else value.get("records", [])
    return {r.get("source_url", r.get("url")): r for r in records}

def task_record(config, task):
    path = ROOT / config["raw_directory"] / task["relative_source_path"]
    return {
        **task, "path": path.relative_to(ROOT).as_posix(),
        "url": task["source_url"], "state": "pending", "status": None,
        "bytes": None, "sha256": None, "retrieved_at_utc": None,
        "http_etag": None, "http_last_modified": None,
        "request_url": None, "response_url": None,
    }

def download_one(config, task, previous):
    record = task_record(config, task)
    path = ROOT / record["path"]
    if path.exists():
        total, sha256 = digest_file(path)
        old = previous if previous and previous.get("sha256") else None
        if old and old.get("sha256") and old["sha256"] != sha256:
            return {**record, "state": "error", "bytes": total, "sha256": sha256,
                    "error": "existing cache checksum differs from prior manifest; not overwritten",
                    "checked_at_utc": utc_now()}
        if (not old and task.get("expected_bytes_from_inventory") is not None
                and total != task["expected_bytes_from_inventory"]):
            return {**record, "state": "error", "bytes": total, "sha256": sha256,
                    "error": "unrecorded cache size differs from inventory; not overwritten",
                    "checked_at_utc": utc_now()}
        result = {**record, **(old or {}), **task, "path": record["path"],
                  "url": task["source_url"], "state": "cache_verified", "status": 200,
                  "bytes": total, "sha256": sha256, "checked_at_utc": utc_now(),
                  "local_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                  "cache_provenance": "known_manifest" if old else "previously_unrecorded_local_file"}
        # Never infer the original network retrieval date from a filesystem timestamp.
        result.setdefault("retrieved_at_utc", None)
        result.pop("error", None)
        return result

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    failures = []
    previous_failed = bool(previous and previous.get("state") == "error")
    request_url = None
    for attempt in range(1, config["download"]["attempts"] + 1):
        request_url = raw_request_url(task["source_url"], attempt, previous_failed)
        try:
            with requests.get(request_url, stream=True, **request_settings(config)) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"expected complete 200 response, got {response.status_code}")
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("received HTML instead of raw text data")
                hasher = hashlib.sha256()
                total = 0
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=config["download"]["chunk_bytes"]):
                        if chunk:
                            handle.write(chunk)
                            total += len(chunk)
                            hasher.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                advertised_size = response.headers.get("Content-Length")
                if total == 0:
                    raise ValueError("empty source file")
                if advertised_size and total != int(advertised_size):
                    raise ValueError(f"Content-Length mismatch: {total} != {advertised_size}")
                # No other process should write this destination during a run.
                if path.exists():
                    raise FileExistsError("destination appeared during transfer; refusing overwrite")
                temporary.rename(path)
                return {**record, "state": "downloaded", "status": response.status_code,
                        "bytes": total, "sha256": hasher.hexdigest(),
                        "retrieved_at_utc": utc_now(), "checked_at_utc": utc_now(),
                        "request_url": request_url, "response_url": response.url,
                        "http_etag": response.headers.get("ETag"),
                        "http_last_modified": response.headers.get("Last-Modified"),
                        "response_content_type": response.headers.get("Content-Type"),
                        "response_content_length": advertised_size,
                        "attempts": attempt, "prior_attempt_failures": failures,
                        "previous_run_failures": previous.get("prior_attempt_failures", []) if previous_failed else [],
                        "inventory_size_matches": task.get("expected_bytes_from_inventory") in (None, total)}
        except (requests.RequestException, ValueError, OSError) as error:
            failures.append({"attempt": attempt, "at_utc": utc_now(),
                             "request_url": request_url,
                             "error": f"{type(error).__name__}: {error}"})
        finally:
            # Delete this invocation's temporary file only, never raw cache files.
            temporary.unlink(missing_ok=True)
        if attempt < config["download"]["attempts"]:
            time.sleep(min(2 ** (attempt - 1), 4))
    return {**record, "state": "error", "checked_at_utc": utc_now(),
            "request_url": request_url,
            "attempts": len(failures), "prior_attempt_failures": failures,
            "error": failures[-1]["error"]}

def update_summary(manifest):
    records = manifest["records"]
    success = [r for r in records if r["state"] in ("downloaded", "cache_verified")]
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = {
        "planned_files": len(records), "successful_files": len(success),
        "pending_files": sum(r["state"] == "pending" for r in records),
        "failed_files": sum(r["state"] == "error" for r in records),
        "local_bytes_verified": sum(r.get("bytes") or 0 for r in success),
        "unknown_original_retrieval_time_files": sum(r.get("retrieved_at_utc") is None for r in success),
        "complete": len(success) == len(records),
    }
