"""Launch or resume archive acquisition and full-row base validation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from threading import Lock
import time
from urllib.parse import urljoin
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests

from download_full import (ROOT, atomic_json, digest_file, download_one, inventory_year,
                           manifest_index, read_json, request_settings, task_record, update_summary, utc_now)
from probe_uscrn_sources import DEFAULT_CONFIG, documentation


NCEI_LISTING_ZONE = ZoneInfo('America/New_York')
SUCCESS = ('downloaded', 'cache_verified')
RETRYABLE_HTTP = {429, 502, 503, 504}
MIN_DOWNLOAD_RETRIES = 6
MAX_BACKOFF_SECONDS = 180
EXPECTED_STATION_YEARS = 3226
EXPECTED_STATIONS = 237
TASK_KEYS = ('station', 'product', 'year', 'filename', 'relative_source_path', 'source_url',
             'expected_bytes_from_inventory', 'directory_last_modified', 'probe_role')


def gate_receipt(config, manifest=None):
    gate = ROOT / 'config/source_gate'
    receipt = read_json(gate / 'decision.json')
    if receipt['status'] != 'passed' or receipt['stage'] != 'gate0_step0a':
        raise ValueError('Complete Gate 0a and backfill the specification before starting ingestion')
    probe = gate / 'probe_summary.json'
    if digest_file(probe)[1] != receipt['probe_summary_sha256']:
        raise ValueError('Gate 0a evidence changed after the decision')
    pinned = receipt['specification_sha256_after_backfill']
    if not re.fullmatch(r'[0-9a-f]{64}', pinned):
        raise ValueError('Gate 0a is missing the frozen specification checksum')
    if manifest is not None and manifest['specification_sha256_after_gate0'] != pinned:
        raise ValueError('The source manifest does not match the frozen Gate 0a decision')
    return receipt


def collect_inventory(config, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda year: inventory_year(config, year), config['years']))
    pairs = sum(record['paired_station_count'] for record in records)
    unique = len({site for record in records for site in record['paired_stations']})
    if (pairs, unique) != (EXPECTED_STATION_YEARS, EXPECTED_STATIONS):
        raise ValueError(f'Fresh archive inventory changed from frozen counts: {pairs} station-years, {unique} sites')
    tasks = [task for record in records for task in record['download_tasks']]
    for item in tasks:
        if not item['expected_bytes_from_inventory'] or not item['directory_last_modified']:
            raise ValueError(f'Missing archive size or modification time: {item["source_url"]}')
        datetime.fromisoformat(item['directory_last_modified'])
    tasks.extend(documentation(config, item) for item in config['documentation_paths'])
    return records, tasks


class RequestPacer:
    """Space request starts across workers and enforce a shared server cooldown."""

    def __init__(self):
        self.lock = Lock()
        self.next_start = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                remaining = self.next_start - now
                if remaining <= 0:
                    self.next_start = now + random.uniform(0.35, 0.85)
                    return
            time.sleep(remaining)

    def defer(self, seconds):
        with self.lock:
            self.next_start = max(self.next_start, time.monotonic() + seconds)


def retry_delay(attempt, retry_after, now):
    if retry_after:
        value = retry_after.strip()
        if re.fullmatch(r'[0-9]+', value):
            return int(value)
        try:
            until = parsedate_to_datetime(value)
            if until.tzinfo is not None:
                return max(0.0, (until - now).total_seconds())
        except (ValueError, OverflowError):
            pass
    return min(MAX_BACKOFF_SECONDS, 5 * 2 ** (attempt - 1))


def verify_source_metadata(result):
    if result['state'] not in SUCCESS or result['year'] is None:
        return result
    if result['bytes'] != result['expected_bytes_from_inventory']:
        return {**result, 'state': 'error', 'status': None, 'error': 'Source size differs from inventory'}
    stamp = result.get('http_last_modified')
    if stamp:
        listed = datetime.fromisoformat(result['directory_last_modified']).replace(tzinfo=NCEI_LISTING_ZONE)
        got = parsedate_to_datetime(stamp).astimezone(timezone.utc)
        result['last_modified_check'] = (
            'match_to_minute' if abs((got - listed.astimezone(timezone.utc)).total_seconds()) < 60
            else 'listed_and_http_date_differ')
    else:
        result['last_modified_check'] = 'listing_timestamp_only'
    return result


def save_download(config, record, path, response):
    if response.status_code != 200 or 'text/html' in response.headers.get('Content-Type', '').lower():
        raise ValueError('Expected a complete HTTP 200 raw data response')
    advertised = response.headers.get('Content-Length')
    expected = record['expected_bytes_from_inventory']
    if advertised is not None and expected is not None and int(advertised) != expected:
        raise ValueError(f'HTTP Content-Length differs from inventory: {advertised} != {expected}')

    temporary = path.with_name(path.name + '.' + uuid4().hex + '.part')
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open('xb') as handle:
            for block in response.iter_content(chunk_size=config['download']['chunk_bytes']):
                if block:
                    size += len(block)
                    digest.update(block)
                    handle.write(block)
            if not size or (advertised is not None and size != int(advertised)) or (
                    expected is not None and size != expected):
                raise ValueError(f'Invalid raw byte count: received {size}, HTTP {advertised}, inventory {expected}')
            handle.flush()
            os.fsync(handle.fileno())
        result = verify_source_metadata({
            **record, 'state': 'downloaded', 'status': 200, 'bytes': size, 'sha256': digest.hexdigest(),
            'retrieved_at_utc': utc_now(), 'checked_at_utc': utc_now(),
            'request_url': record['source_url'], 'response_url': response.url,
            'http_etag': response.headers.get('ETag'), 'http_last_modified': response.headers.get('Last-Modified'),
            'response_content_type': response.headers.get('Content-Type'), 'response_content_length': advertised,
            'inventory_size_matches': expected in (None, size),
        })
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        temporary.rename(path)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def fetch_record(config, task, previous, pacer):
    record = task_record(config, task)
    path = ROOT / record['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    history = list(previous.get('previous_run_failures', ()))
    if previous['state'] == 'error':
        history.extend(previous.get('prior_attempt_failures', ()))
    limit = max(MIN_DOWNLOAD_RETRIES + 1, config['download']['attempts'])
    for attempt in range(1, limit + 1):
        pacer.wait()
        http_status = header = None
        try:
            with requests.get(record['source_url'], stream=True, **request_settings(config)) as response:
                http_status = response.status_code
                header = response.headers.get('Retry-After')
                response.raise_for_status()
                result = save_download(config, record, path, response)
            return {**result, 'attempts': attempt, 'prior_attempt_failures': failures,
                    'previous_run_failures': history}
        except (requests.RequestException, ValueError, OSError) as error:
            failure = {'attempt': attempt, 'at_utc': utc_now(), 'request_url': record['source_url'],
                       'http_status': http_status, 'retry_after': header,
                       'error': f'{type(error).__name__}: {error}'}
            failures.append(failure)
            retryable = (http_status in RETRYABLE_HTTP if isinstance(error, requests.HTTPError)
                         else isinstance(error, (requests.RequestException, ValueError)))
            if not retryable:
                break
            delay = retry_delay(attempt, header, datetime.now(timezone.utc))
            # A server cooldown also applies to other queued requests.
            if attempt < limit or header:
                failure['cooldown_seconds'] = delay
                pacer.defer(delay)
    return {**record, 'state': 'error', 'checked_at_utc': utc_now(), 'request_url': record['source_url'],
            'attempts': len(failures), 'prior_attempt_failures': failures, 'previous_run_failures': history,
            'error': failures[-1]['error']}


def ingest_task(config, task, previous, probe, pacer):
    path = ROOT / previous['path']
    prior = probe if probe and probe.get('sha256') else None
    if not path.exists():
        return fetch_record(config, task, previous, pacer)

    cached = verify_source_metadata(download_one(config, task, prior))
    if cached['state'] in SUCCESS:
        if prior:
            cached['probe_manifest_source'] = 'gate0_step0a'
        return cached

    # Only the unresolved entries reach this branch. Preserve invalid bytes for inspection.
    quarantined = path.with_name(path.name + '.unverified.' + uuid4().hex)
    path.rename(quarantined)
    result = fetch_record(config, task, previous, pacer)
    result['unverified_cache'] = {'path': quarantined.relative_to(ROOT).as_posix(),
                                  'bytes': quarantined.stat().st_size, 'error': cached['error']}
    return result


def validate_manifest(config, config_path, manifest):
    if manifest['dataset_id'] != config['dataset_id'] or manifest['config_sha256'] != digest_file(config_path)[1]:
        raise ValueError('The existing manifest is bound to a different release configuration')
    records = manifest['records']
    documents = {r['relative_source_path'] for r in records if r['year'] is None}
    observations = {(r['station'], r['year'], r['product']) for r in records if r['year'] is not None}
    expected_products = {(station, year, product) for station, year, _ in observations
                         for product in config['products']}
    if (len(records) != EXPECTED_STATION_YEARS * 2 + len(config['documentation_paths'])
            or documents != set(config['documentation_paths']) or observations != expected_products
            or len(observations) != EXPECTED_STATION_YEARS * 2
            or len({station for station, _, _ in observations}) != EXPECTED_STATIONS
            or {year for _, year, _ in observations} != set(config['years'])
            or len({r['source_url'] for r in records}) != len(records)):
        raise ValueError('The existing manifest does not contain the frozen observation and metadata inventory')
    raw_root = (ROOT / config['raw_directory']).resolve()
    for row in records:
        path = ROOT / row['path']
        resolved = path.resolve()
        relative = row['relative_source_path']
        if (not resolved.is_relative_to(raw_root) or resolved != (raw_root / relative).resolve()
                or row['source_url'] != urljoin(config['source_base_url'], relative)):
            raise ValueError(f'Unexpected manifest source or destination: {relative}')
        if row['state'] not in (*SUCCESS, 'pending', 'error'):
            raise ValueError(f'Unknown manifest state for {relative}: {row["state"]}')
        if row['state'] in SUCCESS and (row['status'] != 200 or not path.is_file()
                or path.stat().st_size != row['bytes'] or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'] or '')
                or row['expected_bytes_from_inventory'] not in (None, row['bytes'])):
            raise ValueError(f'Previously verified source now disagrees with its existing success record: {relative}')


def acquire(config, config_path, state_file, receipt, workers, manifest):
    manifest_path = ROOT / config['manifest_path']
    if manifest is None:
        inventories, tasks = collect_inventory(config, workers)
        manifest = {'schema_version': 1, 'dataset_id': config['dataset_id'],
                    'config_path': config_path.relative_to(ROOT).as_posix(),
                    'config_sha256': digest_file(config_path)[1],
                    'specification_sha256_after_gate0': receipt['specification_sha256_after_backfill'],
                    'selection': config['station_selection'],
                    'inventory_paths': [f'{config["inventory_directory"]}/inventory_{r["year"]}.json' for r in inventories],
                    'records': [task_record(config, item) for item in tasks]}
        update_summary(manifest)
        atomic_json(manifest_path, manifest)
    validate_manifest(config, config_path, manifest)
    records = manifest['records']
    waiting = [i for i, row in enumerate(records) if row['state'] not in SUCCESS]
    waiting.sort(key=lambda i: records[i]['state'] != 'error')
    probes = manifest_index(ROOT / config['gate0_directory'] / 'probe_manifest.json')
    if waiting:
        update_summary(manifest)
    state = {'dataset_id': config['dataset_id'], 'pid': os.getpid(), 'stage': 'download',
             'status': 'running', 'gate0_receipt': 'config/source_gate/decision.json',
             'manifest': config['manifest_path'], 'download_workers': workers,
             'retained_successful_records': len(records) - len(waiting), 'scheduled_unresolved_files': len(waiting),
             **manifest['summary']}
    atomic_json(state_file, state)
    print(f'NOAA acquisition: {len(records) - len(waiting)} verified records retained, '
          f'{len(waiting)} unresolved files scheduled with {workers} workers', flush=True)
    pacer = RequestPacer()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for position in waiting:
            previous = records[position]
            item = {key: previous[key] for key in TASK_KEYS if key in previous}
            url = item['source_url']
            job = pool.submit(ingest_task, config, item, previous, probes.get(url), pacer)
            jobs[job] = position
        for done, job in enumerate(as_completed(jobs), 1):
            index = jobs[job]
            try:
                record = job.result()
            except Exception as error:
                record = {**records[index], 'state': 'error', 'checked_at_utc': utc_now(),
                          'error': f'{type(error).__name__}: {error}'}
            records[index] = record
            if done % 12 == 0 or record['state'] == 'error' or done == len(jobs):
                update_summary(manifest)
                atomic_json(manifest_path, manifest)
                state.update(manifest['summary'])
                atomic_json(state_file, state)
                print(f'{state["successful_files"]}/{len(records)} verified, '
                      f'{state["failed_files"]} failed, {state["pending_files"]} pending', flush=True)
    if not manifest['summary']['complete']:
        raise ValueError(f'Source manifest has {state["failed_files"]} failed files')
    return state


@contextmanager
def exclusive_run(state_file):
    """Keep one acquisition or base builder in charge of the release at a time."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.with_suffix('.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('A archive acquisition or base build already holds the pipeline lock') from error
        yield


def run(config_path, workers):
    config_path = config_path.resolve()
    config = read_json(config_path)
    state_file = ROOT / config['base_reports_directory']
    state_file = state_file.parent / 'pipeline_status.json'
    with exclusive_run(state_file):
        run_locked(config, config_path, state_file, workers)


def run_locked(config, config_path, state_file, workers):
    manifest_path = ROOT / config['manifest_path']
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    receipt = gate_receipt(config, manifest)
    try:
        state = acquire(config, config_path, state_file, receipt, workers, manifest)
        state.update(stage='base_build', status='running', base_directory=config['base_directory'],
                     base_reports_directory=config['base_reports_directory'])
        atomic_json(state_file, state)
        subprocess.run([
            sys.executable, str(ROOT / 'scripts/build_full_dataset.py'),
            '--source-manifest', str(ROOT / config['manifest_path']), '--config', str(config_path),
            '--output-root', str(ROOT / config['base_directory']),
            '--reports-dir', str(ROOT / config['base_reports_directory']), '--workers', '4',
        ], cwd=ROOT, check=True)
        built = read_json(ROOT / config['base_reports_directory'] / 'full_build_manifest.json')
        if (built['summary']['successful_pairs'] != EXPECTED_STATION_YEARS or built['summary']['failed_pairs']
                or built['summary']['unpaired_or_unverified']):
            raise ValueError('The full base audit did not cover exactly 3,226 verified pairs')
        state.update(stage='base_build', status='passed', base_build_summary=built['summary'])
        atomic_json(state_file, state)
    except BaseException as error:
        current = read_json(state_file) if state_file.is_file() else {}
        current.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error))
        atomic_json(state_file, current)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    run(args.config, args.workers)
