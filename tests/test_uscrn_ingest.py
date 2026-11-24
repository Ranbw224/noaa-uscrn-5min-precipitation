"""Verify the retry policy and resume without touching the NOAA cache."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import download_full
import run_uscrn_ingest as ingest


def config_for_test():
    return {
        'dataset_id': 'resume-fixture', 'source_base_url': 'https://example.test/',
        'years': [2013], 'products': ['hourly02', 'subhourly01'], 'documentation_paths': ['README.txt'],
        'raw_directory': 'raw', 'gate0_directory': 'gate0', 'manifest_path': 'source_manifest.json',
        'download': {'attempts': 3, 'chunk_bytes': 64,
            'connect_timeout_seconds': 1, 'read_timeout_seconds': 1, 'user_agent': 'fixture'},
    }


def source_task(config, relative, body, product, year=None):
    return {'station': 'AK_Test' if year else None, 'year': year, 'product': product,
            'relative_source_path': relative, 'source_url': config['source_base_url'] + relative,
            'expected_bytes_from_inventory': len(body), 'directory_last_modified': '2026-05-14 11:55'}


class PacerSpy:
    def __init__(self):
        self.calls = 0
        self.delays = []

    def wait(self):
        self.calls += 1

    def defer(self, seconds):
        self.delays.append(seconds)


def patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, 'ROOT', tmp_path)
    monkeypatch.setattr(download_full, 'ROOT', tmp_path)


def http_response(url, status, body=b'', retry_after=None):
    response = requests.Response()
    response.url = url
    response.status_code = status
    response._content = body
    response._content_consumed = True
    response.headers['Content-Length'] = str(len(body))
    if retry_after is not None:
        response.headers['Retry-After'] = retry_after
    return response


def test_six_retries_for_transient_http_and_server_delay(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    config = config_for_test()
    body = b'verified payload\n'
    task = source_task(config, 'hourly02/2013/a.txt', body, 'hourly02', year=2013)
    statuses = [(429, '225'), (502, None), (503, None), (504, '0'), (503, None), (503, None)]
    responses = iter([http_response(task['source_url'], status, retry_after=delay) for status, delay in statuses]
                     + [http_response(task['source_url'], 200, body)])
    monkeypatch.setattr(ingest.requests, 'get', lambda *_args, **_kwargs: next(responses))
    failed = ingest.task_record(config, task)
    failed.update(state='error', prior_attempt_failures=[{'error': 'earlier attempt'}])
    pacer = PacerSpy()

    accepted = ingest.fetch_record(config, task, failed, pacer)
    assert accepted['state'] == 'downloaded'
    assert accepted['attempts'] == pacer.calls == 7
    assert pacer.delays == [225, 10, 20, 0, 80, 160]
    assert [row['http_status'] for row in accepted['prior_attempt_failures']] == [item[0] for item in statuses]
    assert accepted['previous_run_failures'] == failed['prior_attempt_failures']
    assert accepted['sha256'] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / accepted['path']).read_bytes() == body


def test_retry_after_http_date_and_local_cap():
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    assert ingest.retry_delay(12, None, now) == 180
    assert ingest.retry_delay(3, 'malformed', now) == 20
    assert ingest.retry_delay(3, format_datetime(now + timedelta(seconds=222), usegmt=True), now) == 222
    assert ingest.retry_delay(3, format_datetime(now - timedelta(seconds=1), usegmt=True), now) == 0


def test_shared_pacing_has_a_short_gap_and_respects_cooldown(monkeypatch):
    now = 25.0
    sleeps = []

    def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay

    monkeypatch.setattr(ingest.time, 'monotonic', lambda: now)
    monkeypatch.setattr(ingest.time, 'sleep', sleep)
    monkeypatch.setattr(ingest.random, 'uniform', lambda first, last: 0.5 if first <= 0.5 <= last else 0)
    pacer = ingest.RequestPacer()
    pacer.wait()
    pacer.wait()
    pacer.defer(15)
    pacer.wait()
    assert sleeps == [0.5, 15.0]


def test_resume_only_unresolved_and_preserve_success_record(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest, 'EXPECTED_STATION_YEARS', 1)
    monkeypatch.setattr(ingest, 'EXPECTED_STATIONS', 1)
    config = config_for_test()
    config_path = tmp_path / 'configuration.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    source = [source_task(config, 'hourly02/2013/a.txt', b'old data', 'hourly02', 2013),
              source_task(config, 'subhourly01/2013/a.txt', b'new data', 'subhourly01', 2013),
              source_task(config, 'README.txt', b'doc data', 'metadata')]
    records = [ingest.task_record(config, s) for s in source]
    immutable = records[0]
    immutable.update(state='downloaded', status=200, bytes=8,
                     sha256=hashlib.sha256(b'old data').hexdigest(), nested_provenance={'check': [42, None]})
    records[1].update(state='error')
    retained = deepcopy(immutable)
    for row, payload in [(immutable, b'old data'), (records[1], b'bad')]:
        path = tmp_path / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = {'dataset_id': config['dataset_id'], 'config_sha256': ingest.digest_file(config_path)[1],
                'specification_sha256_after_gate0': 'pinned', 'records': records}
    ingest.atomic_json(tmp_path / config['manifest_path'], manifest)
    seen = set()

    def send(url, **_kwargs):
        assert url != immutable['source_url']
        seen.add(url)
        data = b'new data' if '/subhourly01/' in url else b'doc data'
        return http_response(url, 200, data)

    monkeypatch.setattr(ingest.requests, 'get', send)
    monkeypatch.setattr(ingest, 'RequestPacer', PacerSpy)
    state = ingest.acquire(config, config_path, tmp_path / 'pipeline_status.json', {}, 2, manifest)
    persisted = ingest.read_json(tmp_path / config['manifest_path'])
    assert state['successful_files'] == 3 and state['pending_files'] == state['failed_files'] == 0
    assert state['retained_successful_records'] == 1 and state['scheduled_unresolved_files'] == 2
    assert seen == {item['source_url'] for item in source[1:]}
    assert persisted['records'][0] == retained
    assert (tmp_path / retained['path']).read_bytes() == b'old data'
    recovered = persisted['records'][1]
    assert (tmp_path / recovered['path']).read_bytes() == b'new data'
    assert (tmp_path / recovered['unverified_cache']['path']).read_bytes() == b'bad'
