"""Downloader integrity checks using a local HTTP server, never NOAA transfers."""
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("download_full", ROOT / "scripts/download_full.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SourceHandler(BaseHTTPRequestHandler):
    responses = {}
    counts = {}
    requested_urls = []
    recover_on_query = {}

    def do_GET(self):
        key = urlsplit(self.path).path
        type(self).counts[key] = type(self).counts.get(key, 0) + 1
        type(self).requested_urls.append(self.path)
        code, data, length = self.responses.get(self.path, self.responses[key])
        if urlsplit(self.path).query and key in self.recover_on_query:
            code, data, length = 200, self.recover_on_query[key], None
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("ETag", '"fixture-version-1"')
        self.send_header("Last-Modified", "Sun, 01 Nov 2026 00:00:00 GMT")
        self.send_header("Content-Length", str(length if length is not None else len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class DownloadIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def setUp(self):
        (ROOT / "reports").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="_download_test_", dir=ROOT / "reports")
        self.directory = Path(self.temporary.name)
        self.config = json.loads((ROOT / "config/uscrn_2007_2025.json").read_text(encoding="utf-8"))
        self.config["source_base_url"] = self.base
        self.config["raw_directory"] = self.directory.relative_to(ROOT).as_posix()
        self.config["download"].update(connect_timeout_seconds=1, read_timeout_seconds=1, chunk_bytes=4)
        SourceHandler.responses = {}
        SourceHandler.counts = {}
        SourceHandler.requested_urls = []
        SourceHandler.recover_on_query = {}

    def tearDown(self):
        self.temporary.cleanup()

    def task(self, name="fixture.txt", content=b"12345 20240101 0005\n"):
        SourceHandler.responses["/" + name] = (200, content, None)
        return {"relative_source_path": name, "source_url": self.base + name,
                "station": "TEST", "product": "subhourly01", "year": 2024,
                "expected_bytes_from_inventory": len(content)}

    def test_streamed_download_and_cache_keep_original_provenance(self):
        task = self.task()
        first = module.download_one(self.config, task, None)
        self.assertEqual(first["state"], "downloaded")
        self.assertEqual(first["sha256"], hashlib.sha256(b"12345 20240101 0005\n").hexdigest())
        self.assertEqual(first["http_etag"], '"fixture-version-1"')
        self.assertEqual(first["request_url"], task["source_url"])
        datetime.fromisoformat(first["retrieved_at_utc"])
        cached = module.download_one(self.config, task, first)
        self.assertEqual(cached["state"], "cache_verified")
        self.assertEqual(cached["retrieved_at_utc"], first["retrieved_at_utc"])
        self.assertEqual(SourceHandler.counts["/fixture.txt"], 1)

    def test_tampered_cache_is_not_silently_accepted_or_overwritten(self):
        task = self.task()
        first = module.download_one(self.config, task, None)
        destination = ROOT / first["path"]
        destination.write_bytes(b"tampered")
        result = module.download_one(self.config, task, first)
        self.assertEqual(result["state"], "error")
        self.assertEqual(destination.read_bytes(), b"tampered")

    def test_truncated_responses_retry_and_remove_only_owned_temporary_file(self):
        task = self.task(content=b"short")
        SourceHandler.responses["/fixture.txt"] = (200, b"short", 100)
        unrelated = self.directory / "someone_else.part"
        unrelated.write_bytes(b"keep")
        with patch.object(module.time, "sleep"):
            result = module.download_one(self.config, task, None)
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["attempts"], 3)
        self.assertFalse((self.directory / "fixture.txt").exists())
        self.assertEqual(list(self.directory.glob("*.part")), [unrelated])
        self.assertEqual(SourceHandler.counts["/fixture.txt"], 3)

    def test_directory_parser_excludes_foreign_year_and_keeps_exact_station_suffix(self):
        listing = b'<table><tr><td><a href="CRNH0203-2022-AK_Name_1_NE.txt">file</a></td><td>2026-01-01 01:00</td><td align="right">42</td></tr><tr><td><a href="CRNH0203-2021-WRONG.txt">file</a></td><td>today</td><td>99</td></tr></table>'
        SourceHandler.responses["/hourly02/2022/"] = (200, listing, None)
        result = module.fetch_directory(self.config, "hourly02", 2022)
        self.assertEqual(list(result["files"]), ["AK_Name_1_NE"])
        self.assertEqual(result["files"]["AK_Name_1_NE"]["expected_bytes_from_inventory"], 42)

    def test_cache_busting_retry_recovers_truncated_response_and_records_urls(self):
        content = b"full observation fixture\n"
        task = self.task(content=content)
        SourceHandler.responses["/fixture.txt"] = (200, b"truncated", 100)
        SourceHandler.recover_on_query["/fixture.txt"] = content
        with patch.object(module.time, "sleep"):
            result = module.download_one(self.config, task, None)
        self.assertEqual(result["state"], "downloaded")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["source_url"], task["source_url"])
        self.assertIn("?uscrn_retry=", result["request_url"])
        self.assertEqual(result["response_url"], result["request_url"])
        self.assertEqual(result["prior_attempt_failures"][0]["request_url"], task["source_url"])
        self.assertEqual((ROOT / result["path"]).read_bytes(), content)

    def test_previous_error_uses_unique_cache_buster_immediately(self):
        task = self.task()
        previous = {"state": "error", "prior_attempt_failures": [{"error": "truncated"}]}
        result = module.download_one(self.config, task, previous)
        self.assertEqual(result["state"], "downloaded")
        self.assertEqual(result["attempts"], 1)
        self.assertIn("?uscrn_retry=", result["request_url"])
        self.assertEqual(result["previous_run_failures"], previous["prior_attempt_failures"])
        self.assertEqual(SourceHandler.counts["/fixture.txt"], 1)

    def test_manifest_replace_retries_transient_windows_permission_failure(self):
        destination = self.directory / "manifest.json"
        destination.write_text('{"old": true}', encoding="utf-8")
        original_replace = module.os.replace
        calls = []

        def transient_lock(source, target):
            calls.append((source, target))
            if len(calls) == 1:
                raise PermissionError(5, "simulated Windows reader lock")
            return original_replace(source, target)

        with patch.object(module.os, "replace", side_effect=transient_lock), patch.object(module.time, "sleep") as sleep:
            module.atomic_json(destination, {"complete": True})
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"complete": True})
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.05)
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_manifest_replace_permission_failure_is_bounded_and_preserves_old_file(self):
        destination = self.directory / "manifest.json"
        destination.write_text('{"old": true}', encoding="utf-8")
        unrelated = self.directory / "someone_else.part"
        unrelated.write_bytes(b"keep")
        with patch.object(module.os, "replace", side_effect=PermissionError(5, "persistent lock")) as replace, patch.object(module.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                module.atomic_json(destination, {"new": True})
        self.assertEqual(replace.call_count, 9)
        self.assertEqual(sleep.call_count, 8)
        self.assertAlmostEqual(sum(c.args[0] for c in sleep.call_args_list), 2.75)
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(self.directory.glob("*.part")), [unrelated])

    def test_uncheckpointed_raw_cache_keeps_unknown_original_retrieval_time(self):
        task = self.task()
        (self.directory / task["relative_source_path"]).write_bytes(b"12345 20240101 0005\n")
        result = module.download_one(self.config, task, {"state": "error"})
        self.assertEqual(result["state"], "cache_verified")
        self.assertEqual(result["cache_provenance"], "previously_unrecorded_local_file")
        self.assertIsNone(result["retrieved_at_utc"])
        self.assertIsNone(result["http_etag"])
        manifest = {"records": [result]}
        module.update_summary(manifest)
        self.assertEqual(manifest["summary"]["unknown_original_retrieval_time_files"], 1)
        self.assertEqual(SourceHandler.counts, {})


if __name__ == "__main__":
    unittest.main()
