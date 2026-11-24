"""Focused fixtures for the conservative NOAA source-byte adapter."""
from pathlib import Path
import tempfile
import unittest
from source_text import read_clean_source, SourceTextError


def record(hour='0100', newline=b'\n'):
    values = ['12345', '20250101', hour, '20241231', '2000', '2.623', '-111.0', '32.0'] + ['0'] * 30
    values[19] = 'C'
    return ' '.join(values).encode('ascii') + newline


class SourceTextTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.txt'
            path.write_bytes(data)
            result = read_clean_source(path, 38)
            self.assertEqual(path.read_bytes(), data)
            return result

    def test_valid_lf_and_crlf_preserved(self):
        raw = record() + record('0200', b'\r\n')
        text, audit = self.parse(raw)
        self.assertEqual(text.encode('ascii'), raw)
        self.assertEqual(audit['clean_record_count'], 2)
        self.assertEqual(audit['nul_byte_count'], 0)

    def test_leading_nuls_traced_without_imputing(self):
        first = record()
        raw = first + b'\x00' * 244 + record('0300')
        text, audit = self.parse(raw)
        self.assertEqual(text.encode('ascii'), first + record('0300'))
        run = audit['leading_nul_runs'][0]
        self.assertEqual((run['byte_offset'], run['raw_line_number'], run['nul_count']), (len(first), 2, 244))
        self.assertEqual(audit['records_imputed'], 0)
        self.assertEqual(audit['clean_record_count'], 2)

    def test_interior_nul_rejected(self):
        with self.assertRaises(SourceTextError):
            self.parse(record().replace(b'12345', b'12\x00345'))

    def test_missing_field_rejected_even_without_nul(self):
        with self.assertRaises(SourceTextError):
            self.parse(b' '.join(record().split()[:-1]) + b'\n')

    def test_bad_prefixed_identity_rejected(self):
        with self.assertRaises(SourceTextError):
            self.parse(b'\x00' + record().replace(b'12345', b'xxxxx'))

    def test_nul_only_tail_rejected(self):
        with self.assertRaises(SourceTextError):
            self.parse(record() + b'\x00' * 244)


if __name__ == '__main__':
    unittest.main()
