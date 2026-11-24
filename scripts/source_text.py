"""Conservative, byte-traceable handling of NUL prefixes in NOAA ASCII files.

Original files are never modified. Only a leading run of NUL bytes followed by
an otherwise complete, valid ASCII source record may be omitted in memory.
No records, timestamps, or meteorological values are reconstructed.
"""
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re


class SourceTextError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _record_tokens(payload, expected_fields):
    text = payload.decode('ascii')
    tokens = text.split()
    if len(tokens) != expected_fields:
        raise ValueError(f'expected {expected_fields} fields, found {len(tokens)}')
    if not re.fullmatch(r'\d{5}', tokens[0]):
        raise ValueError('first token is not a five-digit WBAN')
    for date, clock in ((tokens[1], tokens[2]), (tokens[3], tokens[4])):
        if not re.fullmatch(r'\d{8}', date) or not re.fullmatch(r'\d{1,4}', clock):
            raise ValueError('invalid source date/time token shape')
        datetime.strptime(date + clock.zfill(4), '%Y%m%d%H%M')
    type_index = 13 if expected_fields == 23 else 19
    if tokens[type_index] not in ('C', 'R', 'U'):
        raise ValueError('unknown surface-temperature type')
    for i, token in enumerate(tokens):
        if i in (0, 1, 2, 3, 4, 5, type_index):
            continue
        if not math.isfinite(float(token)):
            raise ValueError(f'nonfinite, undocumented numeric value at field {i+1}')
    return tokens


def read_clean_source(path, expected_fields):
    """Return ASCII text and audit; reject interior NULs or malformed records.

    Byte offsets are zero based in the original file; raw line numbers are one
    based and refer to LF-delimited binary lines. Existing LF/CRLFs are retained.
    A NUL prefix does not identify how many records were lost; missing timestamps
    must be determined separately from observed source endpoints.
    """
    if expected_fields not in (23, 38):
        raise ValueError('Only documented USCRN 23/38-field products are supported')
    path = Path(path)
    raw = path.read_bytes()
    audit = {'path': str(path), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
             'raw_bytes': len(raw), 'expected_fields': expected_fields,
             'raw_binary_lines': 0, 'clean_record_count': 0,
             'nul_byte_count': raw.count(b'\x00'), 'leading_nul_bytes_removed': 0,
             'leading_nul_runs': [], 'unrecoverable_anomalies': [],
             'raw_file_modified': False, 'records_imputed': 0,
             'policy': 'Remove only leading NUL runs before independently complete ASCII records; never reconstruct lost records.'}
    if not raw:
        raise SourceTextError(f'Empty source: {path}', audit)
    pieces, offset = [], 0
    # Explicit LF splitting avoids treating non-newline control characters as
    # row delimiters. The final record need not have a terminating newline.
    lines = raw.split(b'\n')
    if lines[-1] == b'':
        lines.pop()
    for number, body in enumerate(lines, 1):
        ending = b'\n' if offset + len(body) < len(raw) else b''
        binary_line = body + ending
        audit['raw_binary_lines'] += 1
        prefix = len(body) - len(body.lstrip(b'\x00'))
        payload = body[prefix:]
        try:
            if b'\x00' in payload:
                raise ValueError('interior NUL bytes; automatic interpretation forbidden')
            # A full structural check is essential for any prefixed record.
            # Ordinary rows require field counts and ASCII; scientific parsing
            # and timestamp validation are additionally performed by the caller.
            if prefix:
                tokens = _record_tokens(payload, expected_fields)
            else:
                tokens = payload.decode('ascii').split()
                if len(tokens) != expected_fields:
                    raise ValueError(f'expected {expected_fields} fields, found {len(tokens)}')
        except (ValueError, UnicodeError) as exc:
            anomaly = {'raw_line_number': number, 'byte_offset': offset,
                       'raw_line_bytes': len(binary_line), 'leading_nul_count': prefix,
                       'reason': str(exc)}
            audit['unrecoverable_anomalies'].append(anomaly)
            raise SourceTextError(f'{path}: raw line {number}, byte {offset}: {exc}', audit) from exc
        if prefix:
            audit['leading_nul_runs'].append({'raw_line_number': number, 'byte_offset': offset,
                'nul_count': prefix, 'retained_payload_byte_offset': offset + prefix,
                'retained_wban': tokens[0], 'retained_utc_date': tokens[1],
                'retained_utc_time': tokens[2], 'retained_payload_sha256': hashlib.sha256(payload).hexdigest()})
            audit['leading_nul_bytes_removed'] += prefix
        pieces.append(payload + ending)
        audit['clean_record_count'] += 1
        offset += len(binary_line)
    return b''.join(pieces).decode('ascii'), audit
