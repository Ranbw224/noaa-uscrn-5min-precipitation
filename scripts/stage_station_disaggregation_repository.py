"""Prepare portable observation metadata for deposit without rewriting data pages."""

import argparse
import base64
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from struct import pack, unpack

import pyarrow as pa
import pyarrow.parquet as pq

from package_station_disaggregation_release import RELEASE, package_members, member_path, sha256


PROVENANCE = b'uscrn_provenance'
ARROW_SCHEMA = b'ARROW:schema'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact_binary(data):
    """Encode the length and bytes of a binary field in Compact Thrift."""
    length = len(data)
    prefix = bytearray()
    while length >= 128:
        prefix.append((length & 127) | 128)
        length >>= 7
    prefix.append(length)
    return bytes(prefix) + data


def replace_footer_pair(footer, key, old_value, new_value):
    # Each KeyValue struct has consecutive binary fields key (1) and value (2).
    key_field = b'\x18' + compact_binary(key) + b'\x18'
    before = key_field + compact_binary(old_value) + b'\x00'
    if footer.count(before) != 1:
        raise ValueError(f'Cannot identify the unique Parquet footer entry for {key!r}')
    return footer.replace(before, key_field + compact_binary(new_value) + b'\x00', 1)


def stage_observation(release, staging, sources, entry):
    relative = entry['path']
    source, target = release / relative, staging / relative
    payload = source.read_bytes()
    if payload[:4] != b'PAR1' or payload[-4:] != b'PAR1' or digest(payload) != entry['sha256']:
        raise ValueError(f'Invalid or changed input file: {relative}')
    footer_start = len(payload) - 8 - unpack('<I', payload[-8:-4])[0]
    data_pages, footer = payload[:footer_start], payload[footer_start:-8]
    original = pq.ParquetFile(pa.BufferReader(payload))
    metadata = original.metadata.metadata
    provenance = json.loads(metadata[PROVENANCE])
    for product, record in provenance['source_text_audit'].items():
        raw_path = '/'.join(PureWindowsPath(record['path']).parts[-3:])
        if sources[raw_path] != record['raw_sha256'] or provenance['raw_sha256'][product] != sources[raw_path]:
            raise ValueError(f'Source manifest does not match the observation provenance: {relative}')
        record['path'] = raw_path
    public_provenance = json.dumps(provenance, sort_keys=True).encode('utf-8')
    arrow_schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(metadata[ARROW_SCHEMA])))
    if arrow_schema.metadata[PROVENANCE] != metadata[PROVENANCE]:
        raise ValueError(f'Arrow and Parquet provenance disagree: {relative}')
    public_schema = arrow_schema.with_metadata({**arrow_schema.metadata, PROVENANCE: public_provenance})
    public_ipc = base64.b64encode(public_schema.serialize().to_pybytes())
    footer = replace_footer_pair(footer, PROVENANCE, metadata[PROVENANCE], public_provenance)
    footer = replace_footer_pair(footer, ARROW_SCHEMA, metadata[ARROW_SCHEMA], public_ipc)
    public_payload = data_pages + footer + pack('<I', len(footer)) + b'PAR1'

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.parquet.building')
    tmp.write_bytes(public_payload)
    public = pq.ParquetFile(tmp)
    before = original.schema_arrow.with_metadata(None)
    if (not public.schema_arrow.with_metadata(None).equals(before, check_metadata=True)
            or public.metadata.num_rows != entry['rows']
            or public.metadata.num_row_groups != original.metadata.num_row_groups
            or public.schema_arrow.metadata[PROVENANCE] != public_provenance
            or public.metadata.metadata[PROVENANCE] != public_provenance
            or public.metadata.metadata[b'pandas'] != metadata[b'pandas']
            or public.metadata.serialized_size != len(footer)):
        raise ValueError(f'Published observation failed structural verification: {relative}')
    public.close()
    with tmp.open('rb') as f:
        if f.read(footer_start) != data_pages:
            raise ValueError(f'Published observation data pages differ: {relative}')
    tmp.replace(target)
    return {
        'path': relative, 'rows': entry['rows'], 'source_sha256': entry['sha256'],
        'published_sha256': digest(public_payload), 'published_bytes': len(public_payload),
        'encoded_data_sha256': digest(data_pages),
    }


def write_json(target, data):
    temporary = target.with_suffix('.json.building')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(target)


def stage_release(release, staging):
    if staging == release or staging.is_relative_to(release):
        raise ValueError('Deposit staging must be outside the frozen release')
    packages = package_members(release)
    index_path = release / 'data/portable_index.json'
    audit_path = 'audit_log/validation_summary.json'
    index = json.loads(index_path.read_bytes())
    audit = json.loads((release / audit_path).read_bytes())
    manifest = json.loads((release / 'provenance/source_manifest.json').read_bytes())
    sources = {row['relative_source_path']: row['sha256'] for row in manifest['records']
               if row.get('relative_source_path') and row.get('sha256')}
    observations = [record['outputs']['high_frequency'] for record in index['records']
                    if 'high_frequency' in record['outputs']]

    # Unchanged assets link to the validated files. The transport copy has its own index and audit.
    for part, members in packages.items():
        for name in members:
            if part == 'observations' or name in {'data/portable_index.json', audit_path}:
                continue
            src, dst = member_path(release, name), staging / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(src, dst)
            elif not src.samefile(dst):
                raise ValueError(f'Staging already contains a different asset: {name}')

    def export(entry):
        return stage_observation(release, staging, sources, entry)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(export, observations))
    for entry, result in zip(observations, results, strict=True):
        entry['sha256'], entry['bytes'] = result['published_sha256'], result['published_bytes']

    with (staging / 'audit_log/observation_metadata_export.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    index_target = staging / 'data/portable_index.json'
    write_json(index_target, index)
    audit['observation_metadata_export'] = {
        'source_portable_index_sha256': audit['portable_index_sha256'],
        'observations_exported': len(results), 'observation_rows': sum(row['rows'] for row in results),
        'edit': 'Replace two workstation paths with source-manifest relative paths per observation file.',
        'encoded_data': 'Copied through the original Parquet footer boundary without modification.',
        'manifest': 'audit_log/observation_metadata_export.csv',
    }
    audit['portable_index_sha256'] = sha256(index_target)
    write_json(staging / audit_path, audit)
    package_members(staging)
    print(json.dumps(audit['observation_metadata_export'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staging', type=Path)
    parser.add_argument('--release', type=Path, default=RELEASE)
    args = parser.parse_args()
    stage_release(args.release.resolve(), args.staging.resolve())
