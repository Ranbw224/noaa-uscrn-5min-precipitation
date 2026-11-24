"""List or prepare verified archives of the station disaggregation release."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'output/station_disaggregation_release_2007_2025'
FIELD_DICTIONARY = 'docs/station_disaggregation_release.md'
MANUSCRIPT_RESULTS = ROOT / 'reports/uscrn_2007_2025/manuscript_revision'
MANUSCRIPT_TABLES = {
    f'supplementary/manuscript/{name}.csv': MANUSCRIPT_RESULTS / f'{name}.csv'
    for name in (
        'overview_climate', 'overview_phase_month', 'overview_profile_duration',
        'overview_soil_dry_climate_season', 'replay_peak15_strata',
        'replay_profile_sensitivity', 'replay_proximity_sensitivity', 'station_proximity',
    )
}
OBSERVATION_ARCHIVE_RANGES = (
    (2007, 2009), (2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017),
    (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025),
)


def member_path(release, name):
    if name == FIELD_DICTIONARY:
        return ROOT / name
    path = release / name
    if name in MANUSCRIPT_TABLES and not path.is_file():
        return MANUSCRIPT_TABLES[name]
    return path


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def archive_digest(archive, name):
    with archive.open(name) as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def package_members(release, *, include_manuscript_tables=False):
    index_path = release / 'data/portable_index.json'
    index = json.loads(index_path.read_bytes())
    audit = json.loads((release / 'audit_log/validation_summary.json').read_bytes())
    if (audit['status'] != 'passed' or set(audit['assertions']) != set('123456')
            or set(audit['assertions'].values()) != {'passed'}
            or audit['portable_index_sha256'] != sha256(index_path)):
        raise ValueError('Packaging requires the validated release index')
    groups = {name: {} for name in ('core', 'precipitation', 'covariates', 'observations')}
    indexed = set()
    for row in index['records']:
        for layer, entry in row['outputs'].items():
            name = entry['path']
            path = release / name
            if name in indexed or not path.resolve().is_relative_to(release) or path.stat().st_size != entry['bytes']:
                raise ValueError(f'Missing, duplicate or changed release file: {name}')
            indexed.add(name)
            if layer == 'high_frequency':
                group = 'observations'
            elif layer in ('pairs', 'annotations'):
                group = 'precipitation' if row['variable'] == 'precipitation' else 'covariates'
            else:
                group = 'core'
            groups[group][name] = entry['sha256']
    if len(indexed) != audit['verified_indexed_data_files']:
        raise ValueError('Index member count differs from the release audit')
    supporting = ['README.md', 'data/portable_index.json', 'model_metadata.json', 'training_sample_coverage.csv']
    for folder in ('models', 'audit_log', 'provenance', 'supplementary', 'technical_validation'):
        supporting.extend(path.relative_to(release).as_posix()
                          for path in sorted((release / folder).rglob('*')) if path.is_file())
    for name in supporting:
        if name not in groups['core']:
            groups['core'][name] = sha256(release / name)
    groups['core'][FIELD_DICTIONARY] = sha256(member_path(release, FIELD_DICTIONARY))
    if include_manuscript_tables:
        groups['core'].update({
            name: sha256(member_path(release, name)) for name in MANUSCRIPT_TABLES
        })
    return groups


def extracted_inventory(release, groups):
    return {
        part: {'files': len(files), 'bytes': sum(member_path(release, name).stat().st_size for name in files)}
        for part, files in groups.items()
    }


def observation_archive_members(files):
    groups = {
        f'five_minute_observations_{first}_{last}.zip': {}
        for first, last in OBSERVATION_ARCHIVE_RANGES
    }
    by_year = {
        year: groups[f'five_minute_observations_{first}_{last}.zip']
        for first, last in OBSERVATION_ARCHIVE_RANGES
        for year in range(first, last + 1)
    }
    seen_years = set()
    for name, digest in files.items():
        parts = PurePosixPath(name).parts
        if len(parts) != 5 or parts[:2] != ('data', 'observations') or not parts[2].startswith('source_year='):
            raise ValueError(f'Unexpected observation partition: {name}')
        year = int(parts[2].removeprefix('source_year='))
        by_year[year][name] = digest
        seen_years.add(year)
    if seen_years != set(by_year) or sum(map(len, groups.values())) != len(files):
        raise ValueError('Observation archive groups do not cover the indexed release')
    return groups


def package(release, files, output, *, verification_dir=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix('.zip.building')
    with ZipFile(staging, 'w', allowZip64=True) as writer:
        for name in sorted(files):
            path = member_path(release, name)
            writer.write(path, name, compress_type=ZIP_STORED if path.suffix in ('.parquet', '.joblib') else ZIP_DEFLATED)
    with ZipFile(staging) as archive:
        if sorted(archive.namelist()) != sorted(files):
            raise ValueError('Archive member list differs from the release plan')
        for name, expected in files.items():
            if archive_digest(archive, name) != expected:
                raise ValueError(f'Archive checksum differs: {name}')
    staging.replace(output)
    receipt_root = output.parent if verification_dir is None else verification_dir
    receipt_root.mkdir(parents=True, exist_ok=True)
    report = receipt_root / output.with_suffix('.validation.json').name
    temporary_report = report.with_suffix('.json.tmp')
    receipt = {
        'status': 'passed', 'archive': output.name, 'archive_bytes': output.stat().st_size,
        'archive_sha256': sha256(output), 'verified_archive_members': len(files),
    }
    temporary_report.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    temporary_report.replace(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=RELEASE)
    parser.add_argument('--part', choices=('core', 'precipitation', 'covariates', 'observations'))
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--archive', type=Path, help='Write a local ZIP for the selected part')
    outputs.add_argument('--split-observations-to', type=Path,
                         help='Write the observation ZIPs grouped by source year into this directory')
    parser.add_argument('--include-manuscript-tables', action='store_true',
                        help='Add the frozen-release manuscript diagnostics to the core archive')
    parser.add_argument('--verification-dir', type=Path, help='Write validation receipts outside the data directory')
    args = parser.parse_args()
    if args.archive and not args.part:
        parser.error('--archive requires --part')
    release = args.release.resolve()
    members = package_members(release, include_manuscript_tables=args.include_manuscript_tables)
    if args.archive:
        package(release, members[args.part], args.archive.resolve(), verification_dir=args.verification_dir)
    elif args.split_observations_to:
        for filename, files in observation_archive_members(members['observations']).items():
            output = args.split_observations_to.resolve() / filename
            package(release, files, output, verification_dir=args.verification_dir)
            print(f'{output.name}: {len(files)} verified entries, {output.stat().st_size:,} bytes', flush=True)
    else:
        print(json.dumps(extracted_inventory(release, members), indent=2))


if __name__ == '__main__':
    main()
