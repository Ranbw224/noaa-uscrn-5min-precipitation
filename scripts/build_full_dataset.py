"""Build provenance-preserving USCRN parquet records from verified file pairs.

The two input tracks share instruments and processing. Agreement is internal
consistency, never independent observational validation. Only actual five-minute
source rows are released; absent annual/support intervals are counted in QA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from uscrn_source_schema import read_product, good_values, VARS, SH, HH, interval_grid, characterize
from source_text import read_clean_source

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'subhourly01': re.compile(r'^CRNS0101-05-(\d{4})-(.+)\.txt$'),
    'hourly02': re.compile(r'^CRNH0203-(\d{4})-(.+)\.txt$'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    def finite(obj):
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): finite(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [finite(v) for v in obj]
        return obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(finite(value), indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def split_for_wban(wban):
    fraction = int(hashlib.sha256(('uscrn:' + wban).encode()).hexdigest(), 16) / 2**256
    return 'train' if fraction < .70 else ('validation' if fraction < .85 else 'test')


def exclusion(raw, flag=None, present=None):
    code = np.zeros(len(raw), dtype=np.int8)
    code[(raw.isna() | (raw <= -9990)).to_numpy()] = 1
    if flag is not None:
        code[(flag.notna() & (flag != 0)).to_numpy()] = 2
    if present is not None:
        code[~np.asarray(present)] = 3
    return code


def validate_source(path, subhourly, year):
    """Check actual token count before a named parser can shift extra fields."""
    width = len(SH if subhourly else HH)
    text, byte_audit = read_clean_source(path, width)
    n = byte_audit['clean_record_count']
    if not n:
        raise ValueError(f'Empty source: {path}')
    frame = read_product(path, subhourly, _clean_source=(text, byte_audit))
    if len(frame) != n or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f'Rows are lost, duplicated, or not chronological: {path}')
    unexpected = frame.index.difference(interval_grid(subhourly, year))
    if len(unexpected):
        raise ValueError(f'{path}: {len(unexpected)} timestamps outside annual endpoint grid')
    for sf, hf, sq, hq, _, _ in VARS.values():
        field = sf if subhourly else hf
        raw = pd.to_numeric(frame[field], errors='raise')
        if not np.isfinite(raw).all():
            raise ValueError(f'{path}: nonfinite unencoded source value in {field}')
    wban = frame.WBANNO.astype(str).str.zfill(5)
    if wban.nunique() != 1 or not re.fullmatch(r'\d{5}', wban.iloc[0]):
        raise ValueError(f'WBAN identity ambiguous: {path}')
    return frame, wban.iloc[0]


def quality_records(frame, subhourly, station, year, period):
    rows = characterize(frame, subhourly, station, year)
    step = '5min' if subhourly else 'h'
    support = pd.date_range(frame.index.min(), frame.index.max(), freq=step)
    annual = interval_grid(subhourly, year)
    for row in rows:
        var = row['variable']
        sf, hf, sq, hq, _, _ = VARS[var]
        values = good_values(frame, sf if subhourly else hf, sq if subhourly else hq)
        raw = frame[sf if subhourly else hf]
        flag = frame[sq if subhourly else hq] if (sq if subhourly else hq) else None
        codes = exclusion(raw, flag)
        for key in ('expected_rows', 'absent_timestamps', 'missing_or_bad_count', 'missing_or_bad_fraction', 'gap_runs', 'longest_gap_steps'):
            row[key + '_full_year'] = row.pop(key)
        row.update({
            'observed_rows': len(frame), 'expected_rows_within_observed_bounds': len(support),
            'absent_timestamps_within_observed_bounds': len(support.difference(frame.index)),
            'unobserved_intervals_before_first': int((annual < frame.index.min()).sum()),
            'unobserved_intervals_after_last': int((annual > frame.index.max()).sum()),
            'invalid_observed_count': int(values.isna().sum()),
            'invalid_observed_fraction': float(values.isna().mean()),
            'exclusion_sentinel_count': int((codes == 1).sum()),
            'exclusion_nonzero_qc_count': int((codes == 2).sum()),
            'station_split': split_for_wban(str(row['wban']).zfill(5)),
            'temporal_split': period,
        })
    return rows


def write_parquet(frame, path, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b'uscrn_provenance'] = json.dumps(provenance, sort_keys=True).encode()
    table = table.replace_schema_metadata(metadata)
    tmp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp, compression='zstd', row_group_size=100_000)
    tmp.replace(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'rows': len(frame), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def build_pair(task, output_root, fingerprint):
    station, year = task['station'], task['year']
    start = datetime.now(timezone.utc).isoformat()
    sub, wban = validate_source(ROOT / task['subhourly01']['path'], True, year)
    hr, hwban = validate_source(ROOT / task['hourly02']['path'], False, year)
    byte_audits = {'subhourly01': sub.attrs['source_text_audit'], 'hourly02': hr.attrs['source_text_audit']}
    hashes = {product: audit['raw_sha256'] for product, audit in byte_audits.items()}
    for product, digest in hashes.items():
        if task[product].get('sha256') and digest != task[product]['sha256']:
            raise ValueError(f'Raw hash mismatch for {task[product]["path"]}')
    if wban != hwban:
        raise ValueError(f'Cross-product WBAN mismatch {station}/{year}: {wban} vs {hwban}')
    station_split = split_for_wban(wban)
    period = task.get('temporal_split', 'unassigned')
    common = {'station_name': station, 'station_wban': wban, 'source_year': np.int16(year),
              'station_split': station_split, 'temporal_split': period}
    high = pd.DataFrame(index=sub.index)
    high['time_end_utc'] = sub.index
    high['time_start_utc'] = sub.index - pd.Timedelta(minutes=5)
    for col, val in common.items():
        high[col] = val
    for col in ('WBANNO', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME'):
        high[col] = sub[col]
    if task.get('retain_hydrology_staging'):
        for name in ('SOIL_MOISTURE_5', 'WIND_1_5', 'WIND_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE', 'ST_FLAG'):
            high[name] = sub[name]
    pair_frames, consistency = [], []
    support = pd.date_range(sub.index.min(), sub.index.max(), freq='5min')
    present = pd.Series(1, index=sub.index).reindex(support, fill_value=0)
    actual_count = present.resample('h', closed='right', label='right').sum()
    hour_index = actual_count.index.union(hr.index).sort_values()
    for var, (sf, hf, sq, hq, units, tolerance) in VARS.items():
        fine = good_values(sub, sf, sq)
        high[var] = fine
        high[var + '_source_value'] = sub[sf]
        if sq:
            high[var + '_source_qc'] = sub[sq].astype(np.int16)
        high[var + '_exclusion_code'] = exclusion(sub[sf], sub[sq] if sq else None)
        fine = fine.reindex(support)
        groups = fine.resample('h', closed='right', label='right')
        count = groups.count()
        complete_fine = count == 12
        derived = groups.sum(min_count=12) if var == 'precipitation' else groups.mean().where(complete_fine)
        ramp = fine.diff().where(fine.index.ceil('h') == fine.index.shift(-1, freq='5min').ceil('h')).abs()
        # On a bounded regular support, each retained ramp links adjacent five-minute intervals within one hour.
        ramp_groups = ramp.resample('h', closed='right', label='right')
        pair = pd.DataFrame(index=hour_index)
        pair['time_end_utc'] = hour_index
        pair['time_start_utc'] = hour_index - pd.Timedelta(hours=1)
        for col, val in common.items():
            pair[col] = val
        pair['variable'] = var
        pair['units'] = units
        pair['WBANNO'] = hr.WBANNO.reindex(hour_index).astype('string')
        pair['CRX_VN'] = hr.CRX_VN.reindex(hour_index).astype('string')
        pair['LONGITUDE'] = hr.LONGITUDE.reindex(hour_index)
        pair['LATITUDE'] = hr.LATITUDE.reindex(hour_index)
        pair['official_source_present'] = hour_index.isin(hr.index)
        pair['official_source_value'] = hr[hf].reindex(hour_index)
        flags = hr[hq].reindex(hour_index) if hq else None
        pair['official_source_qc'] = flags.astype('Int16') if hq else pd.Series(pd.NA, index=hour_index, dtype='Int16')
        pair['official_exclusion_code'] = exclusion(pair.official_source_value, flags, pair.official_source_present)
        pair['official_hourly'] = good_values(hr, hf, hq).reindex(hour_index)
        pair['derived_hourly'] = derived.reindex(hour_index)
        pair['fine_record_count'] = actual_count.reindex(hour_index, fill_value=0).astype(np.int16)
        pair['valid_five_minute_count'] = count.reindex(hour_index, fill_value=0).astype(np.int16)
        pair['complete_pair'] = (pair.valid_five_minute_count == 12) & pair.official_hourly.notna() & pair.derived_hourly.notna()
        pair['discrepancy_derived_minus_official'] = pair.derived_hourly - pair.official_hourly
        pair['five_minute_min'] = groups.min().where(complete_fine).reindex(hour_index)
        pair['five_minute_max'] = groups.max().where(complete_fine).reindex(hour_index)
        pair['five_minute_mean_abs_ramp'] = ramp_groups.mean().where(complete_fine).reindex(hour_index)
        pair['five_minute_max_abs_ramp'] = ramp_groups.max().where(complete_fine).reindex(hour_index)
        pair['rain_max_fraction'] = (pair.five_minute_max / pair.derived_hourly).where(pair.derived_hourly > 0) if var == 'precipitation' else np.nan
        good = pair.loc[pair.complete_pair, 'discrepancy_derived_minus_official'].abs()
        consistency.append({'station': station, 'station_wban': wban, 'year': year, 'variable': var,
                            'eligible_hours': len(good), 'total_official_hours': len(hr),
                            'mean_abs_difference': float(good.mean()), 'max_abs_difference': float(good.max()),
                            'diagnostic_tolerance': tolerance, 'above_tolerance_count': int((good > tolerance).sum()),
                            'above_tolerance_fraction': float((good > tolerance).mean()),
                            'validation_scope': 'related_product_internal_consistency'})
        pair_frames.append(pair.reset_index(drop=True))
    pairs = pd.concat(pair_frames, ignore_index=True)
    provenance = {'builder_fingerprint': fingerprint, 'raw_sha256': hashes, 'source_text_audit': byte_audits,
                  'station': station, 'year': year,
                  'interval_convention': '(time_start_utc,time_end_utc]', 'validation_scope': 'related_product_internal_consistency'}
    partition = Path(f'year={year}') / f'station={station}'
    outputs = {
        'high_frequency': write_parquet(high.reset_index(drop=True), output_root / 'high_frequency' / partition / 'observations.parquet', provenance),
        'hourly_pairs': write_parquet(pairs, output_root / 'hourly_pairs' / partition / 'pairs.parquet', provenance),
    }
    if task.get('retain_hydrology_staging'):
        columns = [f'SOIL_MOISTURE_{depth}' for depth in (5, 10, 20, 50, 100)]
        columns += ['SUR_TEMP', 'SUR_TEMP_TYPE', 'SUR_TEMP_FLAG', 'SUR_TEMP_MAX', 'SUR_TEMP_MAX_FLAG',
                    'SUR_TEMP_MIN', 'SUR_TEMP_MIN_FLAG']
        hydrology = hr[columns].assign(time_end_utc=hr.index, station_wban=wban, source_year=np.int16(year))
        outputs['hourly_hydrology'] = write_parquet(
            hydrology.reset_index(drop=True),
            output_root / 'hourly_hydrology' / partition / 'observations.parquet', provenance)
    return {'station': station, 'station_wban': wban, 'year': year, 'status': 'success',
            'started_at_utc': start, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'builder_fingerprint': fingerprint, 'raw_sha256': hashes, 'source_text_audit': byte_audits, 'outputs': outputs,
            'station_split': station_split, 'temporal_split': period,
            'quality': (quality_records(sub, True, station, year, period)
                        + quality_records(hr, False, station, year, period)),
            'consistency': consistency}


def discover_records(manifest, years):
    records = manifest if isinstance(manifest, list) else manifest.get('records', [])
    groups = {}
    for record in records:
        relative = record.get('path')
        if not relative:
            continue
        name = Path(relative).name
        for product, pattern in PATTERNS.items():
            match = pattern.fullmatch(name)
            if not match:
                continue
            year, station = int(match[1]), match[2]
            if year not in years:
                continue
            if not re.fullmatch(r'[A-Z]{2}_[A-Za-z0-9_.-]+', station):
                raise ValueError(f'Unexpected station path token: {station}')
            group = groups.setdefault((station, year), {'station': station, 'year': year})
            if record.get('status') == 200 and (ROOT / relative).is_file():
                if product in group:
                    raise ValueError(f'Duplicate source records for {station}/{year}/{product}')
                group[product] = record
    tasks = [v for v in groups.values() if 'subhourly01' in v and 'hourly02' in v]
    unpaired = [{'station': v['station'], 'year': v['year'],
                 'missing_verified_products': [p for p in PATTERNS if p not in v]} for v in groups.values()
                if 'subhourly01' not in v or 'hourly02' not in v]
    return sorted(tasks, key=lambda t: (t['year'], t['station'])), unpaired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, default=ROOT / 'data/uscrn_2007_2025/source_manifest.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/uscrn_2007_2025.json')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/uscrn_2007_2025/staging/base')
    parser.add_argument('--reports-dir', type=Path, default=ROOT / 'reports/uscrn_2007_2025/base')
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--limit', type=int, help='Build only the first N verified pairs for a smoke test')
    parser.add_argument('--force', action='store_true', help='Rebuild even when all source/output hashes match')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    years = args.years or config['years']
    source_manifest_bytes = args.source_manifest.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    tasks, unpaired = discover_records(source_manifest, years)
    retain_hydrology = bool(config.get('retain_hydrology_staging', False))
    checkpoint_pairs = int(config.get('build_checkpoint_pairs', 1))
    if checkpoint_pairs < 1:
        parser.error('build_checkpoint_pairs must be positive')
    configured_periods = {year: item['label'] for item in config.get('temporal_periods', [])
                          for year in range(item['start_year'], item['end_year'] + 1)}
    if configured_periods and set(years) - set(configured_periods):
        parser.error('temporal_periods must cover every requested year')
    for task in tasks:
        task['retain_hydrology_staging'] = retain_hydrology
        if configured_periods:
            task['temporal_split'] = configured_periods[task['year']]
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        tasks = tasks[:args.limit]
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    target = args.reports_dir / 'full_build_manifest.json'
    prior = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    fingerprint_parts = [sha256(Path(__file__)), sha256(Path(__file__).with_name('uscrn_source_schema.py')),
                         sha256(Path(__file__).with_name('source_text.py')),
                         sha256(args.config) if args.config.exists() else 'no_config']
    fingerprint = hashlib.sha256('|'.join(fingerprint_parts).encode()).hexdigest()
    records = {(r['station'], r['year']): r for r in prior.get('records', [])}
    queued, resumed = [], 0
    for task in tasks:
        old = records.get((task['station'], task['year']))
        reuse = old and old.get('status') == 'success' and old.get('builder_fingerprint') == fingerprint and not args.force
        if reuse:
            try:
                reuse = all(sha256(ROOT / task[p]['path']) == old['raw_sha256'][p] == task[p].get('sha256') for p in PATTERNS)
                reuse = reuse and all((ROOT / out['path']).is_file() and sha256(ROOT / out['path']) == out['sha256'] for out in old['outputs'].values())
                reuse = reuse and all(str((ROOT / out['path']).resolve()).startswith(str(args.output_root.resolve())) for out in old['outputs'].values())
            except (OSError, KeyError):
                reuse = False
        if reuse:
            resumed += 1
        else:
            queued.append(task)
    selected = {(t['station'], t['year']) for t in tasks}

    def save():
        current = [records[key] for key in sorted(selected, key=lambda k: (k[1], k[0])) if key in records]
        summary = {'verified_matched_pairs': len(tasks), 'resumed_pairs': resumed,
                   'successful_pairs': sum(r.get('status') == 'success' for r in current),
                   'failed_pairs': sum(r.get('status') == 'error' for r in current), 'unpaired_or_unverified': len(unpaired)}
        dump_json(target, {'schema_version': 1, 'builder_fingerprint': fingerprint,
                          'source_manifest_sha256': source_manifest_sha256, 'summary': summary,
                          'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                          'unpaired_or_unverified_sources': unpaired, 'records': current})
        return current, summary

    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_pair, task, args.output_root, fingerprint): task for task in queued}
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'station': task['station'], 'year': task['year'], 'status': 'error',
                          'error': str(exc), 'traceback': traceback.format_exc(), 'builder_fingerprint': fingerprint}
            records[(task['station'], task['year'])] = result
            if completed % checkpoint_pairs == 0 or result['status'] == 'error' or completed == len(futures):
                _, summary = save()
                print(json.dumps({'station': task['station'], 'year': task['year'], 'status': result['status'], **summary}), flush=True)
    current, summary = save()
    successful = [r for r in current if r.get('status') == 'success']
    for filename, key in [('full_quality_profile.csv', 'quality'), ('full_cross_resolution_consistency.csv', 'consistency')]:
        pd.DataFrame([row for record in successful for row in record[key]]).to_csv(args.reports_dir / filename, index=False)
    dump_json(args.output_root / 'dataset_schema.json', {
        'schema_version': 1, 'layout': ['high_frequency/year=YYYY/station=SOURCE_NAME/observations.parquet',
                                        'hourly_pairs/year=YYYY/station=SOURCE_NAME/pairs.parquet']
        + (['hourly_hydrology/year=YYYY/station=SOURCE_NAME/observations.parquet'] if retain_hydrology else []),
        'interval_convention': '(time_start_utc,time_end_utc]',
        'high_frequency_support': 'Actual source rows only. No annual padding, no gap filling.',
        'source_byte_policy': 'Original raw bytes and SHA256 are retained. Only leading NUL runs before structurally valid complete ASCII records are omitted in memory; raw line numbers and byte offsets are included in each source_text_audit. Interior NULs and malformed rows cause errors. Lost rows are never reconstructed; grid gaps describe absent timestamps without inferring a record count from NUL byte count.',
        'hourly_pairs_support': 'Union of actual hourly records and hours within the observed five-minute support; fine_record_count and source-present fields identify absent observations.',
        'exclusion_codes': {'0': 'accepted', '1': 'source missing sentinel', '2': 'nonzero source QC (takes priority over sentinel)', '3': 'source row absent (hourly only; high frequency is unpadded)'},
        'missing_sentinels': {'air_temperature': -9999, 'precipitation': -9999, 'solar_radiation': -99999, 'relative_humidity': -9999},
        'variables': {v: {'fine_source_field': x[0], 'hourly_source_field': x[1], 'units': x[4], 'aggregation': 'sum' if v == 'precipitation' else 'arithmetic_mean'} for v, x in VARS.items()},
        'derived_statistics': 'Min/max and absolute consecutive five-minute ramps are descriptive and require 12 valid intervals; rain_max_fraction=max(five-minute precipitation)/hourly derived total, only if total>0. No extreme-event labels.',
        'input_tracks': {'official_hourly': 'Published NOAA hourly product after documented QA', 'derived_hourly': 'Exact sum/mean of the released five-minute published observations, only for 12 valid intervals'},
        'feature_leakage_warning': 'five_minute_min, five_minute_max, five_minute_mean_abs_ramp, five_minute_max_abs_ramp, rain_max_fraction, and discrepancy_derived_minus_official are computed from target observations and MUST NOT be aggregate-only reconstruction model inputs. Fine-validity eligibility must be set before evaluation; test target values may only be used for scoring.',
        'split_definition': {'station_split': 'SHA256 of UTF-8 uscrn:WBANNO, interpreted as unsigned integer / 2**256; <0.70 train, <0.85 validation, else test', 'temporal_split': configured_periods or {'2022': 'train', '2023': 'train', '2024': 'validation', '2025': 'test'}, 'caution': 'Orthogonal tags; station and time are not claimed independent. Select the intended protocol explicitly.'},
        'validation_scope': 'Cross-product agreement is related-product internal consistency, not independent physical validation.',
        'builder_fingerprint': fingerprint,
    })
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary['failed_pairs'] or not tasks else 0


if __name__ == '__main__':
    sys.exit(main())
