"""Summarize the frozen M2 reports and verify endpoint and hydrology coverage.

Reads source files and base Parquet only. Writes audit summaries below the M2
report directory, with no event extraction, fitting, replay or manuscript edits.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from uscrn_source_schema import HH, VARS
from benchmark_full import ROOT, sha256
from download_full import atomic_json
from source_text import read_clean_source


BASE = ROOT / 'reports/uscrn_2007_2025/base'
DESTINATION = ROOT / 'reports/uscrn_2007_2025/m2'
SOIL_FIELDS = tuple(f'SOIL_MOISTURE_{depth}' for depth in (5, 10, 20, 50, 100))
PERIODS = ((2007, 2012), (2013, 2014), (2015, 2020), (2021, 2025))
SUCCESS = {'downloaded', 'cache_verified'}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative_path(path):
    return Path(path).relative_to(ROOT).as_posix()


def save_csv(frame, destination, name):
    frame.to_csv(destination / name, index=False, encoding='utf-8-sig')


def availability_counts(quality):
    keys = ['station', 'wban', 'year']
    assert not quality.duplicated([*keys, 'product', 'variable']).any()
    missing = quality.pivot(index=keys, columns=['variable', 'product'],
                            values='missing_or_bad_fraction_full_year')
    assert missing.shape[1] == len(VARS) * 2 and missing.notna().all().all()
    passed = pd.DataFrame({variable: missing[variable].max(axis=1).le(0.10 + 1e-12)
                           for variable in VARS})
    passed['temperature_and_rain_pass'] = passed[['air_temperature', 'precipitation']].all(axis=1)
    passed['all_four_pass'] = passed[list(VARS)].all(axis=1)
    annual = passed.groupby('year')[['temperature_and_rain_pass', 'all_four_pass']].sum()
    annual['paired_station_years'] = passed.groupby('year').size()
    return annual.reset_index(), passed.reset_index()


def nul_tables(records):
    rows = []
    for record in records:
        for product, audit in record['source_text_audit'].items():
            assert not audit['unrecoverable_anomalies']
            assert audit['nul_byte_count'] == audit['leading_nul_bytes_removed']
            assert not audit['raw_file_modified'] and audit['records_imputed'] == 0
            rows.append({'station': record['station'], 'station_wban': record['station_wban'],
                         'year': record['year'], 'product': product,
                         'source_path': audit['path'], 'raw_sha256': audit['raw_sha256'],
                         'leading_nul_bytes': audit['leading_nul_bytes_removed']})
    files = pd.DataFrame(rows)
    assert not files.source_path.duplicated().any()
    counts = files.assign(affected=files.leading_nul_bytes.gt(0)).groupby('year').agg(
        source_files=('product', 'size'), affected_files=('affected', 'sum'),
        leading_nul_bytes=('leading_nul_bytes', 'sum')).reset_index()
    return files.loc[files.leading_nul_bytes.gt(0)], counts


def column_validity(values, field, wind_flag):
    finite = np.isfinite(values)
    if field == 'WIND_1_5':
        return finite & (values >= 0) & (values != -99) & (wind_flag == 0)
    return finite & (values != -99) & (values >= 0) & (values <= 1)


def coverage_rows(frame, fields, record, expected_hours, step):
    time = pd.DatetimeIndex(frame.time_end_utc)
    assert time.is_unique and time.is_monotonic_increasing
    months = (time - pd.Timedelta(nanoseconds=1)).month.to_numpy()
    first = pd.Timestamp(year=record['year'], month=1, day=1, tz='UTC')
    boundaries = pd.date_range(first, periods=13, freq='MS')
    month_grid = np.diff(boundaries.as_unit('ns').asi8) // pd.Timedelta(minutes=step).value
    observed_by_month = np.bincount(months, minlength=13)[1:]
    assert len(frame) <= expected_hours * 60 // step
    annual, monthly = [], []
    flag = frame.WIND_FLAG.to_numpy() if step == 5 else None
    for field in fields:
        values = frame[field].to_numpy(dtype=float)
        valid = column_validity(values, field, flag)
        valid_by_month = np.bincount(months, weights=valid, minlength=13)[1:].astype('int64')
        positions = np.flatnonzero(valid)
        annual.append({
            'station_wban': record['station_wban'], 'station': record['station'], 'year': record['year'],
            'field': field, 'native_minutes': step, 'valid_rows': int(valid.sum()),
            'observed_rows': len(frame), 'calendar_rows': int(expected_hours * 60 // step),
            'sentinel_rows': int((values == -99).sum()), 'nonfinite_rows': int((~np.isfinite(values)).sum()),
            'outside_bounds_rows': int(((values != -99) & np.isfinite(values)
                                        & ((values < 0) | ((values > 1) if step == 60 else False))).sum()),
            'nonzero_wind_qc_rows': int((flag != 0).sum()) if step == 5 else 0,
            'first_valid_utc': str(time[positions[0]]) if len(positions) else None,
            'last_valid_utc': str(time[positions[-1]]) if len(positions) else None,
        })
        monthly.extend({'year': record['year'], 'month': month, 'field': field,
                        'valid_rows': int(valid_by_month[month - 1]),
                        'observed_rows': int(observed_by_month[month - 1]),
                        'calendar_rows': int(month_grid[month - 1])} for month in range(1, 13))
    return annual, monthly


def raw_hourly_calculation(record):
    audit = record['source_text_audit']['hourly02']
    path = Path(audit['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != audit['raw_sha256']:
        raise ValueError(f'Raw hourly source hash differs at {record["station"]}/{record["year"]}')
    if audit['nul_byte_count']:
        clean, verified = read_clean_source(path, len(HH))
        assert verified == audit
        source = io.StringIO(clean)
    else:
        source = io.BytesIO(raw)
    columns = [HH.index(name) for name in ('WBANNO', 'UTC_DATE', 'UTC_TIME', 'T_CALC')]
    frame = pd.read_csv(source, header=None, sep=r'\s+', usecols=columns,
                        dtype={0: str, 1: str, 2: str}, na_filter=False)
    frame.columns = ['station_wban', 'date', 'clock', 'T_CALC']
    assert frame.station_wban.eq(record['station_wban']).all()
    time = pd.to_datetime(frame.date + frame.clock.str.zfill(4), format='%Y%m%d%H%M', utc=True)
    assert time.is_unique and time.is_monotonic_increasing
    values = pd.to_numeric(frame.T_CALC).to_numpy(dtype=float, copy=True)
    values[values <= -9990] = np.nan
    return pd.DatetimeIndex(time), values


def exception_hours(record):
    if not any(row['above_tolerance_count'] for row in record['consistency']):
        return []
    frame = pq.ParquetFile(ROOT / record['outputs']['hourly_pairs']['path']).read(columns=[
        'time_end_utc', 'variable', 'complete_pair', 'discrepancy_derived_minus_official'],
        use_threads=False).to_pandas()
    thresholds = frame.variable.map({name: source[-1] for name, source in VARS.items()})
    selected = frame.loc[frame.complete_pair & frame.discrepancy_derived_minus_official.abs().gt(thresholds)].copy()
    counts = selected.variable.value_counts()
    assert all(counts.get(row['variable'], 0) == row['above_tolerance_count'] for row in record['consistency'])
    selected['station_wban'] = record['station_wban']
    selected['station'] = record['station']
    selected['source_year'] = record['year']
    return selected.to_dict('records')


def audit_partition(record):
    fine = pq.ParquetFile(ROOT / record['outputs']['high_frequency']['path']).read(columns=[
        'time_end_utc', 'station_wban', 'air_temperature', 'WIND_1_5', 'WIND_FLAG'],
        use_threads=False).to_pandas()
    soil = pq.ParquetFile(ROOT / record['outputs']['hourly_hydrology']['path']).read(columns=[
        'time_end_utc', 'station_wban', *SOIL_FIELDS], use_threads=False).to_pandas()
    assert len(fine) == record['outputs']['high_frequency']['rows']
    assert len(soil) == record['outputs']['hourly_hydrology']['rows']
    assert fine.station_wban.eq(record['station_wban']).all() and soil.station_wban.eq(record['station_wban']).all()
    year = record['year']
    lower = pd.Timestamp(year=year, month=1, day=1, tz='UTC')
    upper = lower + pd.DateOffset(years=1)
    for frame in (fine, soil):
        assert ((frame.time_end_utc > lower) & (frame.time_end_utc <= upper)).all()
    expected_hours = int((upper - lower) / pd.Timedelta(hours=1))
    soil_annual, soil_month = coverage_rows(soil, SOIL_FIELDS, record, expected_hours, 60)
    wind_annual, wind_month = coverage_rows(fine, ['WIND_1_5'], record, expected_hours, 5)

    times, calc = raw_hourly_calculation(record)
    assert times.equals(pd.DatetimeIndex(soil.time_end_utc))
    endpoint = fine.set_index('time_end_utc').air_temperature.reindex(times).to_numpy(dtype=float)
    comparable = np.isfinite(calc) & np.isfinite(endpoint)
    delta = np.abs(calc - endpoint)
    unequal = comparable & (delta != 0)
    checks = {'station_wban': record['station_wban'], 'station': record['station'], 'source_year': year,
              'hourly_source_rows': len(times), 'compared_hours': int(comparable.sum()),
              'equal_hours': int((comparable & (delta == 0)).sum()), 'unequal_hours': int(unequal.sum()),
              'missing_T_CALC_hours': int((~np.isfinite(calc)).sum()),
              'missing_fine_endpoint_hours': int((~np.isfinite(endpoint)).sum()),
              'unscored_hours': int((~comparable).sum())}
    mismatches = [{'station_wban': record['station_wban'], 'source_year': year,
                   'time_end_utc': times[i], 'T_CALC': calc[i], 'fine_temperature': endpoint[i]}
                  for i in np.flatnonzero(unequal)]
    return checks, mismatches, soil_annual + wind_annual, soil_month + wind_month, exception_hours(record)


def coverage_summary(annual, by):
    frame = annual.assign(has_valid=annual.valid_rows.gt(0)).groupby(by).agg(
        valid_rows=('valid_rows', 'sum'), observed_rows=('observed_rows', 'sum'),
        calendar_rows=('calendar_rows', 'sum'), contributing_station_years=('has_valid', 'sum'),
        sentinel_rows=('sentinel_rows', 'sum'), nonfinite_rows=('nonfinite_rows', 'sum'),
        outside_bounds_rows=('outside_bounds_rows', 'sum'),
        nonzero_wind_qc_rows=('nonzero_wind_qc_rows', 'sum')).reset_index()
    frame['valid_fraction_observed'] = frame.valid_rows / frame.observed_rows
    frame['valid_fraction_calendar'] = frame.valid_rows / frame.calendar_rows
    return frame


def run(destination, workers):
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = ROOT / 'data/uscrn_2007_2025/source_manifest.json'
    manifest = load_json(BASE / 'full_build_manifest.json')
    source = load_json(source_manifest)
    records = manifest['records']
    assert manifest['source_manifest_sha256'] == sha256(source_manifest)
    assert source['summary']['complete'] and not source['summary']['failed_files']
    assert len(records) == manifest['summary']['successful_pairs'] and not manifest['summary']['failed_pairs']
    assert len({(r['station_wban'], r['year']) for r in records}) == len(records)
    assert all(row['status'] == 'success' for row in records)
    assert all(row['state'] in SUCCESS for row in source['records'])

    quality = pd.read_csv(BASE / 'full_quality_profile.csv', dtype={'wban': str})
    consistency = pd.read_csv(BASE / 'full_cross_resolution_consistency.csv', dtype={'station_wban': str})
    assert len(quality) == len(records) * len(VARS) * 2
    assert len(consistency) == len(records) * len(VARS)
    assert quality.duplicate_timestamps.eq(0).all() and quality.unexpected_timestamps.eq(0).all()
    assert (quality.missing_or_bad_count_full_year == quality.absent_timestamps_full_year
            + quality.invalid_observed_count).all()
    annual_pass, station_pass = availability_counts(quality)
    save_csv(annual_pass, destination, 'annual_availability.csv')
    save_csv(station_pass, destination, 'station_year_pass.csv')
    affected, nul_annual = nul_tables(records)
    save_csv(affected, destination, 'leading_nul_sources.csv')
    save_csv(nul_annual, destination, 'leading_nul_by_year.csv')

    endpoint_checks, endpoint_mismatches, coverage, monthly, exceptions = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(audit_partition, records, chunksize=2), 1):
            checks, bad_endpoints, annual_fields, monthly_fields, bad_products = result
            endpoint_checks.append(checks)
            endpoint_mismatches.extend(bad_endpoints)
            coverage.extend(annual_fields)
            monthly.extend(monthly_fields)
            exceptions.extend(bad_products)
            if i % 200 == 0 or i == len(records):
                print(f'M2 supplementary audit {i}/{len(records)} station-years', flush=True)

    endpoints = pd.DataFrame(endpoint_checks)
    mismatches = pd.DataFrame(endpoint_mismatches, columns=[
        'station_wban', 'source_year', 'time_end_utc', 'T_CALC', 'fine_temperature'])
    exception_table = pd.DataFrame(exceptions)
    save_csv(endpoints, destination, 'endpoint_consistency_by_station_year.csv')
    save_csv(mismatches, destination, 'endpoint_mismatches.csv')
    save_csv(exception_table, destination, 'cross_product_exception_hours.csv')
    variable_summary = consistency.groupby('variable').agg(
        compared_hours=('eligible_hours', 'sum'), anomalous_hours=('above_tolerance_count', 'sum'),
        tolerance=('diagnostic_tolerance', 'first')).reset_index()
    counted = exception_table.variable.value_counts()
    assert all(counted.get(row.variable, 0) == row.anomalous_hours for row in variable_summary.itertuples())
    save_csv(variable_summary, destination, 'cross_product_summary.csv')

    coverage = pd.DataFrame(coverage)
    coverage['period'] = coverage.year.map({year: f'{start}-{end}' for start, end in PERIODS
                                           for year in range(start, end + 1)})
    save_csv(coverage, destination, 'hydrology_wind_by_station_year.csv')
    all_coverage = coverage_summary(coverage, ['field'])
    annual_coverage = coverage_summary(coverage, ['year', 'field'])
    period_coverage = coverage_summary(coverage, ['period', 'field'])
    save_csv(all_coverage, destination, 'hydrology_wind_overall.csv')
    save_csv(annual_coverage, destination, 'hydrology_wind_by_year.csv')
    save_csv(period_coverage, destination, 'hydrology_wind_by_period.csv')
    monthly = pd.DataFrame(monthly).groupby(['year', 'month', 'field'])[
        ['valid_rows', 'observed_rows', 'calendar_rows']].sum().reset_index()
    monthly['valid_fraction_observed'] = monthly.valid_rows / monthly.observed_rows
    monthly['valid_fraction_calendar'] = monthly.valid_rows / monthly.calendar_rows
    save_csv(monthly, destination, 'hydrology_wind_by_month.csv')
    missingness = quality.groupby(['year', 'product', 'variable']).agg(
        calendar_rows=('expected_rows_full_year', 'sum'), missing_or_bad_rows=('missing_or_bad_count_full_year', 'sum'),
        absent_rows=('absent_timestamps_full_year', 'sum'), invalid_observed_rows=('invalid_observed_count', 'sum')).reset_index()
    missingness['missing_or_bad_fraction_calendar'] = missingness.missing_or_bad_rows / missingness.calendar_rows
    save_csv(missingness, destination, 'annual_variable_missingness.csv')

    totals = endpoints.select_dtypes(include='number').drop(columns='source_year').sum().astype('int64').to_dict()
    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256': sha256(source_manifest),
        'build_manifest_sha256': sha256(BASE / 'full_build_manifest.json'),
        'scope': {'source_years': sorted({r['year'] for r in records}), 'paired_station_years': len(records),
                  'stations': len({r['station_wban'] for r in records}),
                  'observation_source_files': len(records) * 2},
        'leading_nul': {'affected_files': len(affected), 'bytes': int(affected.leading_nul_bytes.sum())},
        'T_CALC_endpoint': {**totals, 'exact_agreement_fraction': totals['equal_hours'] / totals['compared_hours']},
        'cross_product': variable_summary.to_dict('records'),
        'cross_product_anomalous_variable_hours': len(exception_table),
        'cross_product_anomalous_unique_station_hours': len(exception_table[['station_wban', 'time_end_utc']].drop_duplicates()),
        'availability': annual_pass.to_dict('records'),
        'hydrology_wind_overall': all_coverage.to_dict('records'),
        'definitions': {
            'station_year_pass': 'Each required variable has calendar missing/bad fraction <= 0.10 in both products. Review is excluded.',
            'cross_product': 'Complete accepted five-minute targets and a finite screened official hourly value. Absolute difference > source audit tolerance.',
            'endpoint': 'Recorded hourly timestamps with finite, non-sentinel T_CALC and an accepted matching five-minute air-temperature endpoint. No 12-step completeness condition.',
            'soil': 'Finite hourly02 value in [0,1], excluding -99. No separate soil QC flag.',
            'wind': 'Finite subhourly01 WIND_1_5 >= 0, excluding -99, and WIND_FLAG == 0.',
            'observed_rate': 'Pooled valid rows divided by actual recorded rows across paired station-years.',
            'calendar_rate': 'Pooled valid rows divided by full UTC source-year native endpoint grids across paired station-years.',
            'months': 'The right endpoint at month-start 00:00 belongs to the preceding month.',
        },
    }
    atomic_json(destination / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--workers', type=int, default=4, choices=(1, 2, 4))
    arguments = parser.parse_args()
    run(arguments.destination, arguments.workers)
