"""Gate 0a: parse representative NOAA files and audit transition months."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from uscrn_source_schema import SH, HH, interval_grid
from build_full_dataset import validate_source
from download_full import ROOT, atomic_json, download_one, inventory_year, manifest_index, read_json


DEFAULT_CONFIG = ROOT / 'config/uscrn_2007_2025.json'
PRIMARY_YEARS = (2007, 2010, 2013, 2015, 2022)
PRIMARY_SITES = ('TX_Monahans_6_ENE', 'AL_Courtland_2_WSW')
EXTRA_SITES = {
    2012: ('AZ_Tucson_11_W', 'NY_Ithaca_13_E', 'CO_Grand_Junction_9_W'),
    2013: ('AZ_Tucson_11_W', 'NY_Ithaca_13_E', 'CO_Grand_Junction_9_W'),
    2014: ('AZ_Holbrook_17_ESE', 'NY_Ithaca_13_E', 'CO_Grand_Junction_9_W'),
    2015: ('AZ_Tucson_11_W', 'NY_Ithaca_13_E', 'AL_Fairhope_3_NE'),
}
SENTINELS = {'soil': -99.0, 'wind': -99.0, 'surface_temperature': -9999.0}
SOURCE_FIELDS = {
    'subhourly01': [
        ('SOIL_MOISTURE_5', 'soil', None, None, 'm3/m3'),
        ('WIND_1_5', 'wind', 'WIND_FLAG', None, 'm/s'),
        ('SURFACE_TEMPERATURE', 'surface_temperature', 'ST_FLAG', 'ST_TYPE', 'degC'),
    ],
    'hourly02': [
        *((f'SOIL_MOISTURE_{depth}', 'soil', None, None, 'm3/m3') for depth in (5, 10, 20, 50, 100)),
        ('SUR_TEMP', 'surface_temperature', 'SUR_TEMP_FLAG', 'SUR_TEMP_TYPE', 'degC'),
        ('SUR_TEMP_MAX', 'surface_temperature', 'SUR_TEMP_MAX_FLAG', 'SUR_TEMP_TYPE', 'degC'),
        ('SUR_TEMP_MIN', 'surface_temperature', 'SUR_TEMP_MIN_FLAG', 'SUR_TEMP_TYPE', 'degC'),
    ],
}


def inventory_probes(config):
    for year in sorted(set(PRIMARY_YEARS) | set(EXTRA_SITES)):
        roster = dict.fromkeys(PRIMARY_SITES, 'representative') if year in PRIMARY_YEARS else {}
        roster.update(dict.fromkeys(EXTRA_SITES.get(year, ()), 'transition_monthly'))
        listing = inventory_year(config, year, refresh=True)
        for site, role in roster.items():
            if site not in listing['paired_stations']:
                raise ValueError(f'Missing required probe pair: {year}/{site}')
            for product in config['products']:
                yield {**listing['listings'][product]['files'][site], 'probe_role': role}


def documentation(config, relative):
    return {'relative_source_path': relative, 'source_url': urljoin(config['source_base_url'], relative),
            'product': relative.split('/')[0], 'station': None, 'year': None,
            'expected_bytes_from_inventory': None, 'probe_role': 'documentation'}


def verify_headers(raw):
    verified = []
    for product, projection in [('subhourly01', SH), ('hourly02', HH)]:
        lines = (raw / product / 'HEADERS.txt').read_text(encoding='ascii').splitlines()
        numbers, fields, units = [line.split() for line in lines[:3]]
        if fields != projection or [int(number) for number in numbers] != list(range(1, len(projection) + 1)):
            raise ValueError(f'NOAA {product} headers disagree with the project parser')
        given_units = dict(zip(fields, units, strict=True))
        for name, kind, flag, type_field, unit in SOURCE_FIELDS[product]:
            pos = fields.index(name) + 1
            documented = given_units[name]
            if documented != unit.replace('m3/m3', 'm^3/m^3').replace('degC', 'Celsius'):
                raise ValueError(f'Unexpected unit: {product}/{name} {documented}')
            verified.append({'product': product, 'field': name, 'position_1_based': pos,
                             'nominal_interval_minutes': 5 if product == 'subhourly01' else 60,
                             'documented_unit': documented, 'sentinel': SENTINELS[kind],
                             'qc_flag': flag, 'qc_position_1_based': fields.index(flag) + 1 if flag else None,
                             'type_field': type_field,
                             'type_position_1_based': fields.index(type_field) + 1 if type_field else None})
    return verified


def describe_file(record, schema):
    product, year = record['product'], record['year']
    fine = product == 'subhourly01'
    interval = 5 if fine else 60
    frame, wban = validate_source(ROOT / record['path'], fine, year)
    byte_audit = frame.attrs['source_text_audit']
    if byte_audit['raw_sha256'] != record['sha256']:
        raise ValueError(f'Raw hash changed: {record["path"]}')
    # The right endpoint at midnight belongs to the preceding month.
    months = (frame.index - pd.Timedelta(nanoseconds=1)).strftime('%Y-%m')
    expected_months = (interval_grid(fine, year) - pd.Timedelta(nanoseconds=1)).strftime('%Y-%m')
    expected = pd.Series(expected_months).value_counts().sort_index()
    results, monthly = [], []
    for key in schema:
        if key['product'] != product:
            continue
        field, flag, type_column = key['field'], key['qc_flag'], key['type_field']
        source = pd.to_numeric(frame[field], errors='raise')
        sentinel = source.eq(key['sentinel'])
        valid = np.isfinite(source) & ~sentinel
        if field.startswith('SOIL_MOISTURE_'):
            valid &= source.between(0, 1)
        if field == 'WIND_1_5':
            valid &= source.ge(0)
        if flag:
            valid &= frame[flag].eq(0)
        if type_column:
            valid &= frame[type_column].isin(['C', 'R'])
        valid_utc = frame.index[valid]
        gaps = valid_utc.to_series().diff().dropna() / pd.Timedelta(minutes=1)
        results.append({'source_year': year, 'station_wban': wban, 'station': record['station'],
                        'product': product, 'role': record['probe_role'], 'field': field,
                        'n_source_columns': len(frame.columns), 'interval_minutes': interval,
                        'rows': len(frame), 'sentinel_rows': int(sentinel.sum()), 'valid_rows': int(valid.sum()),
                        'zero_rows': int((valid & source.eq(0)).sum()),
                        'first_record_utc': frame.index.min(), 'last_record_utc': frame.index.max(),
                        'first_valid_utc': valid_utc.min(), 'last_valid_utc': valid_utc.max(),
                        'min_value': source[valid].min(), 'max_value': source[valid].max(),
                        'qc_values': ','.join(map(str, sorted(frame[flag].unique()))) if flag else None,
                        'source_type_values': ','.join(map(str, sorted(frame[type_column].unique())))
                        if type_column else None,
                        'median_valid_gap_min': float(gaps.median()) if len(gaps) else None,
                        'nominal_valid_gap_count': int(gaps.eq(interval).sum()),
                        'leading_nul_bytes': byte_audit['leading_nul_bytes_removed']})
        counts = pd.DataFrame({'month': months, 'valid': valid.to_numpy()}).groupby('month').valid.agg(['size', 'sum'])
        counts = counts.reindex(expected.index, fill_value=0)
        for month, goal in expected.items():
            monthly.append({'source_year': year, 'station_wban': wban, 'station': record['station'],
                            'month': month, 'product': product, 'field': field, 'role': record['probe_role'],
                            'expected_rows': goal, 'observed_rows': counts.loc[month, 'size'],
                            'valid_rows': counts.loc[month, 'sum'],
                            'valid_fraction_calendar': counts.loc[month, 'sum'] / goal})
    return results, monthly


def run(config_path):
    config = read_json(config_path)
    raw = ROOT / config['raw_directory']
    destination = ROOT / config['gate0_directory']
    destination.mkdir(parents=True, exist_ok=True)
    manifest_file = destination / 'probe_manifest.json'
    previous = manifest_index(manifest_file)
    tasks = list(inventory_probes(config))
    tasks.extend(documentation(config, item) for item in config['documentation_paths'])
    with ThreadPoolExecutor(max_workers=config['download']['workers']) as pool:
        outcomes = list(pool.map(lambda item: download_one(config, item, previous.get(item['source_url'])),
                                 tasks))
    atomic_json(manifest_file, {'records': outcomes})
    failed = [r for r in outcomes if r['state'] not in ('downloaded', 'cache_verified')
              or (r['expected_bytes_from_inventory'] is not None and r['bytes'] != r['expected_bytes_from_inventory'])]
    if failed:
        raise ValueError(f'{len(failed)} probe inputs failed transfer or size checks. See {manifest_file}')
    schema = verify_headers(raw)
    pd.DataFrame(schema).to_csv(destination / 'source_field_schema.csv', index=False, encoding='utf-8-sig')
    parsed = []
    monthly = []
    for record in outcomes:
        if record['year'] is None:
            continue
        fields, months = describe_file(record, schema)
        parsed.extend(fields)
        monthly.extend(months)
    values = pd.DataFrame(parsed)
    values.to_csv(destination / 'probe_field_availability.csv', index=False, encoding='utf-8-sig')
    month_table = pd.DataFrame(monthly)
    month_table.to_csv(destination / 'transition_monthly_availability.csv', index=False, encoding='utf-8-sig')
    identity = values[['source_year', 'station', 'station_wban', 'product']].drop_duplicates()
    if not identity.groupby(['source_year', 'station']).station_wban.nunique().eq(1).all():
        raise ValueError('Probe station identity mismatch between products')
    fields = []
    for (product, field), group in values.groupby(['product', 'field']):
        fields.append({'product': product, 'field': field, 'valid_rows': int(group.valid_rows.sum()),
                       'station_years_with_valid': int(group.valid_rows.gt(0).sum()),
                       'nominal_gap_count': int(group.nominal_valid_gap_count.sum())})
    summary = {'stage': 'gate0_step0a', 'status': 'probe_parsed', 'probe_pair_count': len(identity) // 2,
               'monthly_station_years': (month_table[month_table.source_year.isin(EXTRA_SITES)]
                                        [['source_year', 'station']].drop_duplicates().shape[0]),
               'UTC': datetime.now(timezone.utc).isoformat(), 'field_summary': fields,
               'source_manifest': manifest_file.relative_to(ROOT).as_posix()}
    atomic_json(destination / 'probe_summary.json', summary)
    print(f'Parsed {len(identity)} files in {len(identity) // 2} probe pairs. Review {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
