"""Check event diagnostics on local stations and a synthetic UTC year boundary."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from uscrn_assertions import (
    PROFILE_FIELDS, assert_p95_reference, assert_portable_index_and_strata,
    assert_soil_moisture_attribution, assert_station_and_feature_policy,
    assert_tier_mapping, assert_tier_partition_and_nulls,
)
from benchmark_full import FEATURES, sha256
from precipitation_intervals import read_partition
from build_precipitation_event_library import annual_grid, catalogue_block, read_station_year
from station_release_tables import AnnualRain, bound, enrich_events
from uscrn_event_diagnostics import (
    AFTER_DEPLOYMENT, BEFORE_DEPLOYMENT, FINE_FIELDS, SOIL_OFFSET, SOIL_REASON_FIELDS,
    SOIL_SOURCE_FIELDS, SOIL_VALUE_FIELDS, STRING_DIAGNOSTICS, VALID,
    add_event_diagnostics, fine_diagnostics, soil_candidate, wet_bulb_temperature,
)


BASE_MANIFEST = ROOT / 'reports/uscrn_2007_2025/base/full_build_manifest.json'
METADATA = ROOT / 'output/station_disaggregation_release_2007_2025/data/station_metadata.csv'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def annual_rain(record, annual):
    year = record['year']
    hours = read_partition(record, 'hourly_pairs', [
        'time_end_utc', 'station_wban', 'variable', 'official_hourly', 'derived_hourly',
    ], annual_grid(year, year, 'h'))
    complete = np.isfinite(annual.amounts).reshape(-1, 12).all(axis=1)
    inputs = {}
    for track in ('official', 'derived'):
        valid = bound(hours[f'{track}_hourly'].to_numpy(dtype=float), 'precipitation')
        padded = np.r_[False, valid, False]
        inputs[track] = valid & padded[:-2] & padded[2:]
    blockers = np.maximum.accumulate(np.where(annual.amounts == 0, -1, np.arange(len(annual.amounts))))
    first_record = int(np.flatnonzero(annual.quality != 1)[0])
    return AnnualRain(year, first_record, annual.amounts, annual.quality, blockers, complete,
                      complete & inputs['official'], complete & inputs['derived'],
                      inputs['official'], hours.official_hourly.to_numpy(dtype=float))


def assert_parquet_tier_three(events, path):
    events.to_parquet(path, index=False)
    strings = ['storm_profile_tercile', *STRING_DIAGNOSTICS]
    table = pq.read_table(path, columns=['usability_tier', *strings])
    tier_three = table.filter(pc.equal(table['usability_tier'], 3))
    assert tier_three.num_rows == events.usability_tier.eq(3).sum()
    assert all(tier_three[name].null_count == tier_three.num_rows for name in strings)
    assert_tier_partition_and_nulls(pd.read_parquet(path))
    return tier_three.num_rows


@pytest.mark.skipif(not BASE_MANIFEST.is_file() or not METADATA.is_file(), reason='Requires local base and release data')
def test_two_real_smoke_station_years(tmp_path):
    records = [record for record in read_json(BASE_MANIFEST)['records']
               if record['year'] == 2013 and record['station_wban'] in {'63868', '53131'}]
    assert {(r['station_wban'], r['year']) for r in records} == {('63868', 2013), ('53131', 2013)}
    station_metadata = pd.read_csv(METADATA, dtype={'station_wban': str})
    stations = station_metadata.loc[station_metadata.station_wban.isin(r['station_wban'] for r in records)]
    assert_station_and_feature_policy(stations, [r['station_wban'] for r in records], FEATURES,
                                      validation_tuning_trials=0)
    catalogue = read_json(ROOT / 'config/precipitation_event_protocol.json')
    profiles = read_json(ROOT / 'config/station_disaggregation_release.json')
    annual_events = []
    for record in records:
        station = stations.loc[stations.station_wban.eq(record['station_wban'])].iloc[0]
        annual = read_station_year(record, None)
        rows = pd.DataFrame(catalogue_block([annual], station, catalogue))
        enriched = pd.DataFrame(enrich_events(rows, {annual.year: annual_rain(record, annual)}, profiles))
        annual_events.append(rows.merge(enriched, on='event_id', validate='one_to_one'))
    events, sensors = add_event_diagnostics(pd.concat(annual_events, ignore_index=True), records)
    assert_tier_mapping(events)
    assert_tier_partition_and_nulls(events)
    assert_soil_moisture_attribution(events, sensors)
    tier_three_count = assert_parquet_tier_three(events, tmp_path / 'smoke_events.parquet')
    for (wban, station), site in events.groupby(['station_wban', 'station']):
        print(json.dumps({
            'station': station, 'station_wban': wban, 'source_year': 2013,
            'events': len(site), 'tier_counts': site.usability_tier.value_counts().sort_index().to_dict(),
            'wetbulb_valid_events': int(site.event_wetbulb_mean_c.notna().sum()),
            'wind_valid_events': int(site.event_wind_mean_ms.notna().sum()),
            'soil_valid_events_by_depth_cm': {
                name: int(site[name].notna().sum()) for name in SOIL_VALUE_FIELDS},
        }, sort_keys=True))
    assert events.event_wind_mean_ms.notna().any()
    assert events.event_wetbulb_mean_c.notna().any()
    assert events.loc[events.station_wban.eq('53131'), SOIL_VALUE_FIELDS[0]].notna().any()
    print(f'Gate 0 Arrow readback: {tier_three_count} Tier 3 rows, '
          f'{len(STRING_DIAGNOSTICS) + 1} true-null string fields per row')


def utc(value):
    return pd.Timestamp(value, tz='UTC')


def stage_boundary_record(tmp_path, year, fine_times, fine_rows, soil_times, soil_rows):
    outputs = {}
    for product, fields, times, rows in (
        ('high_frequency', FINE_FIELDS, fine_times, fine_rows),
        ('hourly_hydrology', SOIL_SOURCE_FIELDS, soil_times, soil_rows),
    ):
        frame = pd.DataFrame(rows, columns=list(fields))
        frame['station_wban'] = '63868'
        frame['source_year'] = year
        frame['time_end_utc'] = pd.to_datetime(times, utc=True)
        if product == 'high_frequency':
            # Wrong sensor to use for phase, deliberately unlike the ambient temperature.
            frame['SURFACE_TEMPERATURE'] = 35.0
        path = tmp_path / f'{product}_{year}.parquet'
        frame.to_parquet(path, index=False)
        outputs[product] = {'path': str(path)}
    return {'station_wban': '63868', 'year': year, 'outputs': outputs}


def test_utc_boundary_and_tier_three(tmp_path):
    previous = stage_boundary_record(
        tmp_path, 2019, ['2019-12-31 23:55', '2020-01-01 00:00'],
        [(40.0, 0, 100.0, 0, 99.0, 0), (-1.0, 0, 100.0, 0, 0.0, 0)],
        ['2019-12-31 23:00', '2020-01-01 00:00'],
        [(0.20, -99.0, 0.20, -99.0, 1.0), (0.25, 0.33, 1.01, -99.0, -99.0)],
    )
    new = stage_boundary_record(
        tmp_path, 2020, ['2020-01-01 00:05', '2020-01-01 00:10', '2020-01-01 01:10'],
        [(2.0, 0, 100.0, 0, 2.0, 0), (5.0, 1, 100.0, 0, 60.0, 1), (3.0, 0, 101.0, 0, -99.0, 0)],
        ['2020-01-01 01:00', '2020-01-01 02:00'],
        [(-99.0, np.nan, -99.0, -99.0, np.nan), (0.81, np.nan, -99.0, 0.44, np.nan)],
    )
    # Each row is a separate counterfactual event-start check on the same input partitions.
    events = pd.DataFrame([
        ('cross_midnight', '2019-12-31 23:55', '2020-01-01 00:10', True, True, 0, 1),
        ('on_midnight', '2020-01-01 00:00', '2020-01-01 00:05', True, True, 0, 1),
        ('after_midnight', '2020-01-01 00:05', '2020-01-01 00:10', True, False, 1, 2),
        ('no_valid_depth', '2020-01-01 01:05', '2020-01-01 01:10', True, True, 0, 1),
        ('censored', '2020-01-01 00:05', '2020-01-01 00:10', False, True, 0, 3),
    ], columns=['event_id', 'event_start_utc', 'event_end_utc', 'boundary_confirmed',
                'hourly_benchmark_eligible', 'ineligible_covering_hours', 'usability_tier'])
    for name in ('event_start_utc', 'event_end_utc'):
        events[name] = pd.to_datetime(events[name], utc=True)
    events['station_wban'] = '63868'
    for name, fill in zip(PROFILE_FIELDS, (0.5, 'mid_loaded', 0.5, False), strict=True):
        events[name] = [fill] * 4 + [None]

    checked, sensors = add_event_diagnostics(events, [new, previous])
    assert_tier_mapping(checked)
    assert_tier_partition_and_nulls(checked)
    assert_soil_moisture_attribution(checked, sensors)
    assert assert_parquet_tier_three(checked, tmp_path / 'boundary_events.parquet') == 1
    rows = checked.set_index('event_id')
    assert soil_candidate(utc('2020-01-01 00:00')) == (utc('2019-12-31 23:00'), 2019, 60)
    assert soil_candidate(utc('2020-01-01 00:05')) == (utc('2020-01-01 00:00'), 2019, 5)
    assert soil_candidate(utc('2020-01-01 00:55')) == (utc('2020-01-01 00:00'), 2019, 55)
    assert rows.loc['on_midnight', SOIL_OFFSET] == 60
    assert rows.loc['after_midnight', SOIL_OFFSET] == 5
    assert rows.loc['after_midnight', list(SOIL_REASON_FIELDS)].tolist() == [
        VALID, VALID, AFTER_DEPLOYMENT, BEFORE_DEPLOYMENT, AFTER_DEPLOYMENT,
    ]
    assert rows.loc['after_midnight', SOIL_VALUE_FIELDS[0]] == 0.25
    assert rows.loc['after_midnight', SOIL_VALUE_FIELDS[1]] == 0.33
    assert rows.loc['no_valid_depth', list(SOIL_VALUE_FIELDS)].isna().all()
    assert pd.isna(rows.loc['no_valid_depth', SOIL_OFFSET])
    assert rows.loc['cross_midnight', 'event_wetbulb_mean_c'] == 0.5
    assert rows.loc['cross_midnight', 'event_wind_mean_ms'] == 1
    assert rows.loc['cross_midnight', 'precip_phase_flag'] == 'mixed_or_transitional'
    assert rows.loc['on_midnight', 'event_wetbulb_mean_c'] == 2
    assert rows.loc['after_midnight', 'precip_phase_flag'] == 'temp_missing'

    with pytest.raises(AssertionError, match='#1'):
        assert_tier_mapping(checked.assign(hourly_benchmark_eligible=False))
    with pytest.raises(AssertionError, match='#2'):
        assert_tier_partition_and_nulls(checked.assign(usability_tier=3))
    corrupted = checked.copy()
    corrupted.loc[corrupted.usability_tier.eq(3), SOIL_REASON_FIELDS[0]] = ''
    assert_soil_moisture_attribution(corrupted, sensors)
    with pytest.raises(AssertionError, match='Tier 3 diagnostics must be null'):
        assert_tier_partition_and_nulls(corrupted)
    corrupted.loc[corrupted.usability_tier.eq(1), SOIL_OFFSET] = 0
    with pytest.raises(AssertionError, match='#5 shared offset'):
        assert_soil_moisture_attribution(corrupted, sensors)
    print('UTC year boundary: previous-source-year midnight matched, lags 5 and 60 min retained')


def test_wet_bulb_phase_and_wind_rules():
    estimated = wet_bulb_temperature([-20, 50, 20, 51, 10, 10], [100, 100, 50, 100, 4, 101])
    np.testing.assert_array_equal(estimated[:2], [-20, 50])
    assert estimated[2] == pytest.approx(13.69934, abs=1e-5)
    assert np.isnan(estimated[3:]).all()
    span = pd.DataFrame([
        (3., 0, 100., 0, 4., 0), (0., 0, 100., 0, -1., 0),
        (-1., 0, 100., 0, np.inf, 0), (2., 0, 100., 1, 99., 2),
        (8., 1, 100., 0, -99., 0),
    ], columns=list(FINE_FIELDS))
    assert fine_diagnostics(span.iloc[[0]])['precip_phase_flag'] == 'liquid_likely'
    assert fine_diagnostics(span.iloc[[2]])['precip_phase_flag'] == 'frozen_likely'
    for pos in (1, 3):
        assert fine_diagnostics(span.iloc[[pos]])['precip_phase_flag'] == 'mixed_or_transitional'
    summary = fine_diagnostics(span)
    assert summary['event_wind_mean_ms'] == 4
    assert summary['event_wetbulb_mean_c'] == pytest.approx(2 / 3)


def test_p95_window_and_validation_feature_guards():
    events = pd.DataFrame([('a', 'train', True, '2007-01-01T06:00Z', '2020-12-31T17:55Z')],
                          columns=['event_id', 'dataset_split', 'boundary_confirmed',
                                   'event_start_utc', 'event_end_utc'])
    assert_p95_reference(events)
    for changed in ({'event_end_utc': '2020-12-31T18:00Z'}, {'event_start_utc': '2007-01-01T05:55Z'},
                    {'dataset_split': 'test'}, {'boundary_confirmed': False}):
        with pytest.raises(AssertionError, match='#3'):
            assert_p95_reference(events.assign(**changed))
    stations = pd.DataFrame({'station_wban': ['63868'], 'station_split': ['train'], 'dataset_split': ['train']})
    expanded = [*FEATURES, *(f'{name}_missing' for name in FEATURES)]
    assert_station_and_feature_policy(stations, ['63868'], expanded, validation_tuning_trials=0)
    for field in ('event_wind_mean_ms', 'T_CALC', 'cross_var_consistency_flag',
                  'is_valid_target', 'soil_moisture_pre_5cm'):
        with pytest.raises(AssertionError, match='#4'):
            assert_station_and_feature_policy(stations, ['63868'], [*FEATURES, field], validation_tuning_trials=0)
    with pytest.raises(AssertionError, match='#4'):
        assert_station_and_feature_policy(stations, ['63868'], expanded, validation_tuning_trials=1)


def test_index_and_strata_guard(tmp_path):
    path = tmp_path / 'events.parquet'
    pd.DataFrame({'event_id': ['a', 'b', 'c']}).to_parquet(path, index=False)
    entry = {'path': path.name, 'rows': 3, 'sha256': sha256(path)}
    index = {'records': [{'outputs': {'events': entry}}]}
    totals = {'catalogue': 3, 'replay': 1}
    strata = {'catalogue': {'tier': [1, 2]}, 'replay': {'network': [1, 0]}}
    assert_portable_index_and_strata(tmp_path, index, totals, strata)
    with pytest.raises(AssertionError, match='#6 Stratified counts'):
        assert_portable_index_and_strata(tmp_path, index, dict(totals, replay=2), strata)
    for field, value in (('rows', 2), ('sha256', '0' * 64), ('path', '../events.parquet')):
        wrong = {'records': [{'outputs': {'events': dict(entry, **{field: value})}}]}
        with pytest.raises(AssertionError, match='#6'):
            assert_portable_index_and_strata(tmp_path, wrong, totals, strata)
