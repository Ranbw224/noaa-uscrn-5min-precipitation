"""M4 freeze, fixed-setting model fitting and held-out rainfall replay."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from uscrn_assertions import (
    BUFFER, REFERENCE_END, REFERENCE_START, assert_p95_reference,
    assert_portable_index_and_strata, assert_station_and_feature_policy,
    assert_tier_mapping, assert_tier_partition_and_nulls,
)
from benchmark_full import (
    FEATURES, ROOT, VARIABLES, add_ratios, all_predictions, fit_models, hourly_targets,
    input_features, load_config, per_hour_metrics, save_json, sha256, strata, sum_metrics,
)
from station_release_tables import bound, describe, export_parquet
from build_uscrn_release import DEFAULT_CONFIG, read_json, stamp
from build_station_disaggregation_manuscript_tables import render
from precipitation_envelope_metrics import metrics_tables, score_event
from station_disaggregation_diagnostics import PROFILES, spread


def reference_mask(events, first, last):
    left = pd.Timestamp(year=first, month=1, day=1, tz='UTC')
    right = pd.Timestamp(year=last + 1, month=1, day=1, tz='UTC') - pd.Timedelta(seconds=1)
    return (events.boundary_confirmed & events.dataset_split.eq('train')
            & (events.event_start_utc - BUFFER >= left) & (events.event_end_utc + BUFFER <= right))


def freeze_reference(events, settings, release):
    first, last = settings['reference_years']
    reference = events.loc[reference_mask(events, first, last)]
    assert_p95_reference(reference)
    cutoff = float(np.quantile(reference.peak_15min_mm, .95, method='linear'))
    events['p95_reference_event'] = reference_mask(events, first, last)
    tolerance = 1e-10
    events['high_intensity'] = pd.array(events.peak_15min_mm.ge(cutoff - tolerance), dtype='boolean')
    events.loc[~events.boundary_confirmed, 'high_intensity'] = pd.NA
    rows = []
    for start in (first, settings['fitting_years'][0], 2015):
        selected = events.loc[reference_mask(events, start, last)]
        assert_p95_reference(selected)
        rows.append({'first_year': start, 'last_year': last, 'station_split': 'train',
                     'events': len(selected), 'stations': selected.station_wban.nunique(),
                     'peak_15min_p95_mm': float(np.quantile(selected.peak_15min_mm, .95, method='linear'))})
    pd.DataFrame(rows).to_csv(release / 'technical_validation/p95_window_sensitivity.csv', index=False)
    result = {'reference_years': [first, last], 'reference_split': 'train', 'reference_events': len(reference),
              'reference_stations': reference.station_wban.nunique(), 'quantile': .95, 'quantile_method': 'linear',
              'cutoff_mm_per_15min': cutoff, 'buffer_hours_each_side': 6,
              'reference_start_utc': str(REFERENCE_START), 'reference_end_utc': str(REFERENCE_END),
              'event_rule': 'Both observed bounds plus full dry buffers lie inside the reference window.',
              'prepared_at_utc': stamp()}
    save_json(release / 'provenance/intensity_threshold.json', result)
    return cutoff, result


def release_reader(record):
    """Read one station-year using the variable/source-year Parquet statistics."""
    root = Path(record['release_root'])
    fine = pq.ParquetFile(ROOT / record['outputs']['high_frequency']['path']).read(
        columns=['time_end_utc', *VARIABLES], use_threads=False).to_pandas()
    parts = []
    for variable in VARIABLES:
        path = root / 'data/pairs' / f'variable={variable}' / f'source_year={record["year"]}' / 'part.parquet'
        frame = ds.dataset(path, format='parquet').to_table(filter=ds.field('station_wban') == record['station_wban'],
            columns=['time_end_utc', 'variable', 'derived_hourly', 'official_hourly', 'LATITUDE', 'LONGITUDE',
                     'station_wban', 'station_split', 'dataset_split', 'source_year', 'is_valid_target',
                     'track_eligible_derived', 'track_eligible_official']).to_pandas()
        if frame.empty or not frame.source_year.eq(record['year']).all():
            raise ValueError(f'Missing release pair partition: {record["station_wban"]}/{record["year"]}/{variable}')
        parts.append(frame)
    pairs = pd.concat(parts, ignore_index=True)
    if not pairs.station_split.eq(record['station_split']).all() or not pairs.dataset_split.eq(record['station_split']).all():
        raise AssertionError('#4 Pair station assignment differs from the base inventory')
    return fine.set_index('time_end_utc'), pairs


def eligible_release_samples(high, pairs, track, variable, hours=None, target_map=None):
    if hours is None:
        hours, target_map = hourly_targets(high)
    x = input_features(pairs, hours, track, variable)
    y = target_map[variable]
    target = np.isfinite(y).all(axis=1) & bound(y, variable).all(axis=1)
    valid_input = bound(x[:, :3], variable).all(axis=1)
    keep = target & valid_input
    label = 'derived' if track == 'derived_hourly' else 'official'
    declared = pairs.loc[pairs.variable.eq(variable)].set_index('time_end_utc')[f'track_eligible_{label}']
    if not np.array_equal(keep, declared.reindex(hours, fill_value=False).to_numpy()):
        raise ValueError(f'Reconstructed and published eligibility differ: {variable}/{track}')
    counts = {'fine_hours': len(hours), 'complete_target_hours': int(target.sum()),
              'complete_target_missing_own_coarse_support': int((target & ~valid_input).sum()),
              'eligible_hours': int(keep.sum())}
    return hours[keep], x[keep], y[keep], counts


def model_fitting(records, config, release, report):
    fixed = load_config(ROOT / config['benchmark_protocol'])
    start, end = config['fitting_years']
    fixed['train_years'] = list(range(start, end + 1))
    fixed['evaluation_years'] = config['replay_years']
    fixed['training_wet_variables'] = config['training_wet_variables']
    fixed['settings_status'] = 'Frozen settings. Validation stations have zero hyperparameter tuning trials.'
    fixed['protocol'] = 'offline_interval_disaggregation_fixed_settings'
    configuration_path = release / 'provenance/benchmark_configuration.json'
    summary_path = report / 'model_fitting_summary.json'
    reusable = summary_path.is_file() and configuration_path.is_file() and read_json(configuration_path) == fixed
    previous = read_json(summary_path) if reusable else {}
    if reusable:
        reusable = (previous['base_manifest_sha256'] == sha256(ROOT / config['base_manifest'])
            and previous['training_coverage_sha256'] == sha256(release / 'training_sample_coverage.csv')
            and len(previous['models']) == len(VARIABLES) * len(fixed['tracks'])
            and all(sha256(release / model['path']) == model['sha256'] for model in previous['models']))
    save_json(configuration_path, fixed)
    if reusable:
        metadata = previous['models']
        print('M4 reused the verified fixed-setting model bundles', flush=True)
    else:
        metadata = fit_models(records, fixed, release, reader=release_reader, sample_selector=eligible_release_samples)
        for model in metadata:
            model['path'] = Path(model['path']).relative_to(release).as_posix()
        save_json(release / 'model_metadata.json', metadata)
    station_by_wban = {record['station_wban']: record['station_split'] for record in records}
    for model in metadata:
        bundle = joblib.load(release / model['path'])
        assert_station_and_feature_policy(
            pd.read_csv(release / 'data/station_metadata.csv', dtype={'station_wban': str}),
            {record['station_wban'] for record in records}, bundle['expanded_feature_names'],
            validation_tuning_trials=config['validation_tuning_trials'])
        if bundle['feature_names'] != FEATURES:
            raise AssertionError('#4 Fitted primary features differ from the frozen whitelist')
        if any(station_by_wban[wban] != 'train' for wban in bundle['training_wbans']):
            raise AssertionError('#4 Fitted sample contains a non-train station')
    exposure = pd.read_csv(release / 'training_sample_coverage.csv', dtype={'station_wban': str})
    if not exposure.station_split.eq('train').all() or not exposure.year.between(start, end).all():
        raise AssertionError('#4 Fitting rows exceed the configured station and time windows')
    save_json(report / 'model_fitting_summary.json', {'models': metadata, 'validation_tuning_trials': 0,
               'fitting_years': [start, end], 'fitting_station_years': exposure[['station_wban', 'year']].drop_duplicates().shape[0],
               'base_manifest_sha256': sha256(ROOT / config['base_manifest']),
               'training_coverage_sha256': sha256(release / 'training_sample_coverage.csv')})
    return fixed, metadata


def replay_roster(records, years):
    sets = {year: {r['station_wban'] for r in records if r['year'] == year and r['station_split'] == 'test'}
            for year in years}
    return sets, set.union(*sets.values()), set.intersection(*sets.values())


def run_rainfall_replay(events, records, config, protocol, cutoff, metadata, fixed, release, report):
    requested = events.loc[events.usability_tier.eq(1) & events.dataset_split.eq('test')
                           & events.first_source_year.isin(config['replay_years'])].copy()
    roster, union, intersection = replay_roster(records, config['replay_years'])
    if not set(requested.station_wban) <= union:
        raise AssertionError('#4 Replay events lie outside the held-out station roster')
    models = {}
    for model in metadata:
        if model['variable'] == 'precipitation':
            if set(model['training_wbans']) & union:
                raise AssertionError('#4 Replay stations overlap the fitted model')
            if sha256(release / model['path']) != model['sha256']:
                raise ValueError('Rainfall model checksum changed')
            models[model['track']] = joblib.load(release / model['path'])
    series_paths = {'official_hourly': release / 'data/envelope_timeseries.parquet'}
    writers = {}
    scores, checks, hourly_rows, cohort_rows = [], [], [], []
    closure_max = 0.0
    try:
        selected_records = [r for r in records if r['year'] in config['replay_years'] and r['station_split'] == 'test']
        for i, record in enumerate(selected_records, 1):
            high, pairs = release_reader(record)
            station_events = requested.loc[requested.station_wban.eq(record['station_wban'])
                                           & requested.first_source_year.eq(record['year'])]
            for track, model in models.items():
                ends, x, targets, _ = eligible_release_samples(high, pairs, track, 'precipitation')
                predictions = all_predictions(model, x, 'precipitation')
                for values in predictions.values():
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError('Rainfall estimates are not finite and nonnegative')
                    closure_max = max(closure_max, float(np.abs(values.sum(axis=1) - x[:, 0]).max()))
                for method, values in predictions.items():
                    metrics = per_hour_metrics(targets, values, x[:, 0], 'precipitation', protocol['peak_depth_tolerance_mm'])
                    for stratum, keep in strata('precipitation', x[:, 0], fixed).items():
                        hourly_rows.append({'station_wban': record['station_wban'], 'year': record['year'],
                                            'variable': 'precipitation', 'track': track, 'method': method,
                                            'stratum': stratum, **sum_metrics(metrics, keep)})
                for event in station_events.itertuples(index=False):
                    hours = pd.date_range(event.first_covering_hour_end_utc, event.last_covering_hour_end_utc, freq='h')
                    index = ends.get_indexer(hours)
                    if (index < 0).any():
                        raise ValueError(f'Replay cohort has an ineligible {track} envelope: {event.event_id}')
                    times = pd.date_range(hours[0] - pd.Timedelta(minutes=55), hours[-1], freq='5min')
                    observed = targets[index].ravel()
                    if len(times) != 12 * event.covering_hours or not np.array_equal(observed, high.precipitation.reindex(times)):
                        raise ValueError(f'Observed envelope does not match the event: {event.event_id}')
                    forecasts = {method: values[index].ravel() for method, values in predictions.items()}
                    rows, comparisons, series = score_event(event, times, observed, forecasts, protocol, cutoff)
                    for row in rows:
                        row['track'] = track
                        scores.append(row)
                    for row in comparisons:
                        row['track'] = track
                        checks.append(row)
                    if track == 'official_hourly':
                        cohort_rows.append({'event_id': event.event_id, 'station_wban': event.station_wban,
                                            'source_year': record['year'], 'event_start_utc': event.event_start_utc,
                                            'event_end_utc': event.event_end_utc, 'covering_hours': event.covering_hours})
                    if track in series_paths:
                        table = pa.Table.from_pandas(series, preserve_index=False)
                        if track not in writers:
                            writers[track] = pq.ParquetWriter(series_paths[track].with_suffix('.parquet.tmp'),
                                                             table.schema, compression='zstd')
                        writers[track].write_table(table)
            save_json(report / 'pipeline_status.json', {'stage': 'm4_replay', 'status': 'running',
                       'completed_station_years': i, 'total_station_years': len(selected_records), 'updated_at_utc': stamp()})
            if i % 5 == 0 or i == len(selected_records):
                print(f'M4 rainfall replay {i}/{len(selected_records)} station-years', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for path in series_paths.values():
        path.with_suffix('.parquet.tmp').replace(path)
    scores = pd.DataFrame(scores)
    checks = pd.DataFrame(checks).drop(columns='track').drop_duplicates()
    if len(checks) != len(requested) * len(protocol['peak_windows_minutes']):
        raise ValueError('Observed event-duration comparisons differ between the two conditioning tracks')
    actual = set(pd.DataFrame(cohort_rows).event_id)
    if actual != set(requested.event_id) or scores.groupby('track').event_id.nunique().ne(len(requested)).any():
        raise ValueError('Replay membership differs from the requested event cohort')
    checks.to_csv(release / 'audit_log/observed_peak_consistency.csv', index=False)
    bad = checks.loc[~checks.matches_within_tolerance]
    bad.to_csv(release / 'audit_log/observed_peak_exceedances.csv', index=False)
    if len(bad):
        raise ValueError('Catalogue and independently calculated envelope peaks exceed tolerance')
    if closure_max > protocol['peak_depth_tolerance_mm']:
        raise ValueError('Hourly precipitation conservation exceeds tolerance')
    pd.DataFrame(cohort_rows).to_csv(release / 'audit_log/replay_cohort.csv', index=False)
    scores.to_csv(release / 'data/envelope_event_scores.csv', index=False)
    main_tables, strata_tables = [], []
    for track, group in scores.groupby('track'):
        main, segmented = metrics_tables(group, protocol)
        main['track'], segmented['track'] = track, track
        main_tables.append(main)
        strata_tables.append(segmented)
    pd.concat(main_tables).to_csv(release / 'supplementary/envelope_replay_summary.csv', index=False)
    pd.concat(strata_tables).to_csv(release / 'supplementary/envelope_error_strata.csv', index=False)
    hourly = pd.DataFrame(hourly_rows)
    hourly.to_csv(release / 'audit_log/hourly_metric_sums.csv', index=False)
    keys = ['variable', 'track', 'method', 'stratum']
    numeric = [column for column in hourly.select_dtypes(include='number').columns if column not in ('year', 'closure_max')]
    pooled = hourly.groupby(keys).agg({**{name: 'sum' for name in numeric}, 'closure_max': 'max'}).reset_index()
    pooled = add_ratios(pooled).rename(columns={'hours': 'hourly_peak_n', 'ramp_n': 'within_hour_increment_n'})
    pooled.to_csv(release / 'supplementary/supplementary_hourly_errors.csv', index=False)
    return requested, scores, checks, roster, union, intersection, closure_max, series_paths


def diagnostic_tables(events, requested, scores, metadata, config, intersection, release):
    official = scores.loc[scores.track.eq('official_hourly') & scores.window_minutes.eq(15)].copy()
    joined = official.merge(events[['event_id', 'storm_profile_tercile', 'earliest_peak_ratio',
                                    'temporal_centroid_ratio', 'has_max_interval_tie']], on='event_id', validate='many_to_one')
    joined['underestimation_rate'] = (joined.observed_peak_mm - joined.predicted_peak_mm) / joined.observed_peak_mm
    minimum = read_json(ROOT / config['diagnostic_protocol'])
    cells = []
    for method, group in joined.groupby('method'):
        main = group[group.observed_peak_mm.ge(minimum['diagnostic_min_observed_15min_mm'] - 1e-10)]
        for label in (*PROFILES, 'high_intensity_pooled'):
            part = main.loc[main.high_intensity] if label == 'high_intensity_pooled' else main.loc[main.storm_profile_tercile.eq(label)]
            reported = len(part) >= minimum['minimum_events'] and part.station_wban.nunique() >= minimum['minimum_sites']
            cells.append({'group': label, 'method': method, 'events': len(part), 'sites': part.station_wban.nunique(),
                          'reported': reported, **(spread(part.underestimation_rate) if reported else {})})
    pd.DataFrame(cells).to_csv(release / 'technical_validation/peak15_profile_diagnostics.csv', index=False)
    joined.to_csv(release / 'technical_validation/peak15_event_diagnostics.csv', index=False)
    sensitivity = []
    for method, group in joined.groupby('method'):
        for label, part in (('primary_ge1mm', group[group.observed_peak_mm.ge(1 - 1e-10)]),
                            ('high_intensity', group[group.high_intensity])):
            common = part[part.station_wban.isin(intersection)]
            enough = len(common) >= minimum['minimum_events'] and common.station_wban.nunique() >= minimum['minimum_sites']
            union_median = part.underestimation_rate.median()
            common_median = common.underestimation_rate.median() if enough else np.nan
            difference = abs(common_median - union_median) * 100
            limit = 2 if label == 'primary_ge1mm' else 5
            sensitivity.append({'method': method, 'cohort': label, 'union_events': len(part),
                'union_sites': part.station_wban.nunique(), 'intersection_events': len(common),
                'intersection_sites': common.station_wban.nunique(), 'intersection_reportable': enough,
                'union_median_underestimation': union_median, 'intersection_median_underestimation': common_median,
                'absolute_median_difference_percentage_points': difference, 'reporting_limit_percentage_points': limit,
                'requires_expanded_table': bool(enough and difference > limit)})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(release / 'technical_validation/replay_roster_sensitivity.csv', index=False)
    expanded = bool(sensitivity.requires_expanded_table.any())
    if expanded:
        sensitivity.to_csv(release / 'supplementary/replay_roster_sensitivity.csv', index=False)
    sensitivity_decision = {'requires_supplementary_table': expanded, 'comparisons': sensitivity.to_dict('records')}
    save_json(release / 'audit_log/replay_roster_sensitivity_decision.json', sensitivity_decision)
    composition = []
    assigned = requested.assign(climate=requested.kg_group.fillna('Unclassified'))
    for dimension, labels in (('climate', ['A', 'B', 'C', 'D', 'E', 'Unclassified']),
                              ('network', sorted(metadata.network.unique()))):
        for label in labels:
            part = assigned.loc[assigned[dimension].eq(label)]
            high = part.loc[part.high_intensity.fillna(False)]
            composition.append({'dimension': dimension, 'group': label, 'replay_events': len(part),
                                'replay_sites': part.station_wban.nunique(), 'high_intensity_events': len(high),
                                'high_intensity_sites': high.station_wban.nunique()})
    coverage = pd.DataFrame(composition)
    coverage.to_csv(release / 'technical_validation/replay_coverage.csv', index=False)
    return coverage, sensitivity_decision


def event_validation_tables(events, release):
    confirmed = events[events.usability_tier.isin([1, 2])].copy()
    confirmed['period'] = pd.cut(confirmed.first_source_year, [2006, 2012, 2014, 2020, 2025],
                                labels=['2007-2012', '2013-2014', '2015-2020', '2021-2025'])
    counts = confirmed.groupby(['period', 'precip_phase_flag'], observed=True).agg(
        events=('event_id', 'size'), sites=('station_wban', 'nunique'), high_intensity_events=('high_intensity', 'sum')).reset_index()
    counts.to_csv(release / 'technical_validation/phase_coverage.csv', index=False)
    rows = []
    for period, group in confirmed.groupby('period', observed=True):
        paired = group[group.antecedent_dry_censored.eq(False) & group.soil_moisture_missing_reason_5cm.eq('valid')]
        rows.append({'period': str(period), 'confirmed_events': len(group),
                     'wetbulb_available_events': int(group.event_wetbulb_mean_c.notna().sum()),
                     'soil_5cm_available_events': int(group.soil_moisture_pre_5cm.notna().sum()),
                     'uncensored_with_soil_5cm_events': len(paired), 'paired_sites': paired.station_wban.nunique(),
                     'pearson_r': paired.antecedent_dry_hours_lower_bound.corr(paired.soil_moisture_pre_5cm),
                     'spearman_rho': paired.antecedent_dry_hours_lower_bound.corr(paired.soil_moisture_pre_5cm, method='spearman')})
    pd.DataFrame(rows).to_csv(release / 'technical_validation/soil_moisture_antecedent_dry.csv', index=False)


def finish_validation(events, requested, coverage, release, series_paths):
    event_path = release / 'data/events.parquet'
    temporary = event_path.with_suffix('.parquet.tmp')
    export_parquet(events, temporary)
    temporary.replace(event_path)
    index_path = release / 'data/portable_index.json'
    index = read_json(index_path)
    for record in index['records']:
        if 'events' in record['outputs']:
            record['outputs']['events'] = describe(release / 'data/events.parquet', len(events), release)
    index['records'].append({'outputs': {track: describe(path, pq.ParquetFile(path).metadata.num_rows, release)
                                         for track, path in series_paths.items()}})
    index['records'].append({'outputs': {'scores': describe(release / 'data/envelope_event_scores.csv',
        len(pd.read_csv(release / 'data/envelope_event_scores.csv', usecols=['event_id'])), release)}})
    catalogue = pd.read_csv(release / 'technical_validation/catalogue_coverage.csv')
    strata_counts = {'catalogue': {str(key): group.events.tolist() for key, group in catalogue.groupby('dimension')},
                     'replay': {str(key): group.replay_events.tolist() for key, group in coverage.groupby('dimension')}}
    for field in ('usability_tier', 'first_source_year', 'dataset_split'):
        strata_counts['catalogue'][field] = events.groupby(field, dropna=False).size().tolist()
    for field in ('first_source_year', 'network', 'kg_group'):
        strata_counts['replay'][field] = requested.groupby(field, dropna=False).size().tolist()
    for year in sorted(events.first_source_year.unique()):
        for variable in VARIABLES:
            pair = pq.ParquetFile(release / 'data/pairs' / f'variable={variable}' / f'source_year={year}' / 'part.parquet')
            annotation = pq.ParquetFile(release / 'data/annotations' / f'variable={variable}' / f'source_year={year}' / 'part.parquet')
            if pair.metadata.num_rows != annotation.metadata.num_rows:
                raise AssertionError(f'#6 Pair and annotation rows differ for {variable}/{year}')
    phase = pd.read_csv(release / 'technical_validation/phase_coverage.csv')
    if phase.events.sum() != events.usability_tier.isin([1, 2]).sum() or (
            phase.high_intensity_events.sum() != events.high_intensity.fillna(False).sum()):
        raise AssertionError('#6 Phase strata do not sum to confirmed and high-intensity event totals')
    assert_portable_index_and_strata(release, index, {'catalogue': len(events), 'replay': len(requested)}, strata_counts)
    save_json(index_path, index)
    return len([entry for record in index['records'] for entry in record['outputs'].values()])


def run(config_path):
    config = read_json(config_path)
    release = ROOT / config['release_directory']
    report = ROOT / config['reports_directory'] / 'm4'
    report.mkdir(parents=True, exist_ok=True)
    m3 = read_json(ROOT / config['reports_directory'] / 'm3/summary.json')
    if m3['status'] != 'passed' or set(m3['assertions'].values()) != {'passed'}:
        raise ValueError('M3 hard assertions have not passed')
    events_path = release / 'data/events.parquet'
    if sha256(events_path) != m3['event_catalogue_sha256']:
        raise ValueError('M3 event catalogue changed before M4')
    save_json(report / 'pipeline_status.json', {'stage': 'm4_reference', 'status': 'running', 'at_utc': stamp()})
    events = pd.read_parquet(events_path)
    assert_tier_mapping(events)
    assert_tier_partition_and_nulls(events)
    cutoff, reference = freeze_reference(events, config, release)
    save_json(report / 'pipeline_status.json', {'stage': 'm4_fit', 'status': 'running', 'at_utc': stamp(), 'reference': reference})
    print('M4 P95 reference frozen ' + json.dumps(reference), flush=True)
    base = read_json(ROOT / config['base_manifest'])
    records = [{**record, 'release_root': str(release)} for record in base['records']]
    metadata = pd.read_csv(release / 'data/station_metadata.csv', dtype={'station_wban': str})
    assert_station_and_feature_policy(metadata, {r['station_wban'] for r in records}, FEATURES,
                                      validation_tuning_trials=config['validation_tuning_trials'])
    fixed, models = model_fitting(records, config, release, report)
    protocol = read_json(release / 'provenance/event_protocol.json')
    requested, scores, checks, roster, union, intersection, closure_max, series_paths = run_rainfall_replay(
        events, records, config, protocol, cutoff, models, fixed, release, report)
    coverage, sensitivity_decision = diagnostic_tables(events, requested, scores, metadata, config, intersection, release)
    event_validation_tables(events, release)
    indexed = finish_validation(events, requested, coverage, release, series_paths)
    summary = {'stage': 'm4', 'status': 'passed', 'finished_at_utc': stamp(), 'reference': reference,
               'fitted_models': len(models), 'validation_tuning_trials': 0, 'replay_events': len(requested),
               'replay_sites_union': len(union), 'replay_sites_intersection': len(intersection),
               'replay_sites_per_year': {str(year): len(wbans) for year, wbans in roster.items()},
               'score_rows': len(scores), 'observed_peak_checks': len(checks),
               'observed_peak_exceedances': int((~checks.matches_within_tolerance).sum()),
               'max_observed_peak_difference_mm': float(checks.signed_difference_mm.abs().max()),
               'max_hourly_closure_error_mm': closure_max, 'verified_indexed_data_files': indexed,
               'portable_index_sha256': sha256(release / 'data/portable_index.json'),
               'roster_sensitivity': sensitivity_decision,
               'assertions': {str(i): 'passed' for i in range(1, 7)}}
    save_json(release / 'audit_log/validation_summary.json', summary)
    macro_path = ROOT / 'manuscript/station_disaggregation_results.tex'
    render(release, macro_path)
    summary['generated_macro_sha256'] = sha256(macro_path)
    save_json(release / 'audit_log/validation_summary.json', summary)
    save_json(report / 'summary.json', summary)
    save_json(report / 'pipeline_status.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException as error:
        config = read_json(args.config)
        save_json(ROOT / config['reports_directory'] / 'm4/pipeline_status.json',
                  {'stage': 'm4', 'status': 'failed', 'at_utc': stamp(), 'error': f'{type(error).__name__}: {error}'})
        raise
