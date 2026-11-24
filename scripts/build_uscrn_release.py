"""M3 release builder using the frozen NOAA event, mask and profile routines."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from uscrn_assertions import (
    assert_soil_moisture_attribution, assert_station_and_feature_policy,
    assert_tier_mapping, assert_tier_partition_and_nulls,
)
from benchmark_full import FEATURES, ROOT, VARIABLES, save_json, sha256
from build_precipitation_event_library import (
    assemble_sensitivity, catalogue_block, read_station_year, sensitivity_results, sensitivity_sample,
)
from precipitation_station_metadata import CLIMATE, sample_climate
from station_release_tables import (
    AnnualRain, HOUR, STEP, annotate, bound, describe, enrich_events, export_parquet, origin, positions,
)
from station_disaggregation_diagnostics import event_coverage
from uscrn_event_diagnostics import STRING_DIAGNOSTICS, add_event_diagnostics


DEFAULT_CONFIG = ROOT / 'config/uscrn_2007_2025_release.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def metadata_for_release(base, source, release, event_settings):
    entry = next(row for row in source['records'] if row['relative_source_path'] == 'stations.tsv')
    path = ROOT / entry['path']
    if sha256(path) != entry['sha256']:
        raise ValueError('NOAA metadata differs from the source manifest')
    noaa = pd.read_csv(path, sep='\t', dtype={'WBAN': str})
    noaa = noaa.loc[noaa.WBAN.str.fullmatch(r'\d{5}', na=False)]
    fields = {'WBAN': 'station_wban', 'NETWORK': 'network', 'LATITUDE': 'latitude',
              'LONGITUDE': 'longitude', 'ELEVATION': 'elevation_ft', 'COUNTRY': 'country', 'STATE': 'state'}
    observed = pd.DataFrame(base['records'])[['station_wban', 'station', 'station_split']].drop_duplicates()
    if observed.station_wban.duplicated().any():
        raise ValueError('Multiple archive identities for a WBAN require review')
    metadata = observed.merge(noaa[list(fields)].rename(columns=fields), on='station_wban',
                              how='left', validate='one_to_one', indicator=True)
    metadata['noaa_catalogue_match'] = metadata.pop('_merge').eq('both')
    metadata['elevation_ft'] = pd.to_numeric(metadata.elevation_ft, errors='raise')
    metadata['elevation_m'] = (metadata.elevation_ft * 0.3048).round(4)
    metadata['elevation_status'] = np.where(metadata.elevation_ft.notna(), 'matched', 'missing')
    metadata['dataset_split'] = metadata.station_split
    climate_source = read_json(CLIMATE / 'source.json')
    if climate_source['archive_sha256'] != event_settings['metadata']['climate_zip_sha256']:
        raise ValueError('Cached climate archive differs from the frozen source')
    for name, digest in climate_source['extracted_sha256'].items():
        if sha256(CLIMATE / name) != digest:
            raise ValueError(f'Cached climate input changed: {name}')
    metadata, grid = sample_climate(metadata, CLIMATE / Path(climate_source['raster_member']).name,
                                     CLIMATE / Path(climate_source['legend_member']).name)
    for field in ('kg_raster_row', 'kg_raster_col', 'kg_numeric'):
        metadata[field] = metadata[field].astype('Int64')
    metadata = metadata.sort_values('station_wban').reset_index(drop=True)
    metadata.to_csv(release / 'data/station_metadata.csv', index=False)
    save_json(release / 'provenance/station_metadata_manifest.json', {
        'source_path': entry['path'], 'source_sha256': entry['sha256'], 'stations': len(metadata),
        'noaa_matches': int(metadata.noaa_catalogue_match.sum()),
        'climate_match_status': metadata.kg_match_status.value_counts().to_dict(),
        'climate_source_sha256': sha256(CLIMATE / 'source.json'), 'raster_grid': grid,
    })
    return metadata


def partition_masks(record, folder, settings):
    """Use the published masks with the full-year grid and preserve source columns."""
    high_entry, pair_entry = [record['outputs'][name] for name in ('high_frequency', 'hourly_pairs')]
    for entry in (high_entry, pair_entry):
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f'Base partition checksum differs: {entry["path"]}')
    fine = pq.ParquetFile(ROOT / high_entry['path']).read(columns=[
        'time_end_utc', 'station_wban', *VARIABLES, *(f'{variable}_exclusion_code' for variable in VARIABLES),
    ]).to_pandas()
    hourly = pq.ParquetFile(ROOT / pair_entry['path']).read().to_pandas()
    if len(fine) != high_entry['rows'] or len(hourly) != pair_entry['rows']:
        raise ValueError('Base partition row count differs')
    if not hourly.station_wban.eq(record['station_wban']).all():
        raise ValueError('Base hourly station identity differs')
    year = record['year']
    hours = int((origin(year + 1) - origin(year)) / HOUR)
    fpos = positions(fine.time_end_utc, year, STEP, hours * 12)
    hourly['dataset_split'] = record['station_split']
    hourly['is_valid_target'] = False
    for track in ('official', 'derived'):
        hourly[f'is_valid_input_{track}'] = False
        hourly[f'track_eligible_{track}'] = False
    annotations, quality_rows = [], []
    eligibility = {}
    for variable in VARIABLES:
        selection = hourly.variable.eq(variable)
        target_hours = hourly.loc[selection]
        hpos = positions(target_hours.time_end_utc, year, HOUR, hours)
        values = fine[variable].to_numpy(dtype=float)
        accepted = fine[f'{variable}_exclusion_code'].eq(0).to_numpy()
        if not np.array_equal(np.isfinite(values), accepted):
            raise ValueError(f'Fine values and codes differ: {record["station"]}/{year}/{variable}')
        physical = accepted & bound(values, variable)
        fine_counts = np.bincount(fpos[accepted] // 12, minlength=hours)
        physical_counts = np.bincount(fpos[physical] // 12, minlength=hours)
        target = physical_counts == 12
        if not np.array_equal(fine_counts[hpos], target_hours.valid_five_minute_count):
            raise ValueError('Fine target counts differ from the base pairing table')
        hourly.loc[selection, 'is_valid_target'] = target[hpos]
        for track in ('official', 'derived'):
            coarse = np.full(hours, np.nan)
            coarse[hpos] = target_hours[f'{track}_hourly'].to_numpy(dtype=float)
            valid = bound(coarse, variable)
            padded = np.r_[False, valid, False]
            input_valid = valid & padded[:-2] & padded[2:]
            eligible = input_valid & target
            hourly.loc[selection, f'is_valid_input_{track}'] = input_valid[hpos]
            hourly.loc[selection, f'track_eligible_{track}'] = eligible[hpos]
            eligibility[(variable, track)] = eligible
            quality_rows.append({'station_wban': record['station_wban'], 'source_year': year,
                                 'variable': variable, 'track': track, 'bounded_target_hours': int(target.sum()),
                                 'valid_input_hours': int(input_valid.sum()), 'published_eligible_hours': int(eligible.sum())})
        bad = np.bincount(fpos[accepted & ~physical] // 12, minlength=hours) > 0
        coarse = target_hours.derived_hourly.to_numpy(dtype=float)
        complete = fine_counts[hpos] == 12
        zeros = complete & ~bad[hpos] & (coarse == 0) if variable != 'air_temperature' else np.zeros(len(coarse), bool)
        upper = complete & ~bad[hpos] & (coarse == 100) if variable == 'relative_humidity' else np.zeros(len(coarse), bool)
        labels = np.select([~complete, bad[hpos], zeros, upper],
                           ['incomplete_targets', 'observed_bound_violation', 'unique_zero_completion', 'unique_upper_completion'],
                           default='aggregate_nonunique')
        zero_count = np.bincount(fpos[accepted & (values == 0)] // 12, minlength=hours)
        wet_count = np.bincount(fpos[accepted & (values > 0)] // 12, minlength=hours)
        annotations.append(pd.DataFrame({
            'time_end_utc': target_hours.time_end_utc.to_numpy(), 'station': record['station'],
            'station_wban': record['station_wban'], 'source_year': year, 'variable': variable,
            'valid_fine_count': fine_counts[hpos], 'derived_coarse': coarse, 'constraint_class': labels,
            'fine_zero_count': zero_count[hpos], 'fine_positive_count': wet_count[hpos],
        }))
    annotation = pd.concat(annotations, ignore_index=True)
    annotate(annotation, hourly, year, settings['diagnostic_flags'])
    pair_path, annotation_path = folder / f'pairs_{year}.parquet', folder / f'annotations_{year}.parquet'
    export_parquet(hourly, pair_path)
    export_parquet(annotation, annotation_path)

    annual = read_station_year(record, None, fine_frame=fine, hourly_frame=hourly)
    official = eligibility[('precipitation', 'official')]
    if not np.array_equal(annual.eligible, official):
        raise AssertionError('#1 Original hourly event eligibility differs from the published bounded mask')
    blocker = np.maximum.accumulate(np.where(annual.amounts == 0, -1, np.arange(len(annual.amounts))))
    rain_hourly = hourly.loc[hourly.variable.eq('precipitation')].set_index('time_end_utc')
    official_values = rain_hourly.official_hourly.reindex(pd.date_range(origin(year) + HOUR, periods=hours, freq=HOUR)).to_numpy()
    rain = AnnualRain(year, int(fpos.min()), annual.amounts, annual.quality, blocker, annual.hour_complete,
                      official, eligibility[('precipitation', 'derived')],
                      np.zeros(hours, dtype=bool), official_values)
    return annual, rain, quality_rows, {'year': year, 'pairs_path': str(pair_path),
                                        'annotations_path': str(annotation_path), 'rows': len(hourly)}


def station_job(job):
    records, station_dict, settings, protocol, staging, fingerprint, sampled = job
    station = pd.Series(station_dict)
    folder = Path(staging) / station.station_wban
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'receipt.json'
    if receipt.is_file():
        cached = read_json(receipt)
        if cached['fingerprint'] == fingerprint and all(
                Path(item['path']).is_file() and sha256(Path(item['path'])) == item['sha256']
                for item in cached['outputs']):
            return cached
    annuals, rain_years, quality, partitions, sensitivity = [], {}, [], [], []
    annual_counts = []
    for record in records:
        annual, rain, counts, part = partition_masks(record, folder, settings)
        annuals.append(annual)
        rain_years[annual.year] = rain
        quality.extend(counts)
        partitions.append(part)
        annual_counts.append({'station_wban': station.station_wban, 'source_year': annual.year,
                              'network': station.network, 'kg_class': station.kg_class, 'kg_group': station.kg_group,
                              'station_split': station.station_split, **annual.counts})
        if (station.station_wban, annual.year) in sampled:
            sensitivity.extend(sensitivity_results(annual, station, protocol))
    rows = []
    for _, sequence in groupby(enumerate(annuals), key=lambda item: item[1].year - item[0]):
        rows.extend(catalogue_block([item[1] for item in sequence], station, protocol))
    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError(f'No observed precipitation segment for station {station.station_wban}')
    additions = pd.DataFrame(enrich_events(events, rain_years, settings))
    audit_columns = ['event_id', 'antecedent_stop_reason', 'antecedent_terminator_utc',
                     'ineligible_derived_covering_hours']
    diagnostics = additions[audit_columns]
    events = events.merge(additions.drop(columns=audit_columns[1:]), on='event_id', validate='one_to_one')
    events['dataset_split'] = station.station_split
    events['comparison_ready'] = events.usability_tier.eq(1)
    events['has_max_interval_tie'] = events.has_max_interval_tie.astype('boolean')
    events['storm_profile_tercile'] = events.storm_profile_tercile.astype('string')
    events['usability_tier'] = events.usability_tier.astype('int8')
    events, first_valid = add_event_diagnostics(events, records)
    try:
        assert_tier_mapping(events)
        assert_tier_partition_and_nulls(events)
        assert_soil_moisture_attribution(events, first_valid)
    except AssertionError:
        export_parquet(events, folder / 'assertion_failed_events.parquet')
        raise
    paths = {'events': folder / 'events.parquet', 'diagnostics': folder / 'event_diagnostics.parquet'}
    export_parquet(events, paths['events'])
    export_parquet(diagnostics, paths['diagnostics'])
    outputs = [{'path': str(path), 'sha256': sha256(path)} for path in paths.values()]
    outputs.extend({'path': part[name], 'sha256': sha256(Path(part[name]))}
                   for part in partitions for name in ('pairs_path', 'annotations_path'))
    result = {'fingerprint': fingerprint, 'station_wban': station.station_wban, 'events': len(events),
              'events_path': str(paths['events']), 'diagnostics_path': str(paths['diagnostics']),
              'partitions': partitions, 'quality': quality, 'annual_counts': annual_counts,
              'sensitivity': sensitivity, 'first_valid': {str(depth): str(value) if pd.notna(value) else None
                for depth, value in first_valid[station.station_wban].items()}, 'outputs': outputs}
    save_json(receipt, result)
    return result


def consolidate_tables(results, release):
    records = []
    years = sorted({part['year'] for result in results for part in result['partitions']})
    for year in years:
        partitions = [part for result in results for part in result['partitions'] if part['year'] == year]
        outputs = {variable: {} for variable in VARIABLES}
        for layer, input_name in (('pairs', 'pairs_path'), ('annotations', 'annotations_path')):
            writers, destinations = {}, {}
            try:
                for part in partitions:
                    frame = pq.ParquetFile(part[input_name]).read().to_pandas()
                    for variable in VARIABLES:
                        path = release / 'data' / layer / f'variable={variable}' / f'source_year={year}' / 'part.parquet'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        table = pa.Table.from_pandas(frame.loc[frame.variable.eq(variable)], preserve_index=False)
                        if variable not in writers:
                            temporary = path.with_suffix('.parquet.tmp')
                            writers[variable] = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                            destinations[variable] = (temporary, path)
                        writers[variable].write_table(table)
            finally:
                for writer in writers.values():
                    writer.close()
            for variable, (temporary, path) in destinations.items():
                temporary.replace(path)
                outputs[variable][layer] = describe(path, pq.ParquetFile(path).metadata.num_rows, release)
        for variable in VARIABLES:
            records.append({'source_year': year, 'variable': variable, 'outputs': outputs[variable]})
        print(f'M3 consolidated variable partitions for {year}', flush=True)
    return records


def run(config_path, workers):
    config = read_json(config_path)
    base = read_json(ROOT / config['base_manifest'])
    source = read_json(ROOT / config['source_manifest'])
    m2 = read_json(ROOT / config['reports_directory'] / 'm2/summary.json')
    if (sha256(ROOT / config['base_manifest']) != m2['build_manifest_sha256']
            or sha256(ROOT / config['source_manifest']) != m2['source_manifest_sha256']):
        raise ValueError('M2 and the M3 input snapshots differ')
    records = sorted(base['records'], key=lambda row: (row['station_wban'], row['year']))
    if any(row['status'] != 'success' for row in records) or base['summary']['unpaired_or_unverified']:
        raise ValueError('Base build is incomplete')
    release, staging = ROOT / config['release_directory'], ROOT / config['staging_directory']
    for name in ('data', 'provenance', 'audit_log', 'technical_validation', 'supplementary'):
        (release / name).mkdir(parents=True, exist_ok=True)
    report = ROOT / config['reports_directory'] / 'm3'
    report.mkdir(parents=True, exist_ok=True)
    state_path = report / 'pipeline_status.json'
    save_json(state_path, {'stage': 'm3', 'status': 'running', 'started_at_utc': stamp()})
    settings = read_json(ROOT / config['diagnostic_protocol'])
    protocol = read_json(ROOT / config['event_protocol'])
    protocol['source_years'] = sorted({r['year'] for r in records})
    protocol['reference']['source_years'] = config['reference_years']
    protocol['envelope_replay']['evaluation_years'] = config['replay_years']
    protocol['envelope_replay']['cohort'] = 'Boundary-confirmed official-eligible test-station events in the replay window'
    metadata = metadata_for_release(base, source, release, protocol)
    assert_station_and_feature_policy(metadata, {r['station_wban'] for r in records}, FEATURES,
                                      validation_tuning_trials=config['validation_tuning_trials'])
    sites = metadata.set_index('station_wban').copy()
    sites[['kg_class', 'kg_group']] = sites[['kg_class', 'kg_group']].fillna('Unclassified')
    sample = sensitivity_sample(records, sites, protocol)
    sample.to_csv(release / 'supplementary/dry_gap_sample.csv', index=False)
    sampled = set(zip(sample.station_wban, sample.year))
    dependencies = ('build_uscrn_release.py', 'build_precipitation_event_library.py',
                    'station_release_tables.py', 'uscrn_event_diagnostics.py', 'uscrn_assertions.py')
    fingerprint = hashlib.sha256('|'.join([sha256(Path(__file__).with_name(name)) for name in dependencies]
        + [sha256(config_path), sha256(ROOT / config['base_manifest']), sha256(release / 'data/station_metadata.csv')]).encode()).hexdigest()
    jobs = []
    for wban, group in groupby(records, key=lambda row: row['station_wban']):
        station = sites.loc[wban].to_dict()
        station['station_wban'] = wban
        jobs.append((list(group), station, settings, protocol, str(staging), fingerprint, sampled))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(station_job, jobs, chunksize=1):
                results.append(result)
                save_json(state_path, {'stage': 'm3', 'status': 'running', 'completed_stations': len(results),
                                      'total_stations': len(jobs), 'updated_at_utc': stamp()})
                print(f'M3 event extraction {len(results)}/{len(jobs)} WBANs, last {result["station_wban"]}', flush=True)
        events = pd.concat([pd.read_parquet(row['events_path']) for row in results], ignore_index=True)
        if events.event_id.duplicated().any():
            raise ValueError('Duplicate event IDs')
        first_valid = {row['station_wban']: {int(depth): pd.Timestamp(value) if value else pd.NaT
                                            for depth, value in row['first_valid'].items()} for row in results}
        assert_tier_mapping(events)
        assert_tier_partition_and_nulls(events)
        assert_soil_moisture_attribution(events, first_valid)
        export_parquet(events, release / 'data/events.parquet')
        arrow = pq.ParquetFile(release / 'data/events.parquet').read(columns=['usability_tier', *STRING_DIAGNOSTICS])
        tier_three = arrow.filter(pc.equal(arrow['usability_tier'], 3))
        if any(tier_three[name].null_count != tier_three.num_rows for name in STRING_DIAGNOSTICS):
            raise AssertionError('#2 Tier 3 strings failed the physical Parquet null check')
        pd.concat([pd.read_parquet(row['diagnostics_path']) for row in results]).to_csv(
            release / 'audit_log/event_eligibility_diagnostics.csv', index=False)
        annual = pd.DataFrame([row for result in results for row in result['annual_counts']])
        annual.to_csv(release / 'technical_validation/station_year_coverage.csv', index=False)
        pd.DataFrame([row for result in results for row in result['quality']]).to_csv(
            release / 'audit_log/eligibility_by_station_year.csv', index=False)
        event_coverage(events, metadata, annual, release)
        sensitivity = [item for result in results for item in result['sensitivity']]
        per_year, dry_summary = assemble_sensitivity(sensitivity)
        per_year.to_csv(release / 'supplementary/dry_gap_sensitivity.csv', index=False)
        dry_summary.to_csv(release / 'supplementary/dry_gap_sensitivity_summary.csv', index=False)
        index_records = consolidate_tables(results, release)
        for record in records:
            entry = record['outputs']['high_frequency']
            destination = release / 'data/observations' / f'source_year={record["year"]}' / f'station={record["station"]}' / 'observations.parquet'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256(destination) != entry['sha256']:
                    raise ValueError(f'Existing observation differs: {destination}')
            else:
                os.link(ROOT / entry['path'], destination)
            index_records.append({'station_wban': record['station_wban'], 'source_year': record['year'],
                'outputs': {'high_frequency': {**entry, 'path': destination.relative_to(release).as_posix()}}})
        index_records.append({'outputs': {
            'events': describe(release / 'data/events.parquet', len(events), release),
            'station_metadata': describe(release / 'data/station_metadata.csv', len(metadata), release)}})
        save_json(release / 'data/portable_index.json', {'schema_version': 2, 'records': index_records,
                   'source_manifest_sha256': m2['source_manifest_sha256'],
                   'base_manifest_sha256': m2['build_manifest_sha256']})
        save_json(release / 'provenance/sensor_first_valid.json', {
            wban: {str(depth): str(value) if pd.notna(value) else None for depth, value in depths.items()}
            for wban, depths in first_valid.items()})
        save_json(release / 'provenance/event_protocol.json', protocol)
        save_json(release / 'provenance/release_configuration.json', config)
        assertions = {'1': 'passed', '2': 'passed', '4': 'passed', '5': 'passed'}
        summary = {'stage': 'm3', 'status': 'passed', 'finished_at_utc': stamp(),
                   'stations': len(metadata), 'station_years': len(records), 'events': len(events),
                   'tiers': events.usability_tier.value_counts().sort_index().to_dict(),
                   'assertions': assertions, 'fingerprint': fingerprint,
                   'event_catalogue_sha256': sha256(release / 'data/events.parquet')}
        save_json(report / 'summary.json', summary)
        save_json(state_path, summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        save_json(state_path, {'stage': 'm3', 'status': 'failed', 'at_utc': stamp(),
                              'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    try:
        run(args.config, args.workers)
    except BaseException as error:
        config = read_json(args.config)
        save_json(ROOT / config['reports_directory'] / 'm3/pipeline_status.json',
                  {'stage': 'm3', 'status': 'failed', 'at_utc': stamp(),
                   'error': f'{type(error).__name__}: {error}'})
        raise
