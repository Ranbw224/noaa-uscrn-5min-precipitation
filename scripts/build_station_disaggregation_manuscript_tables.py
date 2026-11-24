"""Generate LaTeX results from the rebuilt release and its audit products."""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from benchmark_full import ROOT, sha256
from station_disaggregation_diagnostics import METHODS, PROFILES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def count(number):
    return f'{int(number):,}'


def row(*cells):
    return ' & '.join(str(cell) for cell in cells) + r' \\'


def fixed(number, digits=3):
    value = f'{float(number):.{digits}f}'
    return fr'\({value}\)' if value.startswith('-') else value


def coverage_rows(frame, numeric):
    output = []
    for dimension, subset in frame.groupby('dimension', sort=False):
        for index, entry in enumerate(subset.itertuples(index=False)):
            group = {'<500 m': r'\(<500\) m', '500–1500 m': '500--1,500 m',
                     '>1500 m': r'\(>1500\) m'}.get(entry.group, entry.group)
            output.append(row(dimension.capitalize() if index == 0 else '', group,
                              *(count(getattr(entry, column)) for column in numeric)))
    return '\n'.join(output)


def year_range(years):
    return f'{min(years)}--{max(years)}'


def source_audit_distributions(technical, reports, sites, m2, recent_years):
    """Join existing audit counts to networks and export their station-year denominators."""
    station_keys = ['station_wban', 'source_year']
    variable_keys = [*station_keys, 'variable']
    dtype = {'station_wban': str}
    networks = sites[['station_wban', 'network']]
    endpoints = pd.read_csv(reports / 'm2/endpoint_consistency_by_station_year.csv', dtype=dtype)
    endpoints = endpoints.merge(networks, on='station_wban', how='left', validate='many_to_one')
    endpoint_totals = m2['T_CALC_endpoint']
    if (endpoints.duplicated(station_keys).any() or endpoints.network.isna().any()
            or len(endpoints) != m2['scope']['paired_station_years']
            or not endpoints.hourly_source_rows.eq(endpoints.compared_hours + endpoints.unscored_hours).all()
            or any(endpoints[column].sum() != endpoint_totals[column]
                   for column in endpoints.columns.intersection(endpoint_totals))):
        raise ValueError('Temperature endpoint distribution does not reconcile with M2')
    endpoints['excluded_fraction'] = endpoints.unscored_hours / endpoints.hourly_source_rows

    cross = pd.read_csv(reports / 'base/full_cross_resolution_consistency.csv', dtype=dtype).rename(columns={
        'year': 'source_year', 'eligible_hours': 'compared_hours', 'above_tolerance_count': 'anomalous_hours',
        'diagnostic_tolerance': 'absolute_tolerance', 'total_official_hours': 'hourly_source_rows',
    })
    cross = cross.merge(networks, on='station_wban', how='left', validate='many_to_one')
    observed = pd.read_csv(reports / 'm2/cross_product_exception_hours.csv',
                           usecols=[*variable_keys, 'time_end_utc'], dtype=dtype)
    counted = cross.set_index(variable_keys).anomalous_hours
    flagged = observed.groupby(variable_keys).size()
    variable_summary = pd.DataFrame(m2['cross_product']).set_index('variable')
    measures = ['compared_hours', 'anomalous_hours']
    if (cross.duplicated(variable_keys).any() or cross.network.isna().any()
            or not cross.groupby('variable').size().eq(len(endpoints)).all()
            or not pd.MultiIndex.from_frame(cross[station_keys].drop_duplicates()).sort_values().equals(
                pd.MultiIndex.from_frame(endpoints[station_keys]).sort_values())
            or observed.duplicated([*variable_keys, 'time_end_utc']).any()
            or len(observed[['station_wban', 'time_end_utc']].drop_duplicates())
            != m2['cross_product_anomalous_unique_station_hours']
            or len(flagged.index.difference(counted.index))
            or not counted.eq(flagged.reindex(counted.index, fill_value=0)).all()
            or not cross.groupby('variable')[measures].sum().equals(variable_summary[measures].sort_index())
            or cross.anomalous_hours.sum() != m2['cross_product_anomalous_variable_hours']):
        raise ValueError('Cross-product exception distribution does not reconcile with M2')
    cross['anomaly_fraction'] = cross.anomalous_hours / cross.compared_hours.where(cross.compared_hours > 0)

    identifiers = ['source_year', 'network', 'station_wban', 'station']
    for name, frame, columns in (
        ('temperature_endpoint_exclusions.csv', endpoints, [
            'hourly_source_rows', 'compared_hours', 'equal_hours', 'unequal_hours',
            'missing_T_CALC_hours', 'missing_fine_endpoint_hours', 'unscored_hours', 'excluded_fraction']),
        ('cross_product_exception_distribution.csv', cross, [
            'variable', 'hourly_source_rows', 'compared_hours', 'anomalous_hours',
            'absolute_tolerance', 'anomaly_fraction']),
    ):
        destination = technical / name
        temporary = destination.with_suffix('.csv.tmp')
        frame.sort_values([*identifiers, *(['variable'] if 'variable' in columns else [])])[
            [*identifiers, *columns]].to_csv(temporary, index=False, lineterminator='\n')
        temporary.replace(destination)

    excluded = endpoints.unscored_hours.sum()
    anomalies = cross.anomalous_hours.sum()
    recent_exclusions = endpoints.loc[endpoints.source_year.isin(recent_years), 'unscored_hours'].sum()
    endpoint_by_network = endpoints.groupby('network').unscored_hours.sum()
    exceptions_by_network = cross.groupby('network').anomalous_hours.sum()
    return {
        'EndpointRecentShare': f'{100 * recent_exclusions / excluded:.1f}',
        'EndpointUSCRNShare': f'{100 * endpoint_by_network["USCRN"] / excluded:.1f}',
        'EndpointTopWBAN': endpoints.groupby('station_wban').unscored_hours.sum().idxmax(),
        'ExceptionYears': count(cross.loc[cross.anomalous_hours.gt(0), 'source_year'].nunique()),
        'ExceptionUSCRNShare': f'{100 * exceptions_by_network["USCRN"] / anomalies:.1f}',
        'ExceptionMaxStationShare': f'{100 * cross.groupby("station_wban").anomalous_hours.sum().max() / anomalies:.1f}',
    }


def validation_macros(release, audit, m2, protocol, confirmed):
    """Render the observed phase and coverage results, audit denominators and replay checks."""
    technical = release / 'technical_validation'
    reference = audit['reference']
    p95 = pd.read_csv(technical / 'p95_window_sensitivity.csv').sort_values('first_year')
    if (len(p95) != 3 or not p95.station_split.eq(reference['reference_split']).all()
            or p95.iloc[0].peak_15min_p95_mm != reference['cutoff_mm_per_15min']
            or p95.iloc[0].events != reference['reference_events']):
        raise ValueError('The P95 sensitivity table differs from the frozen reference')
    endpoint = m2['T_CALC_endpoint']
    if (endpoint['compared_hours'] + endpoint['unscored_hours'] != endpoint['hourly_source_rows']
            or endpoint['equal_hours'] + endpoint['unequal_hours'] != endpoint['compared_hours']):
        raise ValueError('The temperature endpoint denominators do not reconcile')
    roster = max(entry['absolute_median_difference_percentage_points']
                 for entry in audit['roster_sensitivity']['comparisons'])
    phases = (pd.read_csv(technical / 'phase_coverage.csv').groupby('precip_phase_flag')
              [['events', 'high_intensity_events']].sum())
    high = confirmed.loc[confirmed.high_intensity.fillna(False)]
    if (not phases.events.eq(confirmed.precip_phase_flag.value_counts()).all()
            or not phases.high_intensity_events.eq(high.precip_phase_flag.value_counts()
                                                   .reindex(phases.index, fill_value=0)).all()):
        raise ValueError('Phase totals do not match the event catalogue')
    paired = confirmed.loc[confirmed.antecedent_dry_censored.eq(False)
                           & confirmed.soil_moisture_missing_reason_5cm.eq('valid')]
    dry, soil = paired.antecedent_dry_hours_lower_bound, paired.soil_moisture_pre_5cm
    coverage = pd.read_csv(technical / 'soil_moisture_antecedent_dry.csv')
    if (coverage.confirmed_events.sum() != len(confirmed)
            or coverage.wetbulb_available_events.sum() != confirmed.event_wetbulb_mean_c.count()
            or coverage.soil_5cm_available_events.sum() != confirmed.soil_moisture_pre_5cm.count()
            or coverage.uncensored_with_soil_5cm_events.sum() != len(paired)):
        raise ValueError('Period diagnostic coverage does not match the event catalogue')
    example_id = protocol['envelope_replay']['example_event_id']
    case_rows = confirmed.loc[confirmed.event_id.eq(example_id)]
    if len(case_rows) != 1:
        raise ValueError('The illustrated event is missing or duplicated')
    case = case_rows.iloc[0]
    start, end = case.event_start_utc, case.event_end_utc
    from_hour, to_hour = case.first_covering_hour_end_utc - pd.Timedelta(hours=1), case.last_covering_hour_end_utc
    steps = int(pd.Timedelta(hours=1) / pd.Timedelta(minutes=protocol['interval_minutes']))
    values = {
        'EndpointSourceRows': count(endpoint['hourly_source_rows']),
        'EndpointMissing': count(endpoint['unscored_hours']),
        'EndpointMismatches': count(endpoint['unequal_hours']),
        'ExceptionUniqueHours': count(m2['cross_product_anomalous_unique_station_hours']),
        'TempAndRainPass': count(sum(year['temperature_and_rain_pass'] for year in m2['availability'])),
        'AllFourPass': count(sum(year['all_four_pass'] for year in m2['availability'])),
        'RecentReferencePeriod': year_range([p95.iloc[1].first_year, p95.iloc[1].last_year]),
        'PostRetirementReferencePeriod': year_range([p95.iloc[2].first_year, p95.iloc[2].last_year]),
        'ReferenceCutoff': f'{p95.iloc[0].peak_15min_p95_mm:.1f}',
        'RecentReferenceCutoff': f'{p95.iloc[1].peak_15min_p95_mm:.1f}',
        'PostRetirementReferenceCutoff': f'{p95.iloc[2].peak_15min_p95_mm:.1f}',
        'RosterDeltaUpper': f'{(np.floor(roster * 100) + 1) / 100:.2f}',
        'ObservedPeakExceedances': count(audit['observed_peak_exceedances']),
        'CatalogueHighCount': count(len(high)),
        'SoilDryEvents': count(len(paired)), 'SoilDrySites': count(paired.station_wban.nunique()),
        'SoilDryPearson': fixed(dry.corr(soil)), 'SoilDrySpearman': fixed(dry.corr(soil, method='spearman')),
        'WindCoveragePct': fixed(100 * confirmed.event_wind_mean_ms.count() / len(confirmed), 1),
        'CaseDate': f'{start.day} {start:%B %Y}', 'CaseWBAN': case.station_wban,
        'CaseObservedTimes': f'{start:%H:%M}--{end:%H:%M}',
        'CaseEnvelopeTimes': f'{from_hour:%H:%M}--{to_hour:%H:%M}',
        'CaseBins': count(case.covering_hours * steps),
        'SoilDryCoverageRows': '\n'.join(row(
            item.period.replace('-', '--'), count(item.confirmed_events),
            fixed(100 * item.wetbulb_available_events / item.confirmed_events, 1),
            fixed(100 * item.soil_5cm_available_events / item.confirmed_events, 1),
            count(item.uncensored_with_soil_5cm_events), count(item.paired_sites),
            fixed(item.pearson_r), fixed(item.spearman_rho)) for item in coverage.itertuples(index=False)),
    }
    for label, phase in (('Liquid', 'liquid_likely'), ('Frozen', 'frozen_likely'),
                         ('Mixed', 'mixed_or_transitional'), ('TempMissing', 'temp_missing')):
        values['Phase' + label] = count(phases.loc[phase, 'events'])
        values['HighPhase' + label] = count(phases.loc[phase, 'high_intensity_events'])
    return values


def render(release, destination):
    """Inject measured 2007–2025 values only after all six data assertions pass."""
    audit = read_json(release / 'audit_log/validation_summary.json')
    if set(audit['assertions'].values()) != {'passed'} or set(audit['assertions']) != set('123456'):
        raise ValueError('All six 2007–2025 hard assertions must pass before macro generation')
    if sha256(release / 'data/portable_index.json') != audit['portable_index_sha256']:
        raise ValueError('The validated portable index changed before macro generation')
    configuration = read_json(release / 'provenance/release_configuration.json')
    m2 = read_json(release / 'provenance/m2_summary.json')
    protocol = read_json(release / 'provenance/event_protocol.json')
    base = read_json(ROOT / configuration['base_manifest'])
    sites = pd.read_csv(release / 'data/station_metadata.csv', dtype={'station_wban': str})
    events = pd.read_parquet(release / 'data/events.parquet')
    reference = audit['reference']
    tiers = events.usability_tier.value_counts()
    splits = sites.dataset_split.value_counts()
    requested = events.loc[events.usability_tier.eq(1) & events.dataset_split.eq('test')
                           & events.first_source_year.isin(configuration['replay_years'])]
    main = requested[requested.peak_15min_mm.ge(1 - 1e-10)]
    high = requested[requested.high_intensity.fillna(False)]
    earliest = np.select([main.earliest_peak_ratio < .33, main.earliest_peak_ratio > .67],
                         ['front_loaded', 'back_loaded'], default='mid_loaded')
    matches = int(main.storm_profile_tercile.eq(earliest).sum())
    by_variable = {entry['variable']: entry['anomalous_hours'] for entry in m2['cross_product']}
    # The Southwest retirement is the historical break specified for the coverage figure.
    transition_year = 2015
    before = {r['station_wban'] for r in base['records'] if r['year'] == transition_year - 1}
    after = {r['station_wban'] for r in base['records'] if r['year'] == transition_year}
    retired = before - after
    if any(r['station_wban'] in retired for r in base['records'] if r['year'] >= transition_year):
        raise ValueError('A retired station is active after the inventory discontinuity')
    confirmed = events[events.usability_tier.isin([1, 2])]
    stats = {
        'ArchivePeriod': year_range(protocol['source_years']), 'ArchiveYears': count(len(protocol['source_years'])),
        'ReferencePeriod': year_range(reference['reference_years']),
        'FittingPeriod': year_range(configuration['fitting_years']), 'FittingFirstYear': str(min(configuration['fitting_years'])),
        'ReplayPeriod': year_range(configuration['replay_years']),
        'RetirementTransitionYear': str(transition_year), 'LastPreRetirementYear': str(transition_year - 1),
        'EarlyArchivePeriod': year_range([min(protocol['source_years']), transition_year - 1]),
        'SitesBeforeRetirement': count(len(before)), 'SitesAfterRetirement': count(len(after)),
        'NewSitesAtRetirement': count(len(after - before)),
        'ValidationTuningTrials': count(audit['validation_tuning_trials']),
        'StationCount': count(len(sites)), 'StationYears': count(len(base['records'])),
        'TrainSites': count(splits.get('train', 0)), 'ValidationSites': count(splits.get('validation', 0)),
        'TestSites': count(splits.get('test', 0)),
        'FineCount': count(sum(r['outputs']['high_frequency']['rows'] for r in base['records'])),
        'PairCount': count(sum(r['outputs']['hourly_pairs']['rows'] for r in base['records'])),
        'NulByteCount': count(m2['leading_nul']['bytes']), 'NulFileCount': count(m2['leading_nul']['affected_files']),
        'EndpointChecks': count(m2['T_CALC_endpoint']['compared_hours']),
        'ExceptionTotal': count(m2['cross_product_anomalous_variable_hours']),
        'TempExceptions': count(by_variable['air_temperature']), 'SolarExceptions': count(by_variable['solar_radiation']),
        'RainExceptions': count(by_variable['precipitation']), 'HumidityExceptions': count(by_variable['relative_humidity']),
        'EventCount': count(len(events)), 'ConfirmedCount': count(len(confirmed)),
        'TierOne': count(tiers.get(1, 0)), 'TierTwo': count(tiers.get(2, 0)), 'TierThree': count(tiers.get(3, 0)),
        'BufferedCrossYear': count(confirmed.cross_year.sum()),
        'WetCrossYear': count(confirmed.first_source_year.ne(confirmed.last_source_year).sum()),
        'ReferenceEvents': count(reference['reference_events']), 'ReferenceSites': count(reference['reference_stations']),
        'HighCutoff': f'{reference["cutoff_mm_per_15min"]:g}',
        'ReplayEvents': count(len(requested)), 'ReplaySitesUnion': count(audit['replay_sites_union']),
        'ReplaySitesIntersection': count(audit['replay_sites_intersection']),
        'ReplaySitesPerYear': '/'.join(count(audit['replay_sites_per_year'][str(year)]) for year in configuration['replay_years']),
        'ReplayBins': count(pq.ParquetFile(release / 'data/envelope_timeseries.parquet').metadata.num_rows),
        'RetiredStationCount': count(len(retired)), 'HighIntensityEvents': count(len(high)),
        'HighIntensitySites': count(high.station_wban.nunique()),
        'AlabamaHigh': count(high.network.eq('Alabama-USRCRN').sum()),
        'AlabamaHighSites': count(high.loc[high.network.eq('Alabama-USRCRN'), 'station_wban'].nunique()),
        'ObservedPeakChecks': count(audit['observed_peak_checks']),
        'MainEvents': count(len(main)), 'MainSites': count(main.station_wban.nunique()),
        'ProfileMatches': count(matches), 'ProfileAgreement': f'{100 * matches / len(main):.2f}',
        'UnclassifiedSites': count(sites.kg_class.isna().sum()),
    }
    stats.update(validation_macros(release, audit, m2, protocol, confirmed))
    stats.update(source_audit_distributions(release / 'technical_validation',
                 ROOT / configuration['reports_directory'], sites, m2, configuration['replay_years']))
    for prefix, censored in (('Censored', True), ('Uncensored', False)):
        part = events[events.usability_tier.eq(1) & events.antecedent_dry_censored.eq(censored)]
        stats[prefix + 'Events'] = count(len(part))
        stats[prefix + 'Sites'] = count(part.station_wban.nunique())
        stats[prefix + 'Median'] = f'{part.antecedent_dry_hours_lower_bound.median():.2f}'
        quartiles = part.antecedent_dry_hours_lower_bound.quantile([.25, .75])
        stats[prefix + 'IQR'] = f'{quartiles.iloc[1] - quartiles.iloc[0]:.2f}'
    stats['CensoredPct'] = f'{100 * int(stats["CensoredEvents"].replace(",", "")) / tiers[1]:.2f}'
    dry = pd.read_csv(release / 'supplementary/dry_gap_sensitivity_summary.csv').set_index('minimum_dry_hours')
    stats.update({'DrySampleYears': count(dry.station_years.iloc[0]),
                  'DryShortMedian': f'{dry.loc[2, "median_duration_min"]:g}',
                  'DryLongMedian': f'{dry.loc[12, "median_duration_min"]:g}',
                  'DryShortPQuantile': f'{dry.loc[2, "peak_15min_p95_mm"]:g}',
                  'DryLongPQuantile': f'{dry.loc[12, "peak_15min_p95_mm"]:g}'})
    eligibility = pd.read_csv(release / 'audit_log/eligibility_by_station_year.csv')
    for prefix, variable in (('Rain', 'precipitation'), ('Solar', 'solar_radiation')):
        denominator = eligibility.loc[eligibility.variable.eq(variable) & eligibility.track.eq('derived'), 'bounded_target_hours'].sum()
        zero = 0
        for year in sorted({record['year'] for record in base['records']}):
            path = release / 'data/annotations' / f'variable={variable}' / f'source_year={year}' / 'part.parquet'
            column = pq.ParquetFile(path).read(columns=['constraint_class'])['constraint_class']
            zero += pc.sum(pc.cast(pc.equal(column, 'unique_zero_completion'), 'int64')).as_py()
        stats[prefix + 'ZeroPct'] = f'{100 * zero / denominator:.2f}'
    catalogue = pd.read_csv(release / 'technical_validation/catalogue_coverage.csv')
    coverage = pd.read_csv(release / 'technical_validation/replay_coverage.csv')
    assert catalogue.groupby('dimension').events.sum().eq(len(events)).all()
    assert catalogue.groupby('dimension').metadata_sites.sum().eq(len(sites)).all()
    assert coverage.groupby('dimension').replay_events.sum().eq(len(requested)).all()
    stats['CatalogueCoverageRows'] = coverage_rows(catalogue, ['events', 'metadata_sites', 'tier_1_events', 'tier_2_events', 'tier_3_events'])
    stats['ReplayCoverageRows'] = coverage_rows(coverage, ['replay_events', 'replay_sites', 'high_intensity_events', 'high_intensity_sites'])
    diagnostics = pd.read_csv(release / 'technical_validation/peak15_profile_diagnostics.csv')
    titles = ('Front-loaded', 'Mid-loaded', 'Back-loaded', fr'Pooled \(\geq {stats["HighCutoff"]}\) mm')
    lines = []
    for group, title in zip((*PROFILES, 'high_intensity_pooled'), titles):
        cells = diagnostics[diagnostics.group.eq(group)].set_index('method').loc[list(METHODS)]
        values = ['---' if not entry.reported else f'{fixed(entry.median * 100, 1)} ({fixed(entry.iqr * 100, 1)})'
                  for entry in cells.itertuples()]
        lines.append(row(title, count(cells.events.iloc[0]), count(cells.sites.iloc[0]), *values))
    stats['ProfileDiagnosticRows'] = '\n'.join(lines)
    names = dict(zip(METHODS, ['Uniform', 'Constrained linear', 'ExtraTrees']))
    pooled = pd.read_csv(release / 'supplementary/envelope_replay_summary.csv')
    pooled = pooled[pooled.track.eq('official_hourly')]
    lines = []
    for window in (5, 15, 30):
        for i, method in enumerate(METHODS):
            cells = pooled[pooled.window_minutes.eq(window) & pooled.method.eq(method)].set_index('cohort').loc[['all', 'high_intensity']]
            values = [fixed(value) for value in cells[['peak_depth_mae_mm', 'peak_depth_bias_mm']].to_numpy().ravel()]
            lines.append(row(f'{window} min' if i == 0 else '', names[method], *values))
    stats['SupplementaryReplayRows'] = '\n'.join(lines)
    stats['SupplementaryDryRows'] = '\n'.join(row(f'{hours} h', count(entry.observed_segments),
        count(entry.boundary_confirmed_events), f'{entry.median_duration_min:g}', f'{entry.peak_15min_p95_mm:g}') for hours, entry in dry.iterrows())
    hourly = pd.read_csv(release / 'supplementary/supplementary_hourly_errors.csv')
    hourly = hourly[hourly.track.eq('official_hourly') & hourly.stratum.eq('wet_coarse_gt0')].set_index('method')
    stats['SupplementaryHourlyRows'] = '\n'.join(row(names[method], count(hourly.loc[method, 'hourly_peak_n']),
        count(hourly.loc[method, 'within_hour_increment_n']), fixed(hourly.loc[method, 'hourly_peak_mae']),
        fixed(hourly.loc[method, 'within_hour_ramp_mae'])) for method in METHODS)
    temporary = destination.with_suffix('.tex.tmp')
    temporary.write_text('% Generated from validated 2007–2025 full-run data.\n'
        + ''.join(fr'\newcommand{{\{key}}}{{{value}}}' + '\n' for key, value in stats.items()), encoding='utf-8')
    temporary.replace(destination)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'manuscript/station_disaggregation_results.tex')
    args = parser.parse_args()
    render(args.release, args.destination)


if __name__ == '__main__':
    main()
