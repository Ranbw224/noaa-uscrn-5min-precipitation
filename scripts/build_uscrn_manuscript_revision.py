"""Build the Step 2 manuscript figures and tables from the frozen release.

Pass the release directory as the first argument, or run from that directory.
The existing manuscript, its results macros and the release are read-only.
"""

import argparse
import json
import math
import pickletools
import re
import zlib
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.compute as pc
import pyarrow.parquet as pq

from draw_station_disaggregation_workflow import scalar_macros, workflow_figure
from package_station_disaggregation_release import extracted_inventory, package_members, sha256
from station_disaggregation_diagnostics import METHODS, PROFILES, climate, count_tiers, spread


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MACROS = ROOT / 'manuscript/station_disaggregation_results.tex'
DIAGNOSTIC_PROTOCOL = ROOT / 'config/station_disaggregation_release.json'
CLIMATES = ('A', 'B', 'C', 'D', 'E', 'Unclassified')
SEASONS = ('DJF', 'MAM', 'JJA', 'SON')
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
PHASES = ('liquid_likely', 'mixed_or_transitional', 'frozen_likely', 'temp_missing')
DURATIONS = ('lt1h', '1to3h', '3to6h', 'ge6h')
DURATION_LABELS = ('<1 h', '1–<3 h', '3–<6 h', '≥6 h')
METHOD_LABELS = ('Uniform', 'Constrained linear', 'ExtraTrees')
PROFILE_LABELS = ('Front-loaded', 'Mid-loaded', 'Back-loaded')
PHASE_LABELS = ('Liquid likely', 'Mixed or transitional', 'Frozen likely', 'Temperature missing')
METHOD_COLORS = ('#6B7280', '#B27A44', '#2B5B84')
PROFILE_COLORS = ('#4C78A8', '#76B7B2', '#C58B5C')
PHASE_COLORS = ('#4C78A8', '#C58B5C', '#76B7B2', '#C9CED5')
INK, MUTED, RULE = '#1D2429', '#4C5662', '#D5DCE3'

EVENT_FIELDS = [
    'event_id', 'station_wban', 'dataset_split', 'first_source_year', 'network', 'kg_group',
    'event_start_utc', 'event_end_utc', 'event_duration_min', 'covering_hours',
    'boundary_confirmed', 'usability_tier', 'peak_15min_mm', 'precip_phase_flag',
    'storm_profile_tercile', 'earliest_peak_ratio', 'temporal_centroid_ratio', 'event_wetbulb_mean_c', 'soil_moisture_pre_5cm',
    'soil_moisture_missing_reason_5cm', 'antecedent_dry_hours_lower_bound',
    'antecedent_dry_censored', 'p95_reference_event', 'high_intensity',
]
COUNT_FIELDS = ('events', 'sites')
REPLAY_COUNTS = ('replay_events', 'replay_sites', 'high_intensity_events', 'high_intensity_sites')
CATALOGUE_COUNTS = ('events', 'metadata_sites', 'tier_1_events', 'tier_2_events', 'tier_3_events')
STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'axes.linewidth': .55,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': .55, 'ytick.major.width': .55, 'legend.frameon': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'pdf.compression': 9,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def numeric_table_matches(actual, recorded, keys, columns, tolerance=0.0):
    a = actual.set_index(keys)[list(columns)].sort_index()
    b = recorded.set_index(keys)[list(columns)].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=tolerance == 0,
                                  atol=tolerance, rtol=0.0)


def counts(frame):
    return {'events': len(frame), 'sites': int(frame.station_wban.nunique())}


def fmt_count(value):
    return f'{int(value):,}'


def fmt_value(value, digits):
    if pd.isna(value):
        return '---'
    value = f'{value:.{digits}f}'
    return fr'\({value}\)' if value.startswith('-') else value


def latex_rows(rows):
    return '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)


def check_scalar_macros(macros, calculated):
    for key, value in calculated.items():
        require(macros[key] == value, f'Baseline macro {key} differs: {macros[key]} vs {value}')


def check_coverage_macro(macros, key, frame, fields):
    displayed = {'<500 m': r'\(<500\) m', '500–1500 m': '500--1,500 m',
                 '>1500 m': r'\(>1500\) m'}
    rows = []
    for dimension, group in frame.groupby('dimension', sort=False):
        for i, row in enumerate(group.itertuples(index=False)):
            rows.append((dimension.capitalize() if i == 0 else '', displayed.get(row.group, row.group),
                         *(fmt_count(getattr(row, field)) for field in fields)))
    require(latex_rows(rows) == macros[key], f'Baseline {key} differs')


def check_frozen_inputs(release):
    audit = read_json(release / 'audit_log/validation_summary.json')
    index_path = release / 'data/portable_index.json'
    index = read_json(index_path)
    require(audit['status'] == 'passed' and audit['assertions'] == dict.fromkeys('123456', 'passed'),
            'The release does not have all six passed assertions')
    require(sha256(index_path) == audit['portable_index_sha256'], 'Portable index SHA-256 differs')
    expected_paths = ('data/events.parquet', 'data/station_metadata.csv',
                      'data/envelope_timeseries.parquet', 'data/envelope_event_scores.csv')
    entries = {item['path']: item for record in index['records'] for item in record['outputs'].values()
               if item['path'] in expected_paths}
    for name in expected_paths:
        path, expected = release / name, entries[name]
        require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'],
                f'Frozen data file differs from the portable index: {name}')
    return audit, entries


def load_catalogue(release, macros, audit):
    events = pd.read_parquet(release / 'data/events.parquet', columns=EVENT_FIELDS)
    sites = pd.read_csv(release / 'data/station_metadata.csv', dtype={'station_wban': str})
    require(events.event_id.is_unique and events.event_id.notna().all(), 'Invalid event_id key')
    require(sites.station_wban.is_unique and sites.station_wban.str.fullmatch(r'\d{5}').all(),
            'Invalid station metadata WBAN key')
    elevation = sites.elevation_m.to_numpy()
    sites = sites.assign(climate=climate(sites.kg_group), elevation=np.select(
        [~np.isfinite(elevation), elevation < 500, elevation <= 1500],
        ['Unclassified', '<500 m', '500–1500 m'], default='>1500 m'))
    events = events.merge(sites[['station_wban', 'climate', 'elevation', 'network', 'dataset_split']],
                          on='station_wban', how='left', validate='many_to_one', suffixes=('', '_metadata'))
    require(events.climate.notna().all() and climate(events.kg_group).eq(events.climate).all()
            and events.network.eq(events.network_metadata).all()
            and events.dataset_split.eq(events.dataset_split_metadata).all(),
            'Event station, climate, network or split assignments differ from station metadata')
    tiers = count_tiers(events)
    require(sum(tiers.values()) == len(events), 'The three usability tiers do not cover the catalogue')
    confirmed = events.loc[events.usability_tier.isin((1, 2))].copy()
    require(events.boundary_confirmed.eq(events.usability_tier.isin((1, 2))).all(),
            'Confirmed boundaries differ from the Tier 1/2 mapping')
    reference = events.loc[events.p95_reference_event.eq(True)]
    high = confirmed.loc[confirmed.high_intensity.eq(True)]
    check_scalar_macros(macros, {
        'EventCount': fmt_count(len(events)), 'ConfirmedCount': fmt_count(len(confirmed)),
        'StationCount': fmt_count(len(sites)), 'TierOne': fmt_count(tiers['tier_1_events']),
        'TierTwo': fmt_count(tiers['tier_2_events']), 'TierThree': fmt_count(tiers['tier_3_events']),
        'CatalogueHighCount': fmt_count(len(high)), 'ReferenceEvents': fmt_count(len(reference)),
        'ReferenceSites': fmt_count(reference.station_wban.nunique()),
        'HighCutoff': str(audit['reference']['cutoff_mm_per_15min']),
    })
    return events, sites, confirmed, tiers


def coverage_checks(release, macros, events, sites, replay):
    catalogue, composition = [], []
    for dimension, labels in (
        ('climate', CLIMATES), ('elevation', ('<500 m', '500–1500 m', '>1500 m', 'Unclassified')),
        ('network', sorted(sites.network.unique())),
    ):
        for label in labels:
            part = events.loc[events[dimension].eq(label)]
            catalogue.append({'dimension': dimension, 'group': label, 'events': len(part),
                              'metadata_sites': int(sites[dimension].eq(label).sum()),
                              'sites_with_events': part.station_wban.nunique(), **count_tiers(part)})
            if dimension != 'elevation':
                r = replay.loc[replay[dimension].eq(label)]
                high = r.loc[r.high_intensity.eq(True)]
                composition.append(dict(zip(('dimension', 'group', *REPLAY_COUNTS),
                                            (dimension, label, len(r), r.station_wban.nunique(),
                                             len(high), high.station_wban.nunique()))))
    catalogue, composition = pd.DataFrame(catalogue), pd.DataFrame(composition)
    for actual, name, macro, fields in (
        (catalogue, 'catalogue_coverage.csv', 'CatalogueCoverageRows', CATALOGUE_COUNTS),
        (composition, 'replay_coverage.csv', 'ReplayCoverageRows', REPLAY_COUNTS),
    ):
        recorded = pd.read_csv(release / 'technical_validation' / name)
        required = (*fields, 'sites_with_events') if macro == 'CatalogueCoverageRows' else fields
        numeric_table_matches(actual, recorded, ['dimension', 'group'], required)
        check_coverage_macro(macros, macro, actual, fields)
    return catalogue, composition


def interval_peaks(release, replay, protocol, audit):
    series = pd.read_parquet(release / 'data/envelope_timeseries.parquet', columns=[
        'event_id', 'station_wban', 'time_end_utc', 'observed', *METHODS])
    series = series.sort_values(['event_id', 'time_end_utc'], ignore_index=True)
    require(len(series) > 0 and not series.duplicated(['event_id', 'time_end_utc']).any(),
            'Frozen envelope has a duplicate or empty time-step key')
    roster = replay.set_index('event_id')
    by_event = series.groupby('event_id', sort=False)
    lengths = by_event.size().reindex(roster.index)
    step = protocol['interval_minutes']
    require(by_event.ngroups == len(roster) and lengths.eq(roster.covering_hours * (60 // step)).all()
            and series.station_wban.eq(series.event_id.map(roster.station_wban)).all()
            and by_event.time_end_utc.diff().dropna().eq(pd.Timedelta(minutes=step)).all(),
            'Envelope station, event, interval continuity or 12K support differs from the frozen roster')
    require(np.isfinite(series[['observed', *METHODS]].to_numpy()).all(), 'Non-finite frozen envelope depth')
    # The only time-step calculation here is the rolling sum used to verify the 15-minute peaks.
    peaks = (by_event[['observed', *METHODS]].rolling(15 // step).sum()
             .groupby(level=0, sort=False).max())
    names = ['event_id', 'station_wban', 'track', 'window_minutes', 'method', 'observed_peak_mm',
             'predicted_peak_mm', 'abs_peak_depth_error_mm', 'signed_peak_depth_error_mm']
    scores = pd.read_csv(release / 'data/envelope_event_scores.csv', usecols=names,
                         dtype={'station_wban': str})
    require(len(scores) == audit['score_rows'], 'Score file row count differs from the release audit')
    scores = scores.loc[scores.track.eq(protocol['envelope_replay']['track'])
                        & scores.window_minutes.eq(15)].copy()
    require(len(scores) == len(replay) * len(METHODS) and set(scores.method) == set(METHODS)
            and not scores.duplicated(['event_id', 'method']).any(), 'Incomplete 15-minute score matrix')
    for method in METHODS:
        ids = scores.loc[scores.method.eq(method), 'event_id']
        require(set(ids) == set(roster.index), f'{method} score event set differs from the frozen roster')
    scores = scores.merge(replay, on=['event_id', 'station_wban'], how='left', validate='many_to_one')
    require(scores.source_year.notna().all(), '15-minute score station/event identities differ from the roster')
    derived = peaks.rename(columns={'observed': 'envelope_observed_peak_mm'}).reset_index().melt(
        id_vars=['event_id', 'envelope_observed_peak_mm'], var_name='method', value_name='envelope_predicted_peak_mm')
    scores = scores.merge(derived, on=['event_id', 'method'], how='left', validate='one_to_one')
    maximum = {}
    for key, left, right in (
        ('observed_envelope_to_frozen_score', 'envelope_observed_peak_mm', 'observed_peak_mm'),
        ('predicted_envelope_to_frozen_score', 'envelope_predicted_peak_mm', 'predicted_peak_mm'),
        ('observed_catalogue_to_frozen_score', 'peak_15min_mm', 'observed_peak_mm'),
    ):
        absolute = scores[left].sub(scores[right]).abs()
        require(absolute.le(protocol['peak_depth_tolerance_mm']).all(), f'{key} exceeds peak tolerance')
        maximum[key] = float(absolute.max())
    delta = scores.predicted_peak_mm - scores.observed_peak_mm
    require(np.allclose(delta, scores.signed_peak_depth_error_mm, atol=1e-12, rtol=0)
            and np.allclose(delta.abs(), scores.abs_peak_depth_error_mm, atol=1e-12, rtol=0),
            'Frozen absolute/signed peak-depth scores do not match the verified peaks')
    return scores, len(series), maximum


def load_replay(release, events, protocol, diagnostic, audit, macros):
    roster = pd.read_csv(release / 'audit_log/replay_cohort.csv', dtype={'station_wban': str})
    for column in ('event_start_utc', 'event_end_utc'):
        roster[column] = pd.to_datetime(roster[column], utc=True)
    replay = roster.merge(events, on=['event_id', 'station_wban'], how='left', validate='one_to_one',
                          suffixes=('_roster', ''))
    years = protocol['envelope_replay']['evaluation_years']
    require(replay.usability_tier.eq(1).all()
            and replay.dataset_split.eq(protocol['envelope_replay']['station_split']).all()
            and sorted(replay.source_year.unique()) == years
            and replay.source_year.eq(replay.first_source_year).all(), 'Frozen roster split, years or tiers differ')
    for key in ('event_start_utc', 'event_end_utc', 'covering_hours'):
        require(replay[key].eq(replay[f'{key}_roster']).all(), f'Frozen roster {key} differs from events')
    scores, bins, maximum = interval_peaks(release, replay, protocol, audit)
    cutoff = diagnostic['diagnostic_min_observed_15min_mm']
    tolerance = protocol['peak_depth_tolerance_mm']
    main = scores.loc[scores.observed_peak_mm.ge(cutoff - tolerance)].copy()
    main['underestimation_rate'] = (main.observed_peak_mm - main.predicted_peak_mm) / main.observed_peak_mm
    primary = replay.loc[replay.peak_15min_mm.ge(cutoff - tolerance)]
    require(set(main.event_id) == set(primary.event_id)
            and all(set(group.event_id) == set(primary.event_id) for _, group in main.groupby('method')),
            'The three methods and the catalogue do not share the primary event denominator')
    high = replay.loc[replay.high_intensity.eq(True)]
    by_year = replay.groupby('source_year').station_wban.nunique()
    require(by_year.to_dict() == {int(year): n for year, n in audit['replay_sites_per_year'].items()}
            and len(replay) == audit['replay_events'] and replay.station_wban.nunique() == audit['replay_sites_union'],
            'The replay denominator differs from the frozen audit')
    check_scalar_macros(macros, {
        'ReplayEvents': fmt_count(len(replay)), 'ReplayBins': fmt_count(bins),
        'ReplaySitesUnion': fmt_count(replay.station_wban.nunique()),
        'ReplaySitesPerYear': '/'.join(map(str, by_year)),
        'MainEvents': fmt_count(len(primary)), 'MainSites': fmt_count(primary.station_wban.nunique()),
        'HighIntensityEvents': fmt_count(len(high)), 'HighIntensitySites': fmt_count(high.station_wban.nunique()),
    })
    return replay, scores, main, primary, bins, maximum


def check_profile_scores(release, main, scores, diagnostic, macros):
    min_events, min_sites = diagnostic['minimum_events'], diagnostic['minimum_sites']
    checks = []
    for method in METHODS:
        subset = main.loc[main.method.eq(method)]
        for profile in (*PROFILES, 'high_intensity_pooled'):
            group = subset[subset.high_intensity.eq(True) if profile == 'high_intensity_pooled'
                           else subset.storm_profile_tercile.eq(profile)]
            size = counts(group)
            report = size['events'] >= min_events and size['sites'] >= min_sites
            checks.append({'group': profile, 'method': method, **size, 'reported': report,
                           **(spread(group.underestimation_rate) if report else {})})
    check_frame = pd.DataFrame(checks)
    recorded = pd.read_csv(release / 'technical_validation/peak15_profile_diagnostics.csv')
    fields = (*COUNT_FIELDS, 'reported', 'median', 'q1', 'q3', 'iqr')
    numeric_table_matches(check_frame, recorded, ['group', 'method'], fields, tolerance=1e-12)
    rows = []
    labels = (*PROFILE_LABELS, fr'Pooled \(\geq {macros["HighCutoff"]}\) mm')
    for profile, label in zip((*PROFILES, 'high_intensity_pooled'), labels):
        group = check_frame[check_frame.group.eq(profile)].set_index('method').loc[list(METHODS)]
        row = group.iloc[0]
        rows.append((label, fmt_count(row.events), fmt_count(row.sites),
                     *(f'{cell.median * 100:.1f} ({cell.iqr * 100:.1f})' if cell.reported else '---'
                       for cell in group.itertuples())))
    require(latex_rows(rows) == macros['ProfileDiagnosticRows'],
            'Recalculated Table 6 does not match the original formatted macro')
    pooled = []
    for method, group in scores.groupby('method'):
        for name, frame in (('all', group), ('high_intensity', group.loc[group.high_intensity.eq(True)])):
            pooled.append({'method': method, 'cohort': name, 'events': len(frame),
                           'stations': frame.station_wban.nunique(),
                           'peak_depth_mae_mm': frame.abs_peak_depth_error_mm.mean(),
                           'peak_depth_bias_mm': frame.signed_peak_depth_error_mm.mean()})
    pooled = pd.DataFrame(pooled)
    replay_reference = pd.read_csv(release / 'supplementary/envelope_replay_summary.csv')
    numeric_table_matches(pooled, replay_reference.loc[
        replay_reference.window_minutes.eq(15) & replay_reference.track.eq('official_hourly')],
        ['method', 'cohort'], ['events', 'stations', 'peak_depth_mae_mm', 'peak_depth_bias_mm'], tolerance=1e-12)
    rows = []
    for i, (method, label) in enumerate(zip(METHODS, METHOD_LABELS)):
        cells = pooled.loc[pooled.method.eq(method)].set_index('cohort').loc[['all', 'high_intensity']]
        values = cells[['peak_depth_mae_mm', 'peak_depth_bias_mm']].to_numpy().ravel()
        rows.append(('15 min' if i == 0 else '', label, *(fmt_value(value, 3) for value in values)))
    old = macros['SupplementaryReplayRows'].splitlines()
    start = next(i for i, row in enumerate(old) if row.startswith('15 min &'))
    require(latex_rows(rows) == '\n'.join(old[start:start + len(rows)]),
            '15-minute Supplementary Table S1 does not match the original formatted macro')
    return checks, len(pooled)


def proportions(data, strata, levels, category, labels):
    rows = []
    for level in levels:
        group = data.loc[data[strata].eq(level)]
        require(len(group) > 0, f'Empty denominator for {strata}={level}')
        labelled = group[category].isin(labels)
        require(labelled.all(), f'Unknown {category} within {strata}={level}')
        for label in labels:
            subset = group.loc[group[category].eq(label)]
            rows.append({strata: level, category: label, **counts(subset), 'denominator_events': len(group),
                         'denominator_sites': group.station_wban.nunique(), 'share_pct': 100 * len(subset) / len(group)})
    table = pd.DataFrame(rows)
    grouped = table.groupby(strata, sort=False)
    require(table.events.sum() == len(data) and grouped.events.sum().eq(grouped.denominator_events.first()).all()
            and np.allclose(grouped.share_pct.sum(), 100.0, atol=1e-12, rtol=0), f'{strata} fractions do not reconcile')
    return table


def correlation(group):
    dry, soil = group.antecedent_dry_hours_lower_bound, group.soil_moisture_pre_5cm
    values = {'pearson_r': np.nan, 'spearman_rho': np.nan}
    if dry.nunique() > 1 and soil.nunique() > 1:
        values = {'pearson_r': dry.corr(soil), 'spearman_rho': dry.corr(soil, method='spearman')}
    return values


def overview_tables(release, confirmed, macros, catalogue):
    grouped, distributions = [], []
    for name in CLIMATES:
        part = confirmed.loc[confirmed.climate.eq(name)]
        reference = catalogue.loc[catalogue.dimension.eq('climate') & catalogue.group.eq(name)].iloc[0]
        require(len(part) == reference.tier_1_events + reference.tier_2_events,
                f'Data Overview climate count differs from Table 4: {name}')
        row = {'climate': name, **counts(part), 'metadata_sites': int(reference.metadata_sites)}
        for key, values in (('peak_15min_mm', part.peak_15min_mm), ('duration_min', part.event_duration_min)):
            require(len(values) > 0 and np.isfinite(values).all() and values.gt(0).all(),
                    f'Unusable observed peaks or durations in {name}')
            for label, value in zip(('q05', 'q25', 'median', 'q75', 'q95'),
                                    np.quantile(values, [.05, .25, .5, .75, .95], method='linear')):
                row[f'{key}_{label}'] = value
        distributions.append(row)
    month = confirmed.event_start_utc.dt.month
    confirmed = confirmed.assign(month=month, season=month.map({m: SEASONS[(m % 12) // 3] for m in range(1, 13)}),
                                 duration=pd.cut(confirmed.event_duration_min, [0, 60, 180, 360, np.inf],
                                                 labels=DURATIONS, right=False))
    phases = proportions(confirmed, 'month', range(1, 13), 'precip_phase_flag', PHASES)
    profiles = proportions(confirmed, 'duration', DURATIONS, 'storm_profile_tercile', PROFILES)
    source_phase = pd.read_csv(release / 'technical_validation/phase_coverage.csv')
    observed_phases = phases.groupby('precip_phase_flag').events.sum()
    archived_phases = source_phase.groupby('precip_phase_flag').events.sum()
    require(observed_phases.to_dict() == archived_phases.to_dict(), 'Monthly phase counts differ from the release')
    for flag, name in zip(PHASES, ('PhaseLiquid', 'PhaseMixed', 'PhaseFrozen', 'PhaseTempMissing')):
        check_scalar_macros(macros, {name: fmt_count(observed_phases[flag])})
    paired = confirmed.loc[confirmed.antecedent_dry_censored.eq(False)
                           & confirmed.soil_moisture_missing_reason_5cm.eq('valid')]
    require(np.isfinite(paired[['antecedent_dry_hours_lower_bound', 'soil_moisture_pre_5cm']].to_numpy()).all(),
            'The uncensored confirmed soil-moisture subset contains non-finite pairs')
    pooled = correlation(paired)
    check_scalar_macros(macros, {'SoilDryEvents': fmt_count(len(paired)),
                                 'SoilDrySites': fmt_count(paired.station_wban.nunique()),
                                 'SoilDryPearson': fmt_value(pooled['pearson_r'], 3),
                                 'SoilDrySpearman': fmt_value(pooled['spearman_rho'], 3)})
    period_checks = []
    recorded_periods = pd.read_csv(release / 'technical_validation/soil_moisture_antecedent_dry.csv')
    for period in recorded_periods.period:
        first, last = map(int, period.split('-'))
        part = confirmed.loc[confirmed.first_source_year.between(first, last)]
        samples = paired.loc[paired.first_source_year.between(first, last)]
        period_checks.append({'period': period, 'confirmed_events': len(part),
                              'wetbulb_available_events': int(part.event_wetbulb_mean_c.count()),
                              'soil_5cm_available_events': int(part.soil_moisture_pre_5cm.count()),
                              'uncensored_with_soil_5cm_events': len(samples),
                              'paired_sites': int(samples.station_wban.nunique()), **correlation(samples)})
    numeric_table_matches(pd.DataFrame(period_checks), recorded_periods, ['period'],
                          recorded_periods.columns.drop('period'), tolerance=1e-12)
    coverage_rows = []
    for row in period_checks:
        n = row['confirmed_events']
        coverage_rows.append((row['period'].replace('-', '--'), fmt_count(n),
                              fmt_value(100 * row['wetbulb_available_events'] / n, 1),
                              fmt_value(100 * row['soil_5cm_available_events'] / n, 1),
                              fmt_count(row['uncensored_with_soil_5cm_events']), fmt_count(row['paired_sites']),
                              fmt_value(row['pearson_r'], 3), fmt_value(row['spearman_rho'], 3)))
    require(latex_rows(coverage_rows) == macros['SoilDryCoverageRows'],
            'Supplementary Table S4 does not match the original formatted macro')
    for name in CLIMATES:
        for season in SEASONS:
            part = confirmed.loc[confirmed.climate.eq(name) & confirmed.season.eq(season)]
            samples = paired.loc[paired.climate.eq(name) & paired.season.eq(season)]
            grouped.append({'climate': name, 'season': season, 'confirmed_events': len(part),
                            'confirmed_sites': part.station_wban.nunique(), 'paired_events': len(samples),
                            'paired_sites': samples.station_wban.nunique(), **correlation(samples)})
    soil = pd.DataFrame(grouped)
    require(soil.confirmed_events.sum() == len(confirmed) and soil.paired_events.sum() == len(paired),
            'Climate-by-season partitions do not recover the confirmed/paired event denominator')
    distributions = pd.DataFrame(distributions)
    require(distributions.events.sum() == len(confirmed)
            and distributions.sites.sum() == confirmed.station_wban.nunique(),
            'Climate groups do not recover the confirmed event and station denominators')
    details = {'soil_dry_pooled': {**counts(paired), **pooled}, 'soil_dry_s4_periods': period_checks,
               'climate_support': distributions[['climate', 'events', 'sites', 'metadata_sites']].to_dict('records'),
               'phase_totals': {label: int(n) for label, n in observed_phases.items()},
               'phase_month_denominators': phases.groupby('month').denominator_events.first().to_dict(),
               'profile_duration_denominators': profiles.groupby('duration', sort=False).denominator_events.first().to_dict(),
               'climate_season_cells': len(soil), 'climate_season_paired_events_sum': int(soil.paired_events.sum())}
    return distributions, phases, profiles, soil, details


def replay_strata(scores, main, primary, replay, years, diagnostic):
    tables = []
    min_events, min_sites = diagnostic['minimum_events'], diagnostic['minimum_sites']
    for dimension, levels in (('pooled', ('All',)), ('source_year', years), ('climate', CLIMATES)):
        for level in levels:
            part = scores if dimension == 'pooled' else scores.loc[scores[dimension].eq(level)]
            selected = main if dimension == 'pooled' else main.loc[main[dimension].eq(level)]
            for method in METHODS:
                all_method, primary_method = part.loc[part.method.eq(method)], selected.loc[selected.method.eq(method)]
                n, s = len(primary_method), primary_method.station_wban.nunique()
                all_n, all_s = len(all_method), all_method.station_wban.nunique()
                report, report_all = n >= min_events and s >= min_sites, all_n >= min_events and all_s >= min_sites
                result = spread(primary_method.underestimation_rate) if report else dict.fromkeys(('median', 'q1', 'q3', 'iqr'), np.nan)
                tables.append({'dimension': dimension, 'level': level, 'method': method,
                               'primary_events': n, 'primary_sites': s, 'report_underestimation': report,
                               'median_underestimation_pct': 100 * result['median'], 'q1_underestimation_pct': 100 * result['q1'],
                               'q3_underestimation_pct': 100 * result['q3'], 'iqr_underestimation_pp': 100 * result['iqr'],
                               'all_replay_events': all_n, 'all_replay_sites': all_s, 'report_all_peak_mae': report_all,
                               'all_15min_peak_depth_mae_mm': all_method.abs_peak_depth_error_mm.mean() if report_all else np.nan})
    table = pd.DataFrame(tables)
    for dimension in ('source_year', 'climate'):
        subset = table.loc[table.dimension.eq(dimension)]
        sums = subset.groupby('method')[['primary_events', 'all_replay_events']].sum()
        require(sums.primary_events.eq(len(primary)).all() and sums.all_replay_events.eq(len(replay)).all(),
                f'Replay denominators do not reconcile across {dimension}')
        for level, group in subset.groupby('level'):
            expect = (len(primary.loc[primary[dimension].eq(level)]), primary.loc[primary[dimension].eq(level), 'station_wban'].nunique(),
                      len(replay.loc[replay[dimension].eq(level)]), replay.loc[replay[dimension].eq(level), 'station_wban'].nunique())
            actual = group[['primary_events', 'primary_sites', 'all_replay_events', 'all_replay_sites']]
            require(np.array_equal(actual.to_numpy(), np.tile(expect, (len(METHODS), 1))),
                    f'Methods disagree on event or site support: {dimension} {level}')
    reported = table.loc[table.dimension.eq('climate') & table.report_underestimation, 'level']
    require(set(reported) == {'B', 'C', 'D'}, 'Climate error support differs from the planned B/C/D coverage')
    return table


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, size=10, weight='bold', va='center')
    fig.text(x + .028, y, title, size=9, weight='bold', va='center')


def draw_boxplots(ax, table, column, title):
    headings = ('q05', 'q25', 'median', 'q75', 'q95')
    boxes = [dict(zip(('whislo', 'q1', 'med', 'q3', 'whishi'), values))
             for values in table[[f'{column}_{q}' for q in headings]].to_numpy()]
    ax.bxp(boxes, showfliers=False, patch_artist=True, widths=.48,
           boxprops={'facecolor': '#D7E5EE', 'edgecolor': '#4C78A8', 'linewidth': .8},
           medianprops={'color': '#2B5B84', 'linewidth': 1.15},
           whiskerprops={'color': '#4C78A8', 'linewidth': .65},
           capprops={'color': '#4C78A8', 'linewidth': .65})
    ax.set_xticks(range(1, len(table) + 1), [*CLIMATES[:-1], 'Uncl.'])
    ax.set_yscale('log')
    ticks = [value for value in (.1, 1, 10, 100, 1000) if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.tick_params(axis='y', which='minor', length=0, labelleft=False)
    ax.set_title(title, loc='left', size=8.3, pad=6)


def overview_figure(distributions, phases, profiles, soil):
    fig = plt.figure(figsize=(7.15, 8.06))
    panel(fig, .048, .964, 'a', 'Observed peak depths and durations by climate group')
    draw_boxplots(fig.add_axes([.102, .744, .375, .165]), distributions, 'peak_15min_mm', '15-min peak depth (mm)')
    draw_boxplots(fig.add_axes([.588, .744, .370, .165]), distributions, 'duration_min', 'Event duration (min)')
    fig.text(.103, .701, 'Boxes show the median and IQR. Whiskers span the 5th–95th percentiles. Uncl. = unclassified.',
             size=7.2, color=MUTED)

    panel(fig, .048, .660, 'b', 'Phase screening by UTC start month')
    ax = fig.add_axes([.103, .482, .375, .151])
    month = phases.pivot(index='month', columns='precip_phase_flag', values='share_pct')
    bottom = np.zeros(len(month))
    for flag, label, color in zip(PHASES, PHASE_LABELS, PHASE_COLORS):
        ax.bar(month.index, month[flag], bottom=bottom, width=.76, color=color, label=label)
        bottom += month[flag].to_numpy()
    ax.set_xticks(month.index, [str(m) for m in month.index])
    ax.set_yticks([0, 50, 100], ['0%', '50%', '100%'])
    ax.set_ylim(0, 100)
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(-.13, -.20), fontsize=7.1,
              handlelength=1, handletextpad=.45, columnspacing=.9, borderaxespad=0)

    panel(fig, .538, .660, 'c', 'Storm profiles by event duration')
    ax = fig.add_axes([.646, .482, .312, .151])
    classes = profiles.pivot(index='duration', columns='storm_profile_tercile', values='share_pct').loc[list(DURATIONS)]
    left = np.zeros(len(classes))
    for profile, label, color in zip(PROFILES, PROFILE_LABELS, PROFILE_COLORS):
        ax.barh(range(len(classes)), classes[profile], left=left, height=.61, color=color, label=label)
        left += classes[profile].to_numpy()
    ax.set_yticks(range(len(classes)), DURATION_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(-.34, -.20), fontsize=7.1,
              handlelength=1, handletextpad=.45, columnspacing=.8, borderaxespad=0)

    panel(fig, .048, .394, 'd', 'Pre-event 5 cm soil moisture and uncensored antecedent dry duration')
    fig.text(.076, .365, 'Pearson r / Spearman ρ in bold. Paired events / WBANs below each estimate.',
             size=7.3, color=MUTED)
    ax = fig.add_axes([.066, .039, .896, .300], xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    columns, width = (.164, .374, .584, .794), .200
    for x, season in zip(columns, SEASONS):
        ax.text(x + width / 2, .965, season, weight='bold', ha='center', va='center')
    indexed = soil.set_index(['climate', 'season'])
    for i, name in enumerate(CLIMATES):
        y = .855 - i * .143
        ax.text(.015, y - .051, 'Unclassified' if name == 'Unclassified' else name, ha='left', va='center')
        for x, season in zip(columns, SEASONS):
            cell = indexed.loc[(name, season)]
            ok = pd.notna(cell.pearson_r) and pd.notna(cell.spearman_rho)
            value = (f'{cell.pearson_r:.2f} / {cell.spearman_rho:.2f}'.replace('-', '−') if ok else '—')
            color = '#EAF1F7' if ok else '#F5F6F7'
            ax.add_patch(Rectangle((x, y - .118), width, .132, facecolor=color, edgecolor='none'))
            ax.text(x + width / 2, y - .027, value, weight='bold', size=7.7, ha='center', va='center')
            ax.text(x + width / 2, y - .082, f'{fmt_count(cell.paired_events)} / {fmt_count(cell.paired_sites)}',
                    color=MUTED, size=7.0, ha='center', va='center')
    return fig


def replay_figure(table, years, diagnostic):
    fig = plt.figure(figsize=(7.15, 3.66))
    panel(fig, .049, .810, 'a', 'Frozen source year')
    panel(fig, .540, .810, 'b', 'Köppen–Geiger climate group')
    axes = [fig.add_axes([.105, .298, .38, .447]), fig.add_axes([.574, .298, .38, .447])]
    lower = min(0, 10 * np.floor(table.q1_underestimation_pct.min() / 10))
    for ax, dimension, levels in zip(axes, ('source_year', 'climate'), (years, CLIMATES[:-1])):
        frame = table.loc[table.dimension.eq(dimension)]
        ticks = np.arange(len(levels))
        support = frame.loc[frame.method.eq(METHODS[0])].set_index('level').loc[list(levels)]
        for method, label, color, marker, offset in zip(METHODS, METHOD_LABELS, METHOD_COLORS,
                                                         ('o', 's', 'D'), (-.22, 0, .22)):
            values = frame.loc[frame.method.eq(method)].set_index('level').loc[list(levels)]
            use = values.report_underestimation.to_numpy()
            measured = values.loc[use]
            median = measured.median_underestimation_pct
            ax.errorbar(ticks[use] + offset, median, marker=marker, linestyle='none', color=color, capsize=2.1,
                        yerr=[median - measured.q1_underestimation_pct, measured.q3_underestimation_pct - median],
                        label=label, markersize=4.1, linewidth=.95)
        ax.set_xticks(ticks, [f'{key}\n{n:,} / {s}' for key, n, s in zip(levels, support.primary_events, support.primary_sites)])
        ax.tick_params(axis='x', labelsize=7.5, pad=5)
        ax.set_xlim(-.48, len(levels) - .52)
        ax.set_ylim(lower, 100)
        ax.set_yticks(np.arange(lower, 101, 25) if lower < 0 else [0, 25, 50, 75, 100])
        ax.tick_params(axis='y', labelsize=7.7)
        ax.grid(axis='y', color='#EBEEF1', linewidth=.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('15-min peak underestimation (%)', labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='center', bbox_to_anchor=(.5, .926),
               columnspacing=2, handletextpad=.45)
    min_events, min_sites = diagnostic['minimum_events'], diagnostic['minimum_sites']
    fig.text(.105, .124, 'Points show medians. Bars span the interquartile range. Counts are events / WBANs.',
             color=MUTED, size=7.3)
    fig.text(.105, .066, f'Climate groups A and E are shown as counts. Reporting requires {min_events} events and {min_sites} WBANs.',
             color=MUTED, size=7.3)
    return fig


def visible_text(fig):
    texts = list(fig.texts)
    legends = list(fig.legends)
    for ax in fig.axes:
        texts.extend(ax.texts)
        if not ax.axison:
            continue
        texts.extend([ax.title, ax.get_xaxis().label, ax.get_yaxis().label])
        # pyplot stores left/right-aligned titles separately from the centred title.
        texts.extend(t for t in ax.get_children() if isinstance(t, matplotlib.text.Text) and t not in texts)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo <= tick.get_loc() <= hi:
                    texts.extend([tick.label1, tick.label2])
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for legend in legends:
        texts.extend(legend.get_texts())
    return list(dict.fromkeys(t for t in texts if t.get_visible() and t.get_text().strip()))


def export_figure(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [(text.get_text().replace('\n', ' / '), text.get_window_extent(renderer)) for text in visible_text(fig)]
    width, height = fig.canvas.get_width_height()
    for i, (label, rect) in enumerate(extents):
        require(rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= width and rect.y1 <= height,
                f'{path.name} clipped text: {label}')
        for previous, bbox in extents[:i]:
            dx = min(rect.x1, bbox.x1) - max(rect.x0, bbox.x0)
            dy = min(rect.y1, bbox.y1) - max(rect.y0, bbox.y0)
            require(min(dx, dy) < 1.5, f'{path.name} overlapping text: {label} / {previous}')
    temporary = path.with_suffix('.pdf.tmp')
    fig.savefig(temporary, format='pdf', metadata={'CreationDate': None, 'ModDate': None})
    temporary.replace(path)
    plt.close(fig)
    return {'file': path.name, 'text_boxes_checked': len(extents), 'clipped': 0, 'overlapping': 0,
            'sha256': sha256(path)}


def manuscript_statistics(distributions, phases, profiles, soil, replay, confirmed, protocol, packages):
    peaks = distributions.peak_15min_mm_median
    durations = distributions.duration_min_median
    phase_share = phases.set_index(['month', 'precip_phase_flag']).share_pct
    profile_share = profiles.set_index(['duration', 'storm_profile_tercile']).share_pct
    short = confirmed.loc[confirmed.event_duration_min.lt(60)]
    single = short.loc[short.event_duration_min.eq(protocol['interval_minutes'])]
    require(single.temporal_centroid_ratio.eq(.5).all() and single.storm_profile_tercile.eq('mid_loaded').all(),
            'Single-interval events must have midpoint centroids and mid-loaded labels')
    supported = soil.loc[soil.climate.isin(('B', 'C', 'D')), 'pearson_r']
    a_sites = soil.loc[soil.climate.eq('A'), 'paired_sites']
    require(a_sites.nunique() == 1 and supported.notna().all(), 'Seasonal sample support differs')
    year = replay.loc[replay.dimension.eq('source_year')]
    climate_rows = replay.loc[replay.dimension.eq('climate')]
    annual = year.pivot(index='level', columns='method', values='median_underestimation_pct')[list(METHODS)]
    climate_medians = (climate_rows.loc[climate_rows.report_underestimation]
                       .pivot(index='level', columns='method', values='median_underestimation_pct')[list(METHODS)])
    for levels in (annual, climate_medians):
        require(levels.notna().all(axis=None) and (np.diff(levels.to_numpy(), axis=1) < 0).all(),
                'The stated ordering must hold in every reportable year and climate group')
    ranges = annual.max() - annual.min()
    year_sites = year.groupby('level', sort=True).primary_sites.first()
    support = climate_rows.loc[climate_rows.method.eq(METHODS[0])].set_index('level')
    values = {
        'OverviewPeakMedianRange': f'{peaks.min():g} to {peaks.max():g}',
        'OverviewDurationMedianRange': f'{durations.min():g} to {durations.max():g}',
        'OverviewJanFrozenShare': fmt_value(phase_share.loc[(1, 'frozen_likely')], 1),
        'OverviewJulLiquidShare': fmt_value(phase_share.loc[(7, 'liquid_likely')], 1),
        'OverviewShortMidShare': fmt_value(profile_share.loc[('lt1h', 'mid_loaded')], 1),
        'OverviewShortSingleStepShare': fmt_value(100 * len(single) / len(short), 1),
        'OverviewSoilDryPearsonRange': fr'\ensuremath{{[{supported.min():.3f},\,{supported.max():.3f}]}}',
        'OverviewSoilDryASeasonSites': fmt_count(a_sites.iloc[0]),
        'OverviewSoilDryEPairs': fmt_count(soil.loc[soil.climate.eq('E'), 'paired_events'].sum()),
        'ReplayYearSpreadUpper': f'{math.ceil(10 * ranges.max()) / 10:.1f}',
        'ReplayPrimarySitesPerYear': '/'.join(map(str, year_sites)),
        'ReplayClimateAEvents': fmt_count(support.loc['A', 'all_replay_events']),
        'ReplayClimateESites': fmt_count(support.loc['E', 'primary_sites']),
    }
    for part in ('core', 'precipitation', 'covariates', 'observations'):
        values[f'Release{part.title()}ExtractedGB'] = f'{packages[part]["bytes"] / 10**9:.3f}'
    details = {
        'short_event_count': len(short), 'short_single_interval_events': len(single),
        'annual_median_range_pp_by_method': ranges.to_dict(),
        'median_order_verified': list(METHODS), 'extracted_package_inventory': packages,
        'prose_macros': values,
    }
    return values, details


def read_frozen_extra_trees(release):
    config_path = 'provenance/benchmark_configuration.json'
    configured = read_json(release / config_path)['model']
    metadata = read_json(release / 'model_metadata.json')
    entry, = [row for row in metadata if row['variable'] == 'precipitation'
              and row['track'] == 'official_hourly']
    model = release / entry['path']
    require(sha256(model) == entry['sha256'], 'Frozen official precipitation model hash differs')
    # Read the scalar forest parameters in the compressed pickle header without loading the model.
    with model.open('rb') as handle:
        prefix = zlib.decompressobj().decompress(handle.read(32_768), 16_384)
    scalar_ops = {'SHORT_BINUNICODE', 'BINUNICODE', 'BININT', 'BININT1', 'BININT2', 'BINFLOAT',
                  'NONE', 'NEWFALSE', 'NEWTRUE'}
    keys = {'n_estimators', 'max_features', 'min_samples_leaf', 'max_depth', 'random_state'}
    memo, values = [], []
    for op, value, _ in pickletools.genops(prefix):
        if op.name == 'MEMOIZE':
            memo.append(values[-1] if values else None)
            continue
        if op.name in ('BINGET', 'LONG_BINGET'):
            value = memo[value]
        elif op.name not in scalar_ops:
            value = None
        if value == 'estimators_':
            break
        values.append(value)
    start = values.index('n_estimators')
    parameters = {key: value for key, value in zip(values[start:], values[start + 1:])
                  if isinstance(key, str) and key in keys}
    require(parameters.keys() == keys and 'ExtraTreesRegressor' in values
            and all(configured[key] == value for key, value in parameters.items() if key in configured),
            'Serialized forest settings do not match the frozen configuration')
    details = {'configuration': config_path, 'configuration_sha256': sha256(release / config_path),
               'model': entry['path'], 'model_sha256': entry['sha256'], 'parameters': parameters}
    names = {'n_estimators': 'ETTreeCount', 'max_features': 'ETMaxFeatures',
             'max_depth': 'ETMaxDepth', 'min_samples_leaf': 'ETMinLeaf', 'random_state': 'ETRandomSeed'}
    return {names[key]: str(value) for key, value in sorted(parameters.items())}, details


def proximity_statistics(sites, replay, primary_scores, profile_checks, diagnostic):
    train = sites.loc[sites.dataset_split.eq('train')].sort_values('station_wban')
    test = sites.loc[sites.dataset_split.eq('test')].sort_values('station_wban').copy()
    require(set(replay.station_wban) <= set(test.station_wban), 'Replay includes a non-test station')
    a = np.radians(test[['latitude', 'longitude']].to_numpy())[:, None, :]
    b = np.radians(train[['latitude', 'longitude']].to_numpy())[None, :, :]
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Station coordinates are missing')
    h = np.sin((a[:, :, 0] - b[:, :, 0]) / 2)**2
    h += np.cos(a[:, :, 0]) * np.cos(b[:, :, 0]) * np.sin((a[:, :, 1] - b[:, :, 1]) / 2)**2
    earth_radius_km, cutoff_km = 6371.0, 25.0
    nearest = h.argmin(axis=1)
    test['nearest_train_wban'] = train.station_wban.to_numpy()[nearest]
    test['nearest_train_km'] = 2 * earth_radius_km * np.arcsin(np.sqrt(h.min(axis=1).clip(0, 1)))
    test['active_replay'] = test.station_wban.isin(replay.station_wban)
    close = test.station_wban[test.active_replay & test.nearest_train_km.lt(cutoff_km)]
    active = test.loc[test.active_replay]
    rows = []
    for cell in profile_checks:
        group, method = cell['group'], cell['method']
        part = primary_scores.loc[primary_scores.method.eq(method)]
        part = part.loc[part.high_intensity.eq(True) if group == 'high_intensity_pooled'
                        else part.storm_profile_tercile.eq(group)]
        kept = part.loc[~part.station_wban.isin(close)]
        size = counts(kept)
        require(cell['reported'] and size['events'] >= diagnostic['minimum_events']
                and size['sites'] >= diagnostic['minimum_sites'], f'Insufficient support after exclusion: {group}, {method}')
        median = kept.underestimation_rate.median()
        rows.append({'profile': group, 'method': method, 'baseline_events': cell['events'],
                     'baseline_sites': cell['sites'], **size, 'original_median_pct': 100 * cell['median'],
                     'filtered_median_pct': 100 * median, 'absolute_change_pp': 100 * abs(median - cell['median'])})
    sensitivity = pd.DataFrame(rows)
    maximum = sensitivity.absolute_change_pp.max()
    ranges = {}
    displays = {}
    for name, part in (('primary', primary_scores), ('high', primary_scores.loc[primary_scores.high_intensity.eq(True)])):
        medians = part.groupby('method').underestimation_rate.median() * 100
        ranges[name] = {method: float(medians[method]) for method in METHODS}
        uniform, linear, trees = (fr'{rate:.1f}\%' for rate in ranges[name].values())
        displays[name] = f'{uniform}, {linear} and {trees}'
    values = {
        'TestStationMinDistMin': f'{test.nearest_train_km.min():.1f}',
        'TestStationMinDistMedian': f'{test.nearest_train_km.median():.1f}',
        'ReplayStationMinDistMin': f'{active.nearest_train_km.min():.1f}',
        'ReplayStationMinDistMedian': f'{active.nearest_train_km.median():.1f}',
        'ReplayCloseStationCount': fmt_count(len(close)),
        'ReplayCloseEventCount': fmt_count(replay.station_wban.isin(close).sum()),
        'ReplayClosePrimaryEventCount': fmt_count(primary_scores.loc[primary_scores.station_wban.isin(close), 'event_id'].nunique()),
        'ReplayCloseMaxShift': f'{maximum:.2f}',
        'ReplayPrimaryMedians': displays['primary'],
        'ReplayHighMedians': displays['high'],
    }
    details = {'earth_radius_km': earth_radius_km, 'exclusion_cutoff_km': cutoff_km,
               'test_sites': len(test), 'active_sites': len(active), 'excluded_active_wbans': close.tolist(),
               'max_absolute_change_pp': float(maximum), 'pooled_medians_pct': ranges}
    fields = ['station_wban', 'station', 'latitude', 'longitude', 'nearest_train_wban', 'nearest_train_km', 'active_replay']
    return test[fields], sensitivity, values, details


def summaries_to_macros(path, distributions, phases, profiles, soil, replay, scalars):
    climate = [(r.climate, fmt_count(r.events), fmt_count(r.sites), fmt_value(r.peak_15min_mm_median, 2),
                fmt_value(r.duration_min_median, 1)) for r in distributions.itertuples()]
    month = []
    for key, part in phases.groupby('month', sort=False):
        month.append((MONTHS[key - 1], fmt_count(part.denominator_events.iloc[0]),
                      fmt_count(part.denominator_sites.iloc[0]), *(fmt_value(v, 1) for v in part.share_pct)))
    duration = []
    labels = (r'\(<1\) h', r'1 to \(<3\) h', r'3 to \(<6\) h', r'\(\geq6\) h')
    for (_, part), label in zip(profiles.groupby('duration', sort=False), labels):
        duration.append((label, fmt_count(part.denominator_events.iloc[0]),
                         fmt_count(part.denominator_sites.iloc[0]), *(fmt_value(v, 1) for v in part.share_pct)))
    moisture = [(r.climate, r.season, fmt_count(r.paired_events), fmt_count(r.paired_sites),
                 fmt_value(r.pearson_r, 3), fmt_value(r.spearman_rho, 3)) for r in soil.itertuples()]
    values = {
        'OverviewClimateRows': climate, 'OverviewMonthlyPhaseRows': month,
        'OverviewDurationProfileRows': duration, 'OverviewSoilDryStrataRows': moisture,
    }
    for dimension, name in (('source_year', 'ReplayYearErrorRows'), ('climate', 'ReplayClimateErrorRows')):
        values[name] = []
        for key, group in replay.loc[replay.dimension.eq(dimension)].groupby('level', sort=False):
            row = group.iloc[0]
            errors = [f'{fmt_value(r.median_underestimation_pct, 1)} ({fmt_value(r.iqr_underestimation_pp, 1)})'
                      if r.report_underestimation else '---' for r in group.itertuples()]
            values[name].append((key, fmt_count(row.primary_events), fmt_count(row.primary_sites), *errors))
    header = ('% Generated by scripts/build_uscrn_manuscript_revision.py from the verified release.\n'
              '% Overview rows: category, events, WBANs, then figure statistics. Soil rows include season.\n'
              '% Replay rows: group, primary events, WBANs, then median (%) and IQR (pp) by method.\n')
    macros = {**{name: latex_rows(rows) for name, rows in values.items()}, **scalars}
    content = header + ''.join(fr'\newcommand{{\{name}}}{{{value}}}' + '\n' for name, value in macros.items())
    temporary = path.with_suffix('.tex.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def profile_sensitivity(primary_scores, diagnostic, macros):
    size_check = primary_scores.loc[primary_scores.method.eq(METHODS[0])]
    require(primary_scores.earliest_peak_ratio.between(0, 1).all(), 'Missing earliest-peak ratios in the primary cohort')
    earliest = np.select([primary_scores.earliest_peak_ratio.lt(.33), primary_scores.earliest_peak_ratio.gt(.67)],
                         ['front_loaded', 'back_loaded'], default='mid_loaded')
    scores = primary_scores.assign(earliest_profile=earliest)
    ref = scores.loc[scores.method.eq(METHODS[0])]
    agree = ref.earliest_profile.eq(ref.storm_profile_tercile).sum()
    check_scalar_macros(macros, {'ProfileMatches': fmt_count(agree), 'MainEvents': fmt_count(len(ref)),
                                'ProfileAgreement': f'{100 * agree / len(ref):.2f}'})
    rows = []
    for duration, scope in (('all', scores), ('at_least_60_min', scores.loc[scores.event_duration_min.ge(60)])):
        for rule in ('storm_profile_tercile', 'earliest_profile'):
            for method, group in scope.groupby('method'):
                for profile in PROFILES:
                    part = group.loc[group[rule].eq(profile)]
                    size = counts(part)
                    report = size['events'] >= diagnostic['minimum_events'] and size['sites'] >= diagnostic['minimum_sites']
                    median = spread(part.underestimation_rate) if report else dict.fromkeys(('median', 'iqr'), np.nan)
                    rows.append({'duration': duration, 'classification': rule, 'method': method,
                                 'profile': profile, **size, 'reportable': report,
                                 'median_underestimation_pct': 100 * median['median'], 'iqr_pp': 100 * median['iqr']})
    table = pd.DataFrame(rows)
    selected = table.loc[table.duration.eq('at_least_60_min') & table.classification.eq('storm_profile_tercile')]
    medians = selected.pivot(index='method', columns='profile', values='median_underestimation_pct')
    require(medians.mid_loaded.lt(medians[['front_loaded', 'back_loaded']].min(axis=1)).all(),
            'Long-event profile direction differs from the manuscript')
    trees = table.loc[table.method.eq(METHODS[-1]) & table.duration.eq('all')
                      & table.classification.eq('earliest_profile')].set_index('profile')
    gap = trees.loc['mid_loaded', 'median_underestimation_pct'] - trees.loc['front_loaded', 'median_underestimation_pct']
    require(gap > 0, 'Earliest-peak ExtraTrees contrast differs from the manuscript')
    long = size_check.loc[size_check.event_duration_min.ge(60)]
    values = {
        'ProfileAgreementDisplay': f'{100 * agree / len(ref):.1f}',
        'LongProfileEventCount': fmt_count(len(long)), 'LongProfileSiteCount': fmt_count(long.station_wban.nunique()),
        'EarliestTreeMedianGap': f'{gap:.1f}',
    }
    return table, values


def reporting_precision(release, events, baseline):
    values = {}
    tier_one = events.loc[events.usability_tier.eq(1)]
    for prefix, is_censored in (('Censored', True), ('Uncensored', False)):
        durations = tier_one.loc[tier_one.antecedent_dry_censored.eq(is_censored),
                                 'antecedent_dry_hours_lower_bound']
        quartiles = durations.quantile([.25, .75])
        values[prefix + 'Median'] = durations.median()
        values[prefix + 'IQR'] = quartiles.loc[.75] - quartiles.loc[.25]
    values['CensoredPct'] = 100 * tier_one.antecedent_dry_censored.eq(True).mean()

    eligibility = pd.read_csv(release / 'audit_log/eligibility_by_station_year.csv')
    for prefix, variable in (('Rain', 'precipitation'), ('Solar', 'solar_radiation')):
        rows = eligibility.loc[eligibility.variable.eq(variable) & eligibility.track.eq('derived')]
        annotations = release / 'data/annotations' / f'variable={variable}'
        zero = 0
        for year in sorted(rows.source_year.unique()):
            filename = annotations / f'source_year={year}' / 'part.parquet'
            column = pq.ParquetFile(filename).read(columns=['constraint_class'])['constraint_class']
            zero += pc.sum(pc.cast(pc.equal(column, 'unique_zero_completion'), 'int64')).as_py()
        values[prefix + 'ZeroPct'] = 100 * zero / rows.bounded_target_hours.sum()

    for name, value in values.items():
        require(f'{value:.2f}' == baseline[name], f'{name} differs from the archived result')
    return {name + 'Display': f'{value:.1f}' for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', nargs='?', type=Path, default=Path.cwd())
    parser.add_argument('--figure-dir', type=Path, default=ROOT / 'figures')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/uscrn_2007_2025/manuscript_revision')
    parser.add_argument('--macros-out', type=Path, default=ROOT / 'manuscript/station_disaggregation_revision_results.tex')
    args = parser.parse_args()
    release = args.release.resolve()
    for output in (args.figure_dir, args.report_dir, args.macros_out):
        require(not output.resolve().is_relative_to(release), 'Write revision outputs outside the frozen release')
    require(args.macros_out.resolve() not in {BASELINE_MACROS.resolve(), (ROOT / 'manuscript/full_data_descriptor.tex').resolve()},
            'The new results require a separate macro file')
    frozen, indexed = check_frozen_inputs(release)
    protocol = read_json(release / 'provenance/event_protocol.json')
    diagnostic = read_json(DIAGNOSTIC_PROTOCOL)
    require(diagnostic['minimum_events'] == protocol['error_strata']['min_events']
            and diagnostic['minimum_sites'] == protocol['error_strata']['min_stations']
            and tuple(protocol['envelope_replay']['methods']) == METHODS, 'Frozen diagnostic configurations differ')
    raw = BASELINE_MACROS.read_text(encoding='utf-8-sig')
    macros = dict(re.findall(r'^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}(?=\s*(?:^\\newcommand|\Z))', raw, re.M | re.S))
    events, sites, confirmed, tiers = load_catalogue(release, macros, frozen)
    print(f'Catalogue verified: {len(events):,} segments, {len(confirmed):,} confirmed events, {len(sites)} WBANs.', flush=True)
    replay, scores, main_scores, primary, bins, differences = load_replay(
        release, events, protocol, diagnostic, frozen, macros)
    print(f'Frozen replay verified: {len(replay):,} events, {replay.station_wban.nunique()} WBANs, {bins:,} steps.', flush=True)
    print(f'Primary replay cohort: {len(primary):,} events, {primary.station_wban.nunique()} WBANs.', flush=True)
    catalogue, composition = coverage_checks(release, macros, events, sites, replay)
    profile_checks, supplementary_score_cells = check_profile_scores(release, main_scores, scores, diagnostic, macros)
    distributions, phases, profiles, soil, overview_check = overview_tables(release, confirmed, macros, catalogue)
    years = protocol['envelope_replay']['evaluation_years']
    strata = replay_strata(scores, main_scores, primary, replay, years, diagnostic)
    packages = extracted_inventory(release, package_members(release))
    narrative, narrative_checks = manuscript_statistics(
        distributions, phases, profiles, soil, strata, confirmed, protocol, packages)
    model_macros, model_check = read_frozen_extra_trees(release)
    distances, distance_scores, distance_macros, distance_check = proximity_statistics(
        sites, replay, main_scores, profile_checks, diagnostic)
    sensitivity, sensitivity_macros = profile_sensitivity(main_scores, diagnostic, macros)
    dry = pd.read_csv(release / 'supplementary/dry_gap_sensitivity_summary.csv')
    dry_rows = [(f'{r.minimum_dry_hours:g} h', fmt_count(r.observed_segments),
                 fmt_count(r.boundary_confirmed_events), f'{r.median_duration_min:g}', f'{r.peak_15min_p95_mm:.1f}')
                for r in dry.itertuples(index=False)]
    narrative.update(model_macros)
    narrative.update(distance_macros)
    narrative.update(sensitivity_macros)
    display_macros = reporting_precision(release, events, macros)
    narrative.update(display_macros)
    narrative['ReplayPrimaryPeakMin'] = f'{diagnostic["diagnostic_min_observed_15min_mm"]:g}'
    narrative['SupplementaryDryDisplayRows'] = latex_rows(dry_rows)
    narrative['CatalogueCoverageDisplayRows'] = macros['CatalogueCoverageRows'].replace(
        r'\(>1500\) m', r'\(>1{,}500\) m')
    narrative_checks.update(frozen_extra_trees=model_check, spatial_proximity=distance_check,
                            one_decimal_display=display_macros)
    require(all(observed == indexed[name]['rows'] for name, observed in (
        ('data/events.parquet', len(events)), ('data/station_metadata.csv', len(sites)),
        ('data/envelope_timeseries.parquet', bins), ('data/envelope_event_scores.csv', frozen['score_rows']))),
        'Frozen indexed row counts differ')
    print('Catalogue, replay coverage, profile diagnostics and S1/S4 results reconcile.', flush=True)
    print('Each main/full replay stratum reconciles by method for both source year and climate.', flush=True)

    for folder in (args.figure_dir, args.report_dir, args.macros_out.parent):
        folder.mkdir(parents=True, exist_ok=True)
    deposit_macros = ROOT / 'manuscript/station_disaggregation_deposit_sizes.tex'
    rendered_counts = {**macros, **narrative}
    if deposit_macros.is_file():
        rendered_counts.update(scalar_macros(deposit_macros))
    figs = []
    with plt.rc_context(STYLE):
        for fig, name in ((workflow_figure(rendered_counts, protocol['minimum_dry_hours']),
                           'revision_workflow_schema.pdf'),
                          (overview_figure(distributions, phases, profiles, soil), 'revision_event_data_overview.pdf'),
                          (replay_figure(strata, years, diagnostic), 'revision_replay_peak15_strata.pdf')):
            figs.append(export_figure(fig, args.figure_dir / name))
    for frame, filename in ((distributions, 'overview_climate.csv'), (phases, 'overview_phase_month.csv'),
                            (profiles, 'overview_profile_duration.csv'), (soil, 'overview_soil_dry_climate_season.csv'),
                            (strata, 'replay_peak15_strata.csv'), (distances, 'station_proximity.csv'),
                            (distance_scores, 'replay_proximity_sensitivity.csv'),
                            (sensitivity, 'replay_profile_sensitivity.csv')):
        frame.to_csv(args.report_dir / filename, index=False, lineterminator='\n')
    summaries_to_macros(args.macros_out, distributions, phases, profiles, soil, strata, narrative)

    summary = {
        'status': 'passed',
        'runtime': {'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__,
                    'matplotlib': matplotlib.__version__},
        'frozen_fingerprints': {name: entry['sha256'] for name, entry in indexed.items()},
        'portable_index_sha256': frozen['portable_index_sha256'], 'baseline_macros_sha256': sha256(BASELINE_MACROS),
        'catalogue': {'segments': len(events), 'stations': len(sites), 'confirmed': len(confirmed), **tiers,
                      'coverage_rows_matching': len(catalogue)},
        'overview': overview_check,
        'replay': {'events': len(replay), 'sites': replay.station_wban.nunique(), 'five_minute_steps': bins,
                   'primary': counts(primary), 'high_intensity': counts(replay.loc[replay.high_intensity.eq(True)]),
                   'coverage_rows_matching': len(composition), 'window_minutes': 15,
                   'observed_min_mm': diagnostic['diagnostic_min_observed_15min_mm'],
                   'peak_tolerance_mm': protocol['peak_depth_tolerance_mm'], 'min_sites': diagnostic['minimum_sites'],
                   'min_events': diagnostic['minimum_events'], 'table6_profile_cells': profile_checks,
                   's1_pooled_15min_cells_matching': supplementary_score_cells, 'strata_match_per_method': True,
                   'year_and_climate_sizes': strata.loc[strata.method.eq(METHODS[0]), [
                       'dimension', 'level', 'primary_events', 'primary_sites', 'all_replay_events', 'all_replay_sites',
                       'report_underestimation', 'report_all_peak_mae']].to_dict('records'),
                   'independent_15min_comparison_counts': {'observed': len(replay), 'reconstructed': len(scores)},
                   'peak_max_abs_difference_mm': differences},
        'figures': figs, 'manuscript_statistics': narrative_checks,
        'macros': {'file': args.macros_out.name, 'sha256': sha256(args.macros_out)},
    }
    report = args.report_dir / 'step2_validation.json'
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Passed: {report.resolve()}', flush=True)


if __name__ == '__main__':
    main()
