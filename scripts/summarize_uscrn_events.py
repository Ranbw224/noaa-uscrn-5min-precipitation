"""Report event counts and diagnostic coverage after the M3 gate passes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from uscrn_assertions import assert_tier_mapping, assert_tier_partition_and_nulls
from benchmark_full import ROOT, save_json, sha256
from build_uscrn_release import DEFAULT_CONFIG, read_json
from uscrn_event_diagnostics import SOIL_REASON_FIELDS, STRING_DIAGNOSTICS


def run(config_path):
    config = read_json(config_path)
    release = ROOT / config['release_directory']
    report = ROOT / config['reports_directory'] / 'm3'
    gate = read_json(report / 'summary.json')
    if gate['status'] != 'passed' or set(gate['assertions'].values()) != {'passed'}:
        raise ValueError('M3 has not passed')
    path = release / 'data/events.parquet'
    if sha256(path) != gate['event_catalogue_sha256']:
        raise ValueError('M3 catalogue changed before its diagnostic summary')
    events = pd.read_parquet(path)
    assert_tier_mapping(events)
    assert_tier_partition_and_nulls(events)
    strings = ['storm_profile_tercile', *STRING_DIAGNOSTICS]
    arrow = pq.ParquetFile(path).read(columns=['usability_tier', *strings])
    censored = arrow.filter(pc.equal(arrow['usability_tier'], 3))
    if any(censored[name].null_count != censored.num_rows for name in strings):
        raise AssertionError('#2 Tier 3 string diagnostics must be physical Parquet nulls')
    confirmed = events.loc[events.usability_tier.isin([1, 2])]
    phase = confirmed.groupby('precip_phase_flag').agg(events=('event_id', 'size'),
                                                      stations=('station_wban', 'nunique')).reset_index()
    phase['fraction_of_confirmed'] = phase.events / len(confirmed)
    phase.to_csv(report / 'phase_distribution.csv', index=False)
    available = []
    for field in ('soil_moisture_pre_5cm', 'event_wind_mean_ms', 'event_wetbulb_mean_c'):
        valid = np.isfinite(confirmed[field])
        available.append({'field': field, 'applicable_events': len(confirmed),
                          'valid_events': int(valid.sum()), 'valid_sites': confirmed.loc[valid, 'station_wban'].nunique(),
                          'fraction_of_confirmed': float(valid.mean()),
                          'fraction_of_all_segments': float(valid.sum() / len(events))})
    pd.DataFrame(available).to_csv(report / 'event_diagnostic_coverage.csv', index=False)
    reasons = []
    for field in SOIL_REASON_FIELDS:
        reasons.extend({'field': field, 'reason': label, 'events': int(n)}
                       for label, n in confirmed[field].value_counts().items())
    pd.DataFrame(reasons).to_csv(report / 'soil_reason_counts.csv', index=False)
    result = {'segments': len(events), 'confirmed_events': len(confirmed),
              'tiers': {str(tier): int(events.usability_tier.eq(tier).sum()) for tier in (1, 2, 3)},
              'confirmed_cross_year_with_nominal_dry_buffers': int(confirmed.cross_year.sum()),
              'confirmed_observed_span_cross_year': int(confirmed.first_source_year.ne(confirmed.last_source_year).sum()),
              'tier3_nominal_cross_year_flags': int(events.loc[events.usability_tier.eq(3), 'cross_year'].sum()),
              'phase_distribution': phase.to_dict('records'), 'coverage': available,
              'tier3_arrow_string_nulls': 'passed', 'assertions': gate['assertions']}
    save_json(report / 'event_diagnostics_summary.json', result)
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
