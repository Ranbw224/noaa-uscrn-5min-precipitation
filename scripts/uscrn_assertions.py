"""Six hard release checks from section V.1 of the frozen 2007–2025 specification."""

import csv
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmark_full import ROOT, sha256
from build_full_dataset import split_for_wban
from uscrn_event_diagnostics import (
    AFTER_DEPLOYMENT, BEFORE_DEPLOYMENT, DEPTHS, FIVE_MINUTES, MINUTE,
    NUMERIC_DIAGNOSTICS, SOIL_OFFSET, SOIL_REASON_FIELDS, SOIL_VALUE_FIELDS, STRING_DIAGNOSTICS, VALID,
)


PROFILE_FIELDS = ('temporal_centroid_ratio', 'storm_profile_tercile', 'earliest_peak_ratio',
                  'has_max_interval_tie')
REFERENCE_START = pd.Timestamp('2007-01-01T00:00:00Z')
REFERENCE_END = pd.Timestamp('2020-12-31T23:59:59Z')
BUFFER = pd.Timedelta(hours=6)


def event_errors(number, label, frame, matched):
    bad = ~pd.Series(matched, index=frame.index).fillna(False)
    if bad.any():
        examples = frame.loc[bad, 'event_id'].astype(str).head(12).tolist()
        raise AssertionError(f'#{number} {label}: {int(bad.sum())} offending events, first: {examples}')


def assert_tier_mapping(events):
    count = events.ineligible_covering_hours
    eligible = events.hourly_benchmark_eligible
    event_errors(1, 'covering-hour equivalence', events,
                 count.notna() & count.ge(0) & eligible.isin([True, False])
                 & count.eq(0).eq(eligible))


def assert_tier_partition_and_nulls(events):
    confirmed, eligible = events.boundary_confirmed, events.hourly_benchmark_eligible
    known = confirmed.isin([True, False]) & eligible.isin([True, False])
    expected = np.where(confirmed.eq(True).fillna(False), np.where(eligible.eq(True).fillna(False), 1, 2), 3)
    event_errors(2, 'exclusive and exhaustive tier mapping', events, known & events.usability_tier.eq(expected))
    tier_three = events.loc[events.usability_tier.eq(3)]
    fields = [*PROFILE_FIELDS, *NUMERIC_DIAGNOSTICS, *STRING_DIAGNOSTICS]
    event_errors(2, 'Tier 3 diagnostics must be null', tier_three, tier_three.loc[:, fields].isna().all(axis=1))


def assert_p95_reference(events):
    if set(events.dataset_split.unique()) != {'train'}:
        raise AssertionError('#3 The P95 reference requires only train stations and at least one event')
    start = pd.to_datetime(events.event_start_utc, utc=True)
    end = pd.to_datetime(events.event_end_utc, utc=True)
    event_errors(3, 'confirmed events and both dry buffers must lie in the P95 reference window', events,
                 events.boundary_confirmed.eq(True) & (start - BUFFER >= REFERENCE_START)
                 & (end + BUFFER <= REFERENCE_END))


def assert_station_and_feature_policy(stations, expected_wbans, feature_names, *, validation_tuning_trials):
    wbans = stations.station_wban
    if wbans.isna().any() or wbans.duplicated().any() or set(wbans) != set(expected_wbans):
        raise AssertionError('#4 Station metadata must cover exactly the WBAN inventory')
    expected_split = wbans.map(split_for_wban)
    if not (stations.station_split.eq(expected_split) & stations.dataset_split.eq(expected_split)).fillna(False).all():
        raise AssertionError('#4 station_split and dataset_split must agree with the frozen WBAN hash')
    policy = json.loads((ROOT / 'config/task_input_policy.json').read_text(encoding='utf-8'))
    primary = tuple(policy['allowed_primary_features'])
    expanded = primary + tuple(f'{name}_missing' for name in primary)
    if tuple(feature_names) not in (primary, expanded):
        raise AssertionError('#4 Feature names must match the whitelisted coarse features and their allowed indicators')
    if not isinstance(validation_tuning_trials, Integral) or isinstance(validation_tuning_trials, bool) or (
            validation_tuning_trials != 0):
        raise AssertionError('#4 Validation stations require an explicit count of zero tuning trials')


def assert_soil_moisture_attribution(events, first_valid_by_wban):
    eligible = events.loc[events.usability_tier.isin([1, 2])]
    starts = pd.to_datetime(eligible.event_start_utc, utc=True)
    presence = pd.Series(False, index=eligible.index)
    for depth, value_column, reason_column in zip(DEPTHS, SOIL_VALUE_FIELDS, SOIL_REASON_FIELDS, strict=True):
        values = eligible[value_column]
        reasons = eligible[reason_column]
        lookup = {wban: first_valid_by_wban[wban][depth] for wban in eligible.station_wban.unique()}
        first = pd.to_datetime(eligible.station_wban.map(lookup), utc=True)
        valid = values.notna()
        not_deployed = first.isna() | starts.lt(first)
        expected_reason = np.where(valid, VALID, np.where(not_deployed, BEFORE_DEPLOYMENT, AFTER_DEPLOYMENT))
        event_errors(5, f'{depth} cm values and deployment reasons', eligible,
                     reasons.eq(expected_reason) & (~valid | (values.between(0, 1) & ~not_deployed)))
        presence |= valid
    offset = eligible[SOIL_OFFSET]
    expected_offset = (starts - (starts - FIVE_MINUTES).dt.floor('h')) / MINUTE
    event_errors(5, 'shared offset', eligible,
                 offset.notna().eq(presence) & (~presence | (offset.between(5, 60) & offset.eq(expected_offset))))


def assert_portable_index_and_strata(root, portable_index, cohort_totals, strata):
    """Verify indexed data and additive strata before 2007–2025 macro generation."""
    root = Path(root).resolve()
    indexed = [output for record in portable_index['records'] for output in record['outputs'].values()]
    if not indexed:
        raise AssertionError('#6 The portable index is empty')
    paths = set()
    for entry in indexed:
        relative = Path(entry['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path in paths or not path.is_file():
            raise AssertionError(f'#6 Invalid or duplicate indexed path: {relative}')
        paths.add(path)
        if path.suffix == '.parquet':
            rows = pq.ParquetFile(path).metadata.num_rows
        elif path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as data:
                reader = csv.reader(data)
                next(reader, None)
                rows = sum(1 for _ in reader)
        else:
            raise AssertionError(f'#6 Unexpected indexed dataset format: {relative}')
        if rows != entry['rows'] or sha256(path) != entry['sha256']:
            raise AssertionError(f'#6 Indexed row count or SHA-256 mismatch: {relative}')
    if set(cohort_totals) != set(strata) or not {'catalogue', 'replay'} <= set(strata):
        raise AssertionError('#6 Missing catalogue or replay accounting')
    for cohort, dimensions in strata.items():
        if not dimensions:
            raise AssertionError(f'#6 No stratified counts supplied for {cohort}')
        for dimension, counts in dimensions.items():
            counts = tuple(counts)
            if not counts or any(not isinstance(n, Integral) or n < 0 for n in counts) or (
                    sum(counts) != cohort_totals[cohort]):
                raise AssertionError(f'#6 Stratified counts do not reconcile for {cohort}/{dimension}')
