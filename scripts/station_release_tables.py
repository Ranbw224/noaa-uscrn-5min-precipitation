"""Build event-tier and hourly constraint data."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import pandas as pd
from benchmark_full import sha256

STEP = pd.Timedelta(minutes=5)

HOUR = pd.Timedelta(hours=1)

PRECIPITATION = 'precipitation'

@dataclass
class AnnualRain:
    year: int
    first_record: int
    amounts: np.ndarray
    quality: np.ndarray
    preceding_blocker: np.ndarray
    target: np.ndarray
    official: np.ndarray
    derived: np.ndarray
    official_inputs: np.ndarray
    official_values: np.ndarray

def bound(values, variable):
    valid = np.isfinite(values)
    if variable in (PRECIPITATION, 'solar_radiation'):
        valid &= values >= 0
    if variable == 'relative_humidity':
        valid &= (values >= 0) & (values <= 100)
    return valid

def origin(year):
    return pd.Timestamp(year=year, month=1, day=1, tz='UTC')

def positions(timestamps, year, step, length):
    offsets = pd.DatetimeIndex(timestamps).as_unit('ns').asi8 - origin(year).value
    index = offsets // step.value - 1
    if (offsets % step.value).any() or (index < 0).any() or (index >= length).any():
        raise ValueError(f'Timestamps outside right-closed grid for {year}')
    return index

def describe(path, rows, release):
    return {'path': path.relative_to(release).as_posix(), 'rows': rows, 'bytes': path.stat().st_size,
            'sha256': sha256(path)}

def export_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression='zstd')

def annotate(annotations, hourly, year, settings):
    climate = ['precipitation', 'solar_radiation', 'relative_humidity']
    diagnostic = hourly.pivot(index='time_end_utc', columns='variable',
                             values=settings['input_column'])[climate]
    if not annotations[['time_end_utc', 'variable']].duplicated().sum() == 0:
        raise ValueError(f'Duplicate annotation keys in {year}')
    original_keys = pd.MultiIndex.from_frame(annotations[['time_end_utc', 'variable']])
    new_keys = pd.MultiIndex.from_frame(hourly[['time_end_utc', 'variable']])
    if not original_keys.sort_values().equals(new_keys.sort_values()):
        raise ValueError(f'Annotation and pair keys differ in {year}')
    rain, solar, rh = diagnostic.to_numpy(dtype=float).T
    wet = rain > settings['rain_above_mm']
    hour = diagnostic.index.hour
    first, last = settings['utc_hour_end_inclusive']
    flags = np.select(
        [~np.isfinite(diagnostic.to_numpy(dtype=float)).all(axis=1),
         wet & (rh < settings['low_rh_below_percent']),
         wet & (solar > settings['high_solar_above_W_m2']) & (hour >= first) & (hour <= last)],
        ['missing_covariate', 'rain_low_rh', 'rain_high_rad'], default='consistent')
    annotations['cross_var_consistency_flag'] = pd.Series(flags, index=diagnostic.index).reindex(
        annotations.time_end_utc).to_numpy()
    return Counter(flags)

def covered_slices(start, end, grid_step, bins_by_year):
    """Yield the slices for the left-open, right-closed UTC span (start, end]."""
    for year in range(start.year, (end - grid_step).year + 1):
        start_bin = max(0, int((start - origin(year)) // grid_step))
        stop_bin = min(len(bins_by_year[year]), int((end - origin(year)) // grid_step))
        yield year, slice(start_bin, stop_bin)

def enrich_events(events, years, settings):
    rain = {year: data.amounts for year, data in years.items()}
    official = {year: data.official for year, data in years.items()}
    padding = settings['dry_buffer_steps'] * STEP
    low, high = settings['storm_profile_cutpoints']
    tolerance = settings['precipitation_comparison_tolerance_mm']
    enriched = []
    for event in events.itertuples(index=False):
        start, end = event.event_start_utc, event.event_end_utc
        start_year = years[start.year]
        first = int((start - origin(start.year)) // STEP)
        blocker = int(start_year.preceding_blocker[first - 1]) if first else -1
        dry_steps = first - 1 - blocker
        if blocker < start_year.first_record:
            stop_reason = 'source_year_record_start'
        elif start_year.amounts[blocker] > 0:
            stop_reason = 'preceding_valid_precipitation'
        else:
            stop_reason = f'fine_quality_code_{start_year.quality[blocker]}'
        censored = stop_reason != 'preceding_valid_precipitation'
        fail_count = 0
        derived_fail_count = 0
        fail_years = []
        for year, window in covered_slices(event.first_covering_hour_end_utc - HOUR,
                                           event.last_covering_hour_end_utc, HOUR, official):
            unavailable = int((~years[year].official[window]).sum())
            fail_count += unavailable
            derived_fail_count += int((~years[year].derived[window]).sum())
            if unavailable:
                fail_years.append(str(year))
        tier = 1 if event.boundary_confirmed and not fail_count else 2 if event.boundary_confirmed else 3
        centroid, earliest, profile, tie = np.nan, np.nan, None, None
        if event.boundary_confirmed:
            sequence = np.concatenate([rain[year][window] for year, window in covered_slices(start, end, STEP, rain)])
            n = len(sequence)
            amount = sequence.sum()
            if (n != event.fine_intervals or not np.isfinite(sequence).all() or amount <= 0
                    or abs(amount - event.observed_depth_mm) > tolerance or sequence[0] <= 0 or sequence[-1] <= 0):
                raise ValueError(f'Observed event does not match frozen intervals: {event.event_id}')
            centroid = 0.5 if n == 1 else float(sequence @ (np.arange(n) + 0.5) / (n * amount))
            decision_value, lower_limit, upper_limit = centroid, low, high
            if min(abs(centroid - low), abs(centroid - high)) <= 2e-15:
                # The source depths are decimal. Resolve cutpoint ties before rounding the stored ratio.
                depths = [Fraction(str(value)) for value in sequence]
                decision_value = sum(p * (2 * i + 1) for i, p in enumerate(depths)) / (2 * n * sum(depths))
                lower_limit, upper_limit = Fraction(str(low)), Fraction(str(high))
                centroid = float(decision_value)
            profile = ('front_loaded' if decision_value < lower_limit else
                       'back_loaded' if decision_value > upper_limit else 'mid_loaded')
            earliest = (int(sequence.argmax()) + 0.5) / n
            tie = bool(np.count_nonzero(sequence == sequence.max()) > 1)
        enriched.append({
            'event_id': event.event_id, 'usability_tier': tier,
            'cross_year': (start - padding).year != (end + padding - STEP).year,
            'ineligible_covering_hours': fail_count, 'ineligible_source_years': ','.join(fail_years),
            'temporal_centroid_ratio': centroid, 'storm_profile_tercile': profile, 'earliest_peak_ratio': earliest,
            'has_max_interval_tie': tie, 'antecedent_dry_hours_lower_bound': dry_steps / 12,
            'antecedent_dry_censored': censored, 'antecedent_stop_reason': stop_reason,
            'antecedent_terminator_utc': origin(start.year) + (blocker + 1) * STEP if blocker >= 0 else pd.NaT,
            'ineligible_derived_covering_hours': derived_fail_count,
        })
    return enriched
