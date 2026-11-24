"""Build the observed NOAA precipitation event catalogue."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from precipitation_intervals import DRY_STEPS, event_spans, read_partition, rolling_amounts

FIVE_MINUTES = pd.Timedelta(minutes=5)

QUALITY_STATES = ('valid', 'absent_fine_row', 'invalid_fine_value', 'negative_fine_value')

@dataclass
class AnnualPrecipitation:
    year: int
    amounts: np.ndarray
    quality: np.ndarray
    hour_complete: np.ndarray
    hour_context: np.ndarray
    hour_current: np.ndarray
    eligible: np.ndarray
    counts: dict

def annual_grid(first, last, frequency):
    return pd.date_range(f'{first}-01-01', f'{last + 1}-01-01', inclusive='right',
                         freq=frequency, tz='UTC')

def read_station_year(record, expected_benchmark_hours, *, fine_frame=None, hourly_frame=None):
    year = record['year']
    fine_grid = annual_grid(year, year, '5min')
    hour_grid = annual_grid(year, year, 'h')
    if fine_frame is None:
        fine = read_partition(record, 'high_frequency', [
            'time_end_utc', 'station_wban', 'precipitation', 'precipitation_exclusion_code',
        ], fine_grid)
    else:
        fine = fine_frame.set_index('time_end_utc').reindex(fine_grid)
    if hourly_frame is None:
        hourly = read_partition(record, 'hourly_pairs', [
            'time_end_utc', 'station_wban', 'variable', 'official_hourly', 'official_source_present',
            'valid_five_minute_count', 'fine_record_count',
        ], hour_grid)
    else:
        hourly = hourly_frame.loc[hourly_frame.variable.eq('precipitation')].set_index('time_end_utc').reindex(hour_grid)
    present = fine.station_wban.notna().to_numpy()
    amounts = fine.precipitation.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(amounts)
    if not np.array_equal(finite, fine.precipitation_exclusion_code.eq(0).to_numpy()):
        raise ValueError(f'Fine values and quality codes differ at {record["station"]}/{year}')
    quality = np.zeros(len(fine_grid), dtype=np.uint8)
    quality[~present] = 1
    quality[present & ~finite] = 2
    quality[finite & (amounts < 0)] = 3
    amounts[quality != 0] = np.nan
    valid_per_hour = finite.reshape(-1, 12).sum(axis=1)
    present_per_hour = present.reshape(-1, 12).sum(axis=1)
    if (not np.array_equal(valid_per_hour, hourly.valid_five_minute_count.fillna(0))
            or not np.array_equal(present_per_hour, hourly.fine_record_count.fillna(0))):
        raise ValueError(f'Fine interval counts disagree with pairing table at {record["station"]}/{year}')

    official = hourly.official_hourly.to_numpy(dtype=float)
    available = np.isfinite(official)
    source_present = hourly.official_source_present.eq(True).to_numpy()
    if (available & ~source_present).any():
        raise ValueError('Available official precipitation on an absent source row')
    # Match the original official-hourly benchmark within each source-year partition.
    extended = np.r_[False, available, False]
    context = available & extended[:-2] & extended[2:]
    current = available & (official >= 0)
    complete = valid_per_hour == 12
    eligible = complete & context & current
    if expected_benchmark_hours is not None and int(eligible.sum()) != expected_benchmark_hours:
        raise ValueError(f'Legacy hourly benchmark mask changed at {record["station"]}/{year}')
    counts = {
        'fine_grid_intervals': len(fine_grid), 'fine_present_intervals': int(present.sum()),
        'fine_wet_intervals': int((amounts > 0).sum()), 'fine_observed_zero_intervals': int((amounts == 0).sum()),
        'fine_absent_intervals': int((quality == 1).sum()), 'fine_invalid_present_intervals': int((quality == 2).sum()),
        'fine_negative_intervals': int((quality == 3).sum()), 'valid_fine_intervals': int((quality == 0).sum()),
        'official_current_usable_hours': int(current.sum()), 'hourly_benchmark_eligible_hours': int(eligible.sum()),
    }
    counts['valid_observation_hours'] = counts['valid_fine_intervals'] / 12
    return AnnualPrecipitation(year, amounts, quality, complete, context, current, eligible, counts)

def confirmed_buffer(quality, first_wet, direction, dry_steps):
    positions = first_wet + direction * np.arange(1, dry_steps + 1)
    in_bounds = (positions >= 0) & (positions < len(quality))
    valid_positions = positions[in_bounds]
    gap = np.flatnonzero(quality[valid_positions])
    if gap.size:
        count = int(gap[0])
        return count, QUALITY_STATES[quality[valid_positions[count]]]
    count = len(valid_positions)
    return count, 'observed_dry' if count == dry_steps else 'coverage_block_edge'

def catalogue_peaks(amounts, first, last, timestamps, specification):
    padding = specification['peak_support_dry_minutes_each_side'] // 5
    first_support, last_support = first - padding, last + padding
    series = amounts[first_support:last_support + 1]
    if (not np.all(amounts[first_support:first] == 0)
            or not np.all(amounts[last + 1:last_support + 1] == 0)
            or not np.isfinite(series).all()):
        raise ValueError('Confirmed event lacks the required observed zero support')
    result = {'peak_support_start_utc': timestamps[first_support] - FIVE_MINUTES,
              'peak_support_end_utc': timestamps[last_support]}
    for minutes in specification['peak_windows_minutes']:
        width = minutes // 5
        sums = rolling_amounts(series, width)
        index = int(np.flatnonzero(sums >= sums.max() - specification['peak_depth_tolerance_mm'])[0])
        result[f'peak_{minutes}min_mm'] = float(sums.max())
        result[f'peak_{minutes}min_end_utc'] = timestamps[first_support + index + width - 1]
    return result

def catalogue_block(annual, station, specification):
    first_year, last_year = annual[0].year, annual[-1].year
    amounts = np.concatenate([item.amounts for item in annual])
    quality = np.concatenate([item.quality for item in annual])
    complete = np.concatenate([item.hour_complete for item in annual])
    current = np.concatenate([item.hour_current for item in annual])
    context = np.concatenate([item.hour_context for item in annual])
    eligible = np.concatenate([item.eligible for item in annual])
    fine_times = annual_grid(first_year, last_year, '5min')
    hour_times = annual_grid(first_year, last_year, 'h')
    if len(amounts) != len(fine_times):
        raise ValueError('Unexpected gap or overlap between concatenated source years')
    year_stops = np.cumsum([len(item.amounts) for item in annual])

    rows = []
    for first, last in event_spans(amounts):
        before, before_state = confirmed_buffer(quality, first, -1, DRY_STEPS)
        after, after_state = confirmed_buffer(quality, last, 1, DRY_STEPS)
        if np.any(amounts[first - before:first] != 0) or np.any(amounts[last + 1:last + 1 + after] != 0):
            raise ValueError('An observed boundary buffer contains precipitation')
        confirmed = before == after == DRY_STEPS
        start = fine_times[first] - FIVE_MINUTES
        end = fine_times[last]
        start_year, end_year = np.searchsorted(year_stops, [first, last], side='right') + first_year
        window = slice(first // 12, last // 12 + 1)
        span = amounts[first:last + 1]
        n_eligible = int(eligible[window].sum())
        row = {
            'event_id': f'{station.station_wban}_{start:%Y%m%dT%H%MZ}',
            'station_wban': station.station_wban, 'station': station.station, 'network': station.network,
            'station_split': station.station_split, 'kg_class': station.kg_class, 'kg_group': station.kg_group,
            'first_source_year': int(start_year), 'last_source_year': int(end_year),
            'event_start_utc': start, 'event_end_utc': end, 'event_duration_min': len(span) * 5,
            'fine_intervals': len(span), 'fine_wet_intervals': int((span > 0).sum()),
            'fine_observed_zero_intervals': int((span == 0).sum()),
            'observed_depth_mm': round(float(span.sum()), 9), 'boundary_confirmed': confirmed,
            'pre_dry_intervals': before, 'post_dry_intervals': after,
            'pre_boundary_status': before_state, 'post_boundary_status': after_state,
            'first_covering_hour_end_utc': hour_times[window.start],
            'last_covering_hour_end_utc': hour_times[window.stop - 1],
            'covering_hours': window.stop - window.start,
            'covering_hours_with_complete_fine': int(complete[window].sum()),
            'covering_hours_with_usable_current': int(current[window].sum()),
            'covering_hours_with_finite_context': int(context[window].sum()),
            'hourly_benchmark_eligible_hours': n_eligible,
            'hourly_benchmark_eligible': n_eligible == window.stop - window.start,
        }
        if confirmed:
            row.update(catalogue_peaks(amounts, first, last, fine_times, specification))
        rows.append(row)
    if sum(row['fine_wet_intervals'] for row in rows) != int((amounts > 0).sum()):
        raise ValueError('Observed wet intervals do not reconcile with the catalogue')
    if not np.isclose(sum(row['observed_depth_mm'] for row in rows), np.nansum(amounts),
                      atol=1e-7, rtol=0):
        raise ValueError('Observed precipitation depth does not reconcile')
    return rows

def dry_period_spans(values, dry_steps):
    if dry_steps == DRY_STEPS:
        return event_spans(values)
    wet = np.flatnonzero(values > 0)
    if not len(wet):
        return []
    invalid = np.r_[0, np.cumsum(~np.isfinite(values), dtype=np.int64)]
    split = (np.diff(wet) > dry_steps) | (invalid[wet[1:]] != invalid[wet[:-1] + 1])
    return list(zip(wet[np.r_[True, split]], wet[np.r_[split, True]], strict=True))

def sensitivity_sample(records, stations, specification):
    table = pd.DataFrame(records)[['station_wban', 'year']].merge(
        stations.reset_index()[['station_wban', 'network', 'kg_group']], on='station_wban', validate='many_to_one')
    seed = specification['dry_gap_sensitivity']['sample_seed']
    table['rank_sha256'] = [hashlib.sha256(f'{seed}:{r.station_wban}:{r.year}'.encode()).hexdigest()
                            for r in table.itertuples(index=False)]
    groups = ['year', 'network', 'kg_group']
    return table.sort_values('rank_sha256').drop_duplicates(groups).sort_values(groups).reset_index(drop=True)

def sensitivity_results(annual, station, specification):
    values = annual.amounts
    padding = specification['peak_support_dry_minutes_each_side'] // 5
    rows = []
    for hours in specification['dry_gap_sensitivity']['dry_hours']:
        dry_steps = hours * 12
        spans = dry_period_spans(values, dry_steps)
        durations, maxima = [], []
        for first, last in spans:
            if first < dry_steps or last + dry_steps >= len(values):
                continue
            if not ((values[first - dry_steps:first] == 0).all() and (values[last + 1:last + 1 + dry_steps] == 0).all()):
                continue
            durations.append((last - first + 1) * 5)
            maxima.append(float(rolling_amounts(values[first - padding:last + padding + 1], 3).max()))
        rows.append(({
            'station_wban': station.station_wban, 'source_year': annual.year,
            'network': station.network, 'kg_group': station.kg_group, 'minimum_dry_hours': hours,
            'observed_segments': len(spans), 'boundary_confirmed_events': len(maxima),
        }, durations, maxima))
    return rows

def assemble_sensitivity(results):
    pooled = defaultdict(lambda: {'segments': 0, 'duration': [], 'peaks': []})
    rows = []
    for base, durations, maxima in results:
        rows.append({**base, 'median_duration_min': np.median(durations) if durations else np.nan,
                     'peak_15min_p95_mm': np.quantile(maxima, .95) if maxima else np.nan})
        group = pooled[base['minimum_dry_hours']]
        group['segments'] += base['observed_segments']
        group['duration'].extend(durations)
        group['peaks'].extend(maxima)
    summaries = [
        {'minimum_dry_hours': hours, 'station_years': len(rows) // len(pooled),
         'observed_segments': group['segments'], 'boundary_confirmed_events': len(group['peaks']),
         'median_duration_min': np.median(group['duration']), 'peak_15min_p95_mm': np.quantile(group['peaks'], .95)}
        for hours, group in sorted(pooled.items())
    ]
    return pd.DataFrame(rows), pd.DataFrame(summaries)
