"""Score precipitation reconstructions over complete covering hours."""

import numpy as np
import pandas as pd

METHOD_META = ['event_id', 'station_wban', 'station', 'network', 'kg_class', 'kg_group', 'high_intensity']

def convolution_peak(values, width, tolerance):
    """Compute rolling depths independently of the catalogue's strided sums."""
    windows = np.convolve(values, np.ones(width), mode='valid')
    maximum = float(windows.max())
    return maximum, np.flatnonzero(windows >= maximum - tolerance) + width - 1

def score_event(event, timestamps, observed, forecasts, protocol, cutoff):
    inside = (timestamps > event.event_start_utc) & (timestamps <= event.event_end_utc)
    tolerance = protocol['peak_depth_tolerance_mm']
    if (int(inside.sum()) != event.fine_intervals or np.any(observed[~inside] != 0)
            or not np.isclose(observed[inside].sum(), event.observed_depth_mm, atol=tolerance, rtol=0)):
        raise ValueError(f'Observed full-hour envelope disagrees with dry event boundaries for {event.event_id}')
    metadata = {col: getattr(event, col) for col in METHOD_META}
    score_rows, checks = [], []
    for minutes in protocol['peak_windows_minutes']:
        width = minutes // 5
        observed_peak, observed_ties = convolution_peak(observed, width, tolerance)
        catalogue_peak = getattr(event, f'peak_{minutes}min_mm')
        difference = observed_peak - catalogue_peak
        observed_end = timestamps[observed_ties[0]]
        checks.append({**metadata, 'window_minutes': minutes, 'catalogue_peak_mm': catalogue_peak,
                       'envelope_observed_peak_mm': observed_peak, 'signed_difference_mm': difference,
                       'tolerance_mm': tolerance, 'matches_within_tolerance': abs(difference) <= tolerance,
                       'catalogue_peak_end_utc': getattr(event, f'peak_{minutes}min_end_utc'),
                       'envelope_observed_peak_end_utc': observed_end})
        if minutes == 15 and (observed_peak >= cutoff - tolerance) != event.high_intensity:
            raise ValueError(f'15-minute intensity grouping disagrees for {event.event_id}')
        for method, predicted in forecasts.items():
            maximum, tied = convolution_peak(predicted, width, tolerance)
            miss = maximum <= tolerance
            predicted_end = pd.NaT if miss else timestamps[tied[0]]
            time_error = np.nan if miss else (predicted_end - observed_end) / pd.Timedelta(minutes=1)
            total = float(predicted.sum())
            outside = float(predicted[~inside].sum())
            zero_total = total == 0
            score_rows.append({
                **metadata, 'window_minutes': minutes, 'method': method,
                'covering_hours': event.covering_hours, 'five_minute_steps': len(timestamps),
                'observed_peak_mm': observed_peak, 'predicted_peak_mm': maximum,
                'signed_peak_depth_error_mm': maximum - observed_peak, 'abs_peak_depth_error_mm': abs(maximum - observed_peak),
                'observed_peak_end_utc': observed_end, 'predicted_peak_end_utc': predicted_end,
                'observed_peak_tie_count': len(observed_ties), 'predicted_peak_tie_count': len(tied),
                'observed_peak_crosses_hour': timestamps[observed_ties[0] - width + 1].ceil('h') != observed_end.ceil('h'),
                'signed_peak_time_error_min': time_error, 'abs_peak_time_error_min': abs(time_error),
                'missed_positive_peak': miss, 'envelope_predicted_total_mm': total,
                'outside_observed_event_mm': outside, 'outside_observed_event_fraction': np.nan if zero_total else outside / total,
                'outside_fraction_zero_denominator': zero_total,
            })
    series = pd.DataFrame({**metadata, 'time_end_utc': timestamps, 'inside_observed_event': inside,
                           'observed': observed, **forecasts})
    return score_rows, checks, series

def summarize(group, minimum_stations, minimum_events):
    events = len(group)
    stations = group.station_wban.nunique()
    timing = group.abs_peak_time_error_min.dropna()
    ratio = group.outside_observed_event_fraction.dropna()
    counts = {
        'events': events, 'stations': stations, 'timing_events': len(timing),
        'outside_fraction_events': len(ratio),
        'outside_fraction_zero_denominator_events': int(group.outside_fraction_zero_denominator.sum()),
        'missed_positive_peak_events': int(group.missed_positive_peak.sum()),
        'observed_tied_peak_events': int(group.observed_peak_tie_count.gt(1).sum()),
        'predicted_tied_peak_events': int(group.predicted_peak_tie_count.gt(1).sum()),
    }
    report = stations >= minimum_stations and events >= minimum_events
    metrics = {
        'peak_depth_mae_mm': group.abs_peak_depth_error_mm.mean(),
        'peak_depth_bias_mm': group.signed_peak_depth_error_mm.mean(),
        'peak_time_mae_min': timing.mean(),
        'peak_time_median_abs_min': timing.median(),
        'peak_time_p90_abs_min': timing.quantile(.90),
        'peak_time_bias_min': group.signed_peak_time_error_min.mean(),
        'observed_tied_peak_proportion': counts['observed_tied_peak_events'] / events,
        'predicted_tied_peak_proportion': counts['predicted_tied_peak_events'] / events,
        'mean_outside_observed_event_fraction': ratio.mean(),
    }
    return {**counts, 'report_errors': report, **{name: value if report else np.nan for name, value in metrics.items()}}

def metrics_tables(scores, protocol):
    main_rows, stratified = [], []
    settings = protocol['error_strata']
    for cohort_name, frame in [('all', scores), ('high_intensity', scores[scores.high_intensity])]:
        for (window, method), group in frame.groupby(['window_minutes', 'method']):
            main_rows.append({'cohort': cohort_name, 'window_minutes': window, 'method': method,
                              **summarize(group, 1, 1)})
        for field in ('network', 'kg_class', 'kg_group'):
            for (label, window, method), group in frame.groupby([field, 'window_minutes', 'method']):
                stratified.append({'cohort': cohort_name, 'stratification': field, 'level': label,
                                   'window_minutes': window, 'method': method,
                                   **summarize(group, settings['min_stations'], settings['min_events'])})
    return pd.DataFrame(main_rows), pd.DataFrame(stratified)
