"""Join archived hyetographs, estimate local P95s or summarise soil-moisture reason codes."""

import argparse
from pathlib import Path

import pandas as pd


def load_profile(root, profile='front_loaded'):
    keys = ['event_id', 'station_wban']
    events = pd.read_parquet(root / 'data/events.parquet',
                             columns=[*keys, 'usability_tier', 'storm_profile_tercile'])
    events = events.loc[events.usability_tier.eq(1)
                        & events.storm_profile_tercile.eq(profile)]
    series = pd.read_parquet(root / 'data/envelope_timeseries.parquet')
    sites = pd.read_csv(root / 'data/station_metadata.csv',
                        dtype={'station_wban': str})
    selected = series.merge(
        events[[*keys, 'storm_profile_tercile']], on=keys, validate='many_to_one')
    return selected.merge(
        sites[['station_wban', 'elevation_m', 'dataset_split']],
        on='station_wban', validate='many_to_one')


def local_p95(root, key='station_wban'):
    peaks = pd.read_parquet(root / 'data/events.parquet',
                            columns=['p95_reference_event', key, 'peak_15min_mm'])
    reference = peaks.loc[peaks.p95_reference_event.eq(True)]
    return reference.groupby(key, dropna=False).peak_15min_mm.agg(
        events='count', p95_mm=lambda values: values.quantile(.95))


def warm_season_soil_reasons(events, warm_months):
    keep = (events.usability_tier.isin((1, 2))
            & events.precip_phase_flag.eq('liquid_likely')
            & events.high_intensity.eq(True)
            & events.event_start_utc.dt.month.isin(warm_months))
    selected = events.loc[keep]
    reasons = selected.filter(like='soil_moisture_missing_reason_')
    return selected, reasons.apply(pd.Series.value_counts).fillna(0).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--profile', choices=('front_loaded', 'mid_loaded', 'back_loaded'), default='front_loaded')
    mode.add_argument('--local-p95', choices=('station_wban', 'kg_group'))
    mode.add_argument('--warm-months', nargs='+', type=int, choices=range(1, 13))
    args = parser.parse_args()
    if args.local_p95:
        print(local_p95(args.release, args.local_p95).to_string())
        return
    if args.warm_months:
        reason_columns = [f'soil_moisture_missing_reason_{depth}cm' for depth in (5, 10, 20, 50, 100)]
        fields = ['station_wban', 'event_start_utc', 'usability_tier', 'precip_phase_flag', 'high_intensity']
        events = pd.read_parquet(args.release / 'data/events.parquet', columns=[*fields, *reason_columns])
        selected, summary = warm_season_soil_reasons(events, args.warm_months)
        print(f'{len(selected)} events at {selected.station_wban.nunique()} WBANs')
        print(summary.to_string())
        return
    result = load_profile(args.release, args.profile)
    print(f'{len(result)} steps, {result.event_id.nunique()} events, {result.station_wban.nunique()} WBANs')
    print(result.groupby('kg_group', dropna=False).event_id.nunique().to_string())


if __name__ == '__main__':
    main()
