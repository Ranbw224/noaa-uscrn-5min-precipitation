"""Attach the frozen 2007–2025 diagnostics to observed precipitation events."""

from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmark_full import record_path


DEPTHS = (5, 10, 20, 50, 100)
SOIL_SOURCE_FIELDS = tuple(f'SOIL_MOISTURE_{depth}' for depth in DEPTHS)
SOIL_VALUE_FIELDS = tuple(f'soil_moisture_pre_{depth}cm' for depth in DEPTHS)
SOIL_REASON_FIELDS = tuple(f'soil_moisture_missing_reason_{depth}cm' for depth in DEPTHS)
SOIL_OFFSET = 'soil_moisture_pre_event_offset_min'
VALID = 'valid'
BEFORE_DEPLOYMENT = 'no_sensor_prior_to_deployment'
AFTER_DEPLOYMENT = 'transient_missing_after_deployment'
FINE_FIELDS = (
    'air_temperature', 'air_temperature_exclusion_code',
    'relative_humidity', 'relative_humidity_exclusion_code', 'WIND_1_5', 'WIND_FLAG',
)
TEMPERATURE_FIELDS = ('event_temp_min_c', 'event_temp_mean_c', 'event_temp_max_c')
NUMERIC_DIAGNOSTICS = (*TEMPERATURE_FIELDS, 'event_wetbulb_mean_c', *SOIL_VALUE_FIELDS,
                       SOIL_OFFSET, 'event_wind_mean_ms')
STRING_DIAGNOSTICS = ('precip_phase_flag', *SOIL_REASON_FIELDS)
FIVE_MINUTES = pd.Timedelta(minutes=5)
MINUTE = pd.Timedelta(minutes=1)
NANOSECOND = pd.Timedelta(nanoseconds=1)


def soil_candidate(start):
    """Return the single UTC hourly endpoint, its archive year, and the lag."""
    start = pd.Timestamp(start)
    if pd.isna(start) or start.tzinfo is None:
        raise ValueError('Event start requires a timezone-aware timestamp')
    start = start.tz_convert('UTC')
    if start != start.floor('5min'):
        raise ValueError(f'Event start is outside the five-minute grid: {start}')
    candidate = (start - FIVE_MINUTES).floor('h')
    offset = (start - candidate) / MINUTE
    if not 5 <= offset <= 60:
        raise ValueError(f'Candidate is outside the frozen pre-event window: {candidate}/{start}')
    return candidate, (candidate - NANOSECOND).year, float(offset)


def wet_bulb_temperature(temperature, humidity):
    """Stull (2011), evaluated at sea-level pressure in the frozen domain."""
    temperature = np.asarray(temperature, dtype=float)
    humidity = np.asarray(humidity, dtype=float)
    eligible = ((temperature >= -20) & (temperature <= 50)
                & (humidity >= 5) & (humidity <= 100))
    output = np.full_like(temperature, np.nan)
    saturated = eligible & (humidity == 100)
    output[saturated] = temperature[saturated]
    estimated = eligible & ~saturated
    t, rh = temperature[estimated], humidity[estimated]
    output[estimated] = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
        - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return output


def fine_diagnostics(span):
    temperature = span.air_temperature.where(span.air_temperature_exclusion_code.eq(0))
    temperature = temperature.where(np.isfinite(temperature))
    humidity = span.relative_humidity.where(span.relative_humidity_exclusion_code.eq(0))
    t_min, t_mean, t_max = temperature.min(), temperature.mean(), temperature.max()
    if pd.isna(t_min):
        phase = 'temp_missing'
    elif t_min > 2:
        phase = 'liquid_likely'
    elif t_max < 0:
        phase = 'frozen_likely'
    else:
        phase = 'mixed_or_transitional'
    wet_bulb = wet_bulb_temperature(temperature, humidity)
    wind = span.WIND_1_5
    wind = wind.where(np.isfinite(wind) & wind.ge(0) & wind.ne(-99) & span.WIND_FLAG.eq(0))
    return {
        'event_temp_min_c': t_min, 'event_temp_mean_c': t_mean, 'event_temp_max_c': t_max,
        'precip_phase_flag': phase, 'event_wetbulb_mean_c': pd.Series(wet_bulb).mean(),
        'event_wind_mean_ms': wind.mean(),
    }


def soil_diagnostics(start, hourly_by_year, first_valid):
    candidate, year, offset = soil_candidate(start)
    hourly = hourly_by_year.get(year)
    row = hourly.loc[candidate] if hourly is not None and candidate in hourly.index else None
    values, reasons = [], []
    for depth, field in zip(DEPTHS, SOIL_SOURCE_FIELDS, strict=True):
        available_since = first_valid[depth]
        value = row[field] if row is not None else np.nan
        if pd.notna(available_since) and available_since <= candidate and pd.notna(value) and 0 <= value <= 1:
            values.append(float(value))
            reasons.append(VALID)
        else:
            values.append(np.nan)
            reasons.append(BEFORE_DEPLOYMENT if pd.isna(available_since) or start < available_since
                           else AFTER_DEPLOYMENT)
    return {**dict(zip(SOIL_VALUE_FIELDS, values, strict=True)),
            **dict(zip(SOIL_REASON_FIELDS, reasons, strict=True)),
            SOIL_OFFSET: offset if VALID in reasons else np.nan}


def read_partition(record, product, columns):
    frame = pq.ParquetFile(record_path(record, product)).read(
        columns=['time_end_utc', 'station_wban', 'source_year', *columns]).to_pandas()
    if frame.empty or not frame.station_wban.eq(record['station_wban']).all() or (
            not frame.source_year.eq(record['year']).all()):
        raise ValueError(f'Unexpected {product} partition identity: {record["station_wban"]}/{record["year"]}')
    index = pd.DatetimeIndex(frame.time_end_utc)
    lower = pd.Timestamp(year=record['year'], month=1, day=1, tz='UTC')
    upper = lower + pd.DateOffset(years=1)
    if index.tz is None or not index.is_unique or not index.is_monotonic_increasing or (
            not ((index > lower) & (index <= upper)).all()):
        raise ValueError(f'Invalid {product} right endpoints: {record["station_wban"]}/{record["year"]}')
    frame.index = index.tz_convert('UTC')
    step = 'h' if product == 'hourly_hydrology' else '5min'
    if not frame.index.equals(frame.index.floor(step)):
        raise ValueError(f'Invalid {product} endpoint cadence: {record["station_wban"]}/{record["year"]}')
    return frame.loc[:, list(columns)]


def read_station_diagnostics(records):
    fine, hourly = {}, {}
    first_valid = dict.fromkeys(DEPTHS, pd.NaT)
    for record in sorted(records, key=lambda row: row['year']):
        year = record['year']
        if year in fine:
            raise ValueError(f'Duplicate source-year input for {record["station_wban"]}/{year}')
        fine[year] = read_partition(record, 'high_frequency', FINE_FIELDS)
        soil = read_partition(record, 'hourly_hydrology', SOIL_SOURCE_FIELDS)
        soil = soil.where(np.isfinite(soil) & soil.ne(-99) & soil.ge(0) & soil.le(1))
        hourly[year] = soil
        for depth, field in zip(DEPTHS, SOIL_SOURCE_FIELDS, strict=True):
            if pd.isna(first_valid[depth]):
                first_valid[depth] = soil[field].first_valid_index()
    return fine, hourly, first_valid


def add_event_diagnostics(events, base_records):
    """Return the augmented catalogue and the sensor-start evidence for assertion 5."""
    by_wban = defaultdict(list)
    for record in base_records:
        by_wban[record['station_wban']].append(record)
    result = events.reset_index(drop=True).copy()
    confirmed = result[result.usability_tier.isin([1, 2])]
    numerics = {name: np.full(len(result), np.nan) for name in NUMERIC_DIAGNOSTICS}
    strings = {name: [None] * len(result) for name in STRING_DIAGNOSTICS}
    availability = {}
    for wban, group in confirmed.groupby('station_wban', sort=False):
        fine, hourly, first_valid = read_station_diagnostics(by_wban[wban])
        availability[wban] = first_valid
        for event in group.itertuples():
            start = pd.Timestamp(event.event_start_utc).tz_convert('UTC')
            end = pd.Timestamp(event.event_end_utc).tz_convert('UTC')
            if end <= start:
                raise ValueError(f'Invalid observed interval for {event.event_id}')
            parts = []
            for year in range(start.year, (end - NANOSECOND).year + 1):
                frame = fine[year]
                left, right = frame.index.searchsorted([start, end], side='right')
                parts.append(frame.iloc[left:right])
            span = parts[0] if len(parts) == 1 else pd.concat(parts)
            values = {**fine_diagnostics(span), **soil_diagnostics(start, hourly, first_valid)}
            for name, value in values.items():
                (numerics if name in numerics else strings)[name][event.Index] = value
    for name, values in numerics.items():
        result[name] = values
    for name, values in strings.items():
        result[name] = pd.array(values, dtype='string')
    return result, availability
