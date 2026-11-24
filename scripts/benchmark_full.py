"""Fit and evaluate fixed offline models for hourly interval disaggregation."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]

VARIABLES = ['air_temperature', 'precipitation', 'solar_radiation', 'relative_humidity']

FEATURES = ['own_current', 'own_previous', 'own_next'] + ['current_' + v for v in VARIABLES] + [
    'utc_hour_sin', 'utc_hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'latitude', 'longitude']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def stable_seed(seed, *tokens):
    value = '|'.join([str(seed), *map(str, tokens)]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little')

def record_path(record, product):
    path = Path(record['outputs'][product]['path'])
    return path if path.is_absolute() else ROOT / path

def read_record(record, verify=True):
    paths = {key: record_path(record, key) for key in ['high_frequency', 'hourly_pairs']}
    if verify:
        for key, path in paths.items():
            if sha256(path) != record['outputs'][key]['sha256']:
                raise ValueError(f'Release hash mismatch: {path}')
    # ParquetFile avoids automatically treating directory partition names as columns.
    high = pq.ParquetFile(paths['high_frequency']).read(columns=['time_end_utc'] + VARIABLES).to_pandas()
    pairs = pq.ParquetFile(paths['hourly_pairs']).read(columns=[
        'time_end_utc', 'variable', 'derived_hourly', 'official_hourly', 'LATITUDE', 'LONGITUDE',
        'station_wban', 'station_split', 'source_year']).to_pandas()
    high['time_end_utc'] = pd.to_datetime(high.time_end_utc, utc=True)
    pairs['time_end_utc'] = pd.to_datetime(pairs.time_end_utc, utc=True)
    if high.time_end_utc.duplicated().any() or pairs.duplicated(['time_end_utc', 'variable']).any():
        raise ValueError(f'Duplicate release keys: {record["station"]}/{record["year"]}')
    if set(pairs.station_split.dropna().unique()) != {record['station_split']}:
        raise ValueError('Manifest/release station split mismatch')
    return high.set_index('time_end_utc'), pairs

def hourly_targets(high):
    """Exact index lookup: no interpolation, padding, fill, or implicit reshape."""
    hours = high.index.ceil('h').unique().sort_values()
    # pandas 3 may retain microsecond input units; normalize before ns arithmetic.
    ends = hours.as_unit('ns').asi8[:, None] - (np.arange(11, -1, -1) * 5 * 60 * 10**9)[None, :]
    targets = {}
    lookup = pd.to_datetime(ends.ravel(), utc=True)
    for variable in VARIABLES:
        targets[variable] = high[variable].reindex(lookup).to_numpy(dtype=float).reshape(-1, 12)
    return hours, targets

def input_features(pairs, hours, track, variable):
    coarse = pairs.pivot(index='time_end_utc', columns='variable', values=track).reindex(columns=VARIABLES)
    own = coarse[variable]
    current = own.reindex(hours).to_numpy(dtype=float)
    previous = own.reindex(hours - pd.Timedelta(hours=1)).to_numpy(dtype=float)
    following = own.reindex(hours + pd.Timedelta(hours=1)).to_numpy(dtype=float)
    # Use coarse interval midpoints for calendar phase, always UTC.
    middle = hours - pd.Timedelta(minutes=30)
    phase_hour = 2 * np.pi * (middle.hour.to_numpy() + middle.minute.to_numpy() / 60) / 24
    phase_day = 2 * np.pi * (middle.dayofyear.to_numpy() - 1) / 365.2425
    location = pairs.drop_duplicates('time_end_utc').set_index('time_end_utc')
    latitude = location.LATITUDE.reindex(hours).to_numpy(dtype=float)
    longitude = location.LONGITUDE.reindex(hours).to_numpy(dtype=float)
    data = np.column_stack([current, previous, following, coarse.reindex(hours).to_numpy(dtype=float),
                            np.sin(phase_hour), np.cos(phase_hour), np.sin(phase_day), np.cos(phase_day),
                            latitude, longitude])
    assert data.shape[1] == len(FEATURES)
    return data

def eligible_samples(high, pairs, track, variable, hours=None, target_map=None):
    if hours is None:
        hours, target_map = hourly_targets(high)
    y = target_map[variable]
    x = input_features(pairs, hours, track, variable)
    complete = np.isfinite(y).all(axis=1)
    own_good = np.isfinite(x[:, :3]).all(axis=1)
    feasible = np.ones(len(hours), dtype=bool)
    if variable in ('precipitation', 'solar_radiation'):
        feasible &= x[:, 0] >= 0
    elif variable == 'relative_humidity':
        feasible &= (x[:, 0] >= 0) & (x[:, 0] <= 100)
    keep = complete & own_good & feasible
    counts = {'fine_hours': len(hours), 'complete_target_hours': int(complete.sum()),
              'complete_target_missing_own_coarse_support': int((complete & ~own_good).sum()),
              'complete_target_infeasible_current_coarse': int((complete & own_good & ~feasible).sum()),
              'eligible_hours': int(keep.sum())}
    return hours[keep], x[keep], y[keep], counts

def fit_medians(x):
    """Only the selected training sample reaches this function."""
    finite = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(np.where(finite, x, np.nan), axis=0)
    empty = ~np.isfinite(medians)
    medians[empty] = 0
    return medians, [FEATURES[i] for i in np.flatnonzero(empty)]

def transform_features(x, medians):
    missing = ~np.isfinite(x)
    return np.column_stack([np.where(missing, medians[None, :], x), missing.astype(float)]).astype(np.float32)

def project_mean(values, coarse, variable):
    """Euclidean box/sum projection; preserves the intentional coarse condition."""
    values = np.asarray(values, dtype=float)
    coarse = np.asarray(coarse, dtype=float)
    total = coarse if variable == 'precipitation' else 12 * coarse
    if variable == 'air_temperature':
        return values + (coarse - values.mean(axis=1))[:, None]
    upper = 100.0 if variable == 'relative_humidity' else np.inf
    lo = -np.max(values, axis=1) - 1
    hi = total - np.min(values, axis=1) + 1
    for _ in range(55):
        shift = (lo + hi) / 2
        mass = np.clip(values + shift[:, None], 0, upper).sum(axis=1)
        lo = np.where(mass < total, shift, lo)
        hi = np.where(mass >= total, shift, hi)
    return np.clip(values + ((lo + hi) / 2)[:, None], 0, upper)

def linear_predictions(x, variable):
    """Adjacent-hour means at centres; fine interval centres are (j+.5)/12."""
    own = x[:, :3] / (12 if variable == 'precipitation' else 1)
    current, previous, following = own.T
    position = (np.arange(12) + .5) / 12
    values = np.empty((len(x), 12), dtype=float)
    left = position < .5
    values[:, left] = previous[:, None] + (position[left] + .5)[None, :] * (current - previous)[:, None]
    values[:, ~left] = current[:, None] + (position[~left] - .5)[None, :] * (following - current)[:, None]
    return values + (current - values.mean(axis=1))[:, None]

def model_predictions(bundle, x, variable):
    output = bundle['model'].predict(transform_features(x, bundle['feature_medians']))
    if variable == 'precipitation':
        fractions = np.maximum(output, 0)
        mass = fractions.sum(axis=1)
        fractions = np.divide(fractions, mass[:, None], out=np.full_like(fractions, 1 / 12), where=mass[:, None] > 0)
        return fractions * x[:, 0, None]
    return project_mean(output + x[:, 0, None], x[:, 0], variable)

def all_predictions(bundle, x, variable):
    uniform = np.repeat((x[:, 0] / (12 if variable == 'precipitation' else 1))[:, None], 12, axis=1)
    output = {'uniform_within_hour': uniform}
    linear = linear_predictions(x, variable)
    if variable != 'precipitation':
        output['linear_mean_constrained'] = linear
    if variable != 'air_temperature':
        output['linear_bounded_mean_constrained'] = project_mean(linear, x[:, 0], variable)
    output['extra_trees_constrained'] = model_predictions(bundle, x, variable)
    return output

def strata(variable, coarse, config):
    masks = {'all': np.ones(len(coarse), dtype=bool)}
    if variable == 'solar_radiation':
        masks['illuminated_coarse_gt20'] = coarse > config['solar_illuminated_threshold_W_m2']
    if variable == 'precipitation':
        masks['wet_coarse_gt0'] = coarse > config['rain_wet_threshold_mm']
        masks['rain_coarse_ge1mm'] = coarse >= config['rain_heavy_threshold_mm']
    return masks

def per_hour_metrics(y, predicted, coarse, variable, tolerance):
    error = predicted - y
    closure = predicted.sum(axis=1) if variable == 'precipitation' else predicted.mean(axis=1)
    bounded = variable in ('precipitation', 'solar_radiation', 'relative_humidity')
    invalid = predicted < -tolerance if bounded else np.zeros_like(predicted, dtype=bool)
    if variable == 'relative_humidity':
        invalid |= predicted > 100 + tolerance
    return {'n': np.full(len(y), 12, dtype=np.int32), 'sum_absolute_error': np.abs(error).sum(axis=1),
            'sum_squared_error': (error ** 2).sum(axis=1),
            'sum_peak_absolute_error': np.abs(predicted.max(axis=1) - y.max(axis=1)),
            'sum_ramp_absolute_error': np.abs(np.diff(predicted, axis=1) - np.diff(y, axis=1)).sum(axis=1),
            'ramp_n': np.full(len(y), 11, dtype=np.int32),
            'closure_max': np.abs(closure - coarse), 'bound_invalid_count': invalid.sum(axis=1)}

def sum_metrics(values, mask):
    result = {key: float(array[mask].sum()) for key, array in values.items() if key != 'closure_max'}
    result['hours'] = int(mask.sum())
    result['closure_max'] = float(values['closure_max'][mask].max()) if mask.any() else 0.0
    return result

def add_ratios(frame):
    frame = frame.copy()
    frame['mae'] = frame.sum_absolute_error / frame.n.replace(0, np.nan)
    frame['rmse'] = np.sqrt(frame.sum_squared_error / frame.n.replace(0, np.nan))
    frame['hourly_peak_mae'] = frame.sum_peak_absolute_error / frame.hours.replace(0, np.nan)
    frame['within_hour_ramp_mae'] = frame.sum_ramp_absolute_error / frame.ramp_n.replace(0, np.nan)
    return frame

def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config['feature_names'] != FEATURES or config['variables'] != VARIABLES:
        raise ValueError('Feature whitelist or variable schema differs from this implementation')
    if config['model']['class'] != 'ExtraTreesRegressor':
        raise ValueError('Only the fixed ExtraTrees model is implemented')
    if set(config['train_years']) & set(config['evaluation_years']):
        raise ValueError('Training/evaluation years overlap')
    return config

def cap_training_pool(x, y, row_wbans, cap, seed):
    """Keep row identity aligned through the final pooled random selection."""
    if len(x) != len(y) or len(x) != len(row_wbans):
        raise ValueError('Training features, targets and station identities are misaligned')
    if cap < 1:
        raise ValueError('Training cap must be positive')
    if len(x) > cap:
        chosen = np.sort(np.random.default_rng(seed).choice(len(x), cap, replace=False))
        return x[chosen], y[chosen], row_wbans[chosen]
    return x, y, row_wbans

def fit_models(records, config, output_dir, *, reader=None, sample_selector=None):
    reader = reader or read_record
    sample_selector = sample_selector or eligible_samples
    keys = [(variable, track) for variable in VARIABLES for track in config['tracks']]
    samples = {key: [] for key in keys}
    sample_records = []
    training = [r for r in records if r['year'] in config['train_years'] and r['station_split'] == config['training_station_split']]
    if not training:
        raise ValueError('No training-station records in the configured fitting years')
    for number, record in enumerate(training, 1):
        high, pairs = reader(record)
        hours, target_map = hourly_targets(high)
        for variable, track in keys:
            ends, x, y, counts = sample_selector(high, pairs, track, variable, hours, target_map)
            if variable in config.get('training_wet_variables', ['precipitation']):
                wet = x[:, FEATURES.index('current_precipitation')] > 0
                if variable == 'precipitation':
                    wet &= y.sum(axis=1) > 0
                ends, x, y = ends[wet], x[wet], y[wet]
            rng = np.random.default_rng(stable_seed(config['random_seed'], record['station_wban'], record['year'], variable, track))
            chosen = np.sort(rng.choice(len(x), min(len(x), config['training_cap_per_station_year']), replace=False))
            selected_x, selected_y = x[chosen].astype(np.float32), y[chosen]
            if variable == 'precipitation':
                selected_y = selected_y / selected_y.sum(axis=1, keepdims=True)
            else:
                selected_y = selected_y - selected_x[:, 0, None]
            if len(chosen):
                row_wbans = np.full(len(chosen), str(record['station_wban']).zfill(5), dtype='U5')
                samples[(variable, track)].append((selected_x, selected_y.astype(np.float32), row_wbans))
            sample_records.append({'station': record['station'], 'station_wban': record['station_wban'],
                                   'year': record['year'], 'station_split': record['station_split'],
                                   'variable': variable, 'track': track, **counts,
                                   'available_training_hours_after_wet_selection': len(x), 'sampled_hours': len(chosen),
                                   'selected_hour_endpoints_sha256': hashlib.sha256(ends[chosen].asi8.tobytes()).hexdigest()})
        if number % 10 == 0:
            print(json.dumps({'phase': 'sample_training', 'completed': number, 'total': len(training)}), flush=True)
    model_metadata = []
    for variable, track in keys:
        arrays = samples.pop((variable, track))
        if not arrays:
            raise ValueError(f'No usable training targets for {variable}/{track}')
        x = np.concatenate([a[0] for a in arrays])
        y = np.concatenate([a[1] for a in arrays])
        row_wbans = np.concatenate([a[2] for a in arrays])
        before_cap = len(x)
        x, y, row_wbans = cap_training_pool(x, y, row_wbans, config['training_cap_pooled'],
                                          stable_seed(config['random_seed'], 'pooled', variable, track))
        wban_values, wban_counts = np.unique(row_wbans, return_counts=True)
        training_wbans = wban_values.tolist()
        training_wban_row_counts = dict(zip(training_wbans, map(int, wban_counts)))
        medians, empty = fit_medians(x)
        params = {key: value for key, value in config['model'].items() if key != 'class'}
        model = ExtraTreesRegressor(**params)
        start = time.monotonic()
        model.fit(transform_features(x, medians), y)
        filename = output_dir / 'models' / f'{variable}__{track}.joblib'
        filename.parent.mkdir(parents=True, exist_ok=True)
        bundle = {'model': model, 'feature_medians': medians, 'feature_names': FEATURES,
                  'expanded_feature_names': FEATURES + [f'{f}_missing' for f in FEATURES],
                  'variable': variable, 'track': track, 'settings': config,
                  'training_rows': len(x), 'all_missing_training_features': empty,
                  'training_wbans': training_wbans, 'training_wban_row_counts': training_wban_row_counts}
        joblib.dump(bundle, filename, compress=3)
        metadata = {'variable': variable, 'track': track, 'training_rows': len(x),
                    'training_rows_before_pooled_cap': before_cap, 'fit_seconds': time.monotonic() - start,
                    'feature_medians': medians.tolist(), 'all_missing_training_features': empty,
                    'training_wbans': training_wbans, 'training_wban_row_counts': training_wban_row_counts,
                    'path': str(filename.resolve()), 'sha256': sha256(filename)}
        model_metadata.append(metadata)
        print(json.dumps({'phase': 'fit', **metadata}), flush=True)
    pd.DataFrame(sample_records).to_csv(output_dir / 'training_sample_coverage.csv', index=False)
    save_json(output_dir / 'model_metadata.json', model_metadata)
    return model_metadata
