"""Methodological checks using synthetic arrays, never real test-set outcomes."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_full as benchmark


class BenchmarkFullTests(unittest.TestCase):
    def test_linear_interpolates_interval_centres_and_preserves_current_mean(self):
        x = np.zeros((2, len(benchmark.FEATURES)))
        x[:, :3] = [[10, 8, 12], [-10, -12, -8]]
        position = (np.arange(12) + .5) / 12
        expected = np.vstack([10 + 2 * (position - .5), -10 + 2 * (position - .5)])
        actual = benchmark.linear_predictions(x, 'air_temperature')
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.mean(axis=1), x[:, 0], rtol=0, atol=1e-12)

    def test_projection_preserves_closure_and_physical_bounds(self):
        rng = np.random.default_rng(19)
        raw = rng.normal(0, 200, (40, 12))
        for variable, upper in [('air_temperature', 40), ('precipitation', 20),
                                ('solar_radiation', 500), ('relative_humidity', 100)]:
            with self.subTest(variable=variable):
                coarse = rng.uniform(0, upper, 40)
                coarse[0] = 0
                if variable == 'relative_humidity':
                    coarse[1] = 100
                result = benchmark.project_mean(raw, coarse, variable)
                aggregate = result.sum(axis=1) if variable == 'precipitation' else result.mean(axis=1)
                np.testing.assert_allclose(aggregate, coarse, rtol=0, atol=1e-10)
                if variable != 'air_temperature':
                    self.assertGreaterEqual(result.min(), 0)
                if variable == 'relative_humidity':
                    self.assertLessEqual(result.max(), 100)

    def test_target_alignment_handles_microsecond_and_nanosecond_datetime_units(self):
        # Cross leap-day midnight. pandas 3 defaults often use us, not ns.
        index = pd.date_range('2024-02-28T23:05Z', periods=36, freq='5min')
        expected = np.arange(36, dtype=float).reshape(3, 12)
        for unit in ['us', 'ns']:
            with self.subTest(unit=unit):
                high = pd.DataFrame({variable: np.arange(36, dtype=float)
                                     for variable in benchmark.VARIABLES}, index=index.as_unit(unit))
                ends, targets = benchmark.hourly_targets(high)
                self.assertEqual(list(ends.strftime('%Y-%m-%d %H:%M')),
                                 ['2024-02-29 00:00', '2024-02-29 01:00', '2024-02-29 02:00'])
                for variable in benchmark.VARIABLES:
                    np.testing.assert_array_equal(targets[variable], expected)

    def test_target_derived_columns_cannot_change_whitelisted_features(self):
        hours = pd.date_range('2024-01-01T01:00Z', periods=4, freq='h')
        rows = []
        for end in hours:
            for variable in benchmark.VARIABLES:
                rows.append({'time_end_utc': end, 'variable': variable,
                             'derived_hourly': 10.0, 'official_hourly': 11.0,
                             'LATITUDE': 40.0, 'LONGITUDE': -100.0,
                             'five_minute_min': -1000, 'five_minute_max': 9000,
                             'rain_max_fraction': .9, 'valid_five_minute_count': 12,
                             'discrepancy_derived_minus_official': 123})
        pairs = pd.DataFrame(rows)
        first = benchmark.input_features(pairs, hours, 'derived_hourly', 'air_temperature')
        for field in ['five_minute_min', 'five_minute_max', 'rain_max_fraction',
                      'valid_five_minute_count', 'discrepancy_derived_minus_official']:
            pairs[field] = np.arange(len(pairs)) * 987654
        second = benchmark.input_features(pairs, hours, 'derived_hourly', 'air_temperature')
        np.testing.assert_allclose(first, second, equal_nan=True)
        self.assertEqual(first.shape[1], len(benchmark.FEATURES))
        self.assertNotIn('five_minute_max', benchmark.FEATURES)

    def test_pooled_cap_keeps_actual_wban_labels_aligned(self):
        # With cap one, a nominal training station necessarily disappears.
        x = np.arange(12, dtype=float).reshape(4, 3)
        y = x[:, :1] * 5
        wbans = np.array(['11111', '11111', '22222', '22222'])
        actual_x, actual_y, actual_wbans = benchmark.cap_training_pool(x, y, wbans, cap=1, seed=71)
        self.assertEqual(len(set(actual_wbans)), 1)
        row = int(actual_x[0, 0] / 3)
        np.testing.assert_array_equal(actual_y[0], y[row])
        self.assertEqual(actual_wbans[0], wbans[row])
        self.assertNotEqual(set(actual_wbans), set(wbans))

    def test_imputation_medians_and_missing_indicators_use_fitted_sample(self):
        train = np.tile(np.arange(len(benchmark.FEATURES), dtype=float), (2, 1))
        train[1] += 2
        train[:, 4] = np.nan
        medians, empty = benchmark.fit_medians(train)
        self.assertEqual(medians[0], 1)
        self.assertEqual(medians[4], 0)
        self.assertIn(benchmark.FEATURES[4], empty)
        held_out = np.full((1, len(benchmark.FEATURES)), np.nan)
        result = benchmark.transform_features(held_out, medians)
        np.testing.assert_array_equal(result[0, :len(medians)], medians)
        np.testing.assert_array_equal(result[0, len(medians):], np.ones(len(medians)))

    def test_fitted_bundles_preserve_actual_post_cap_wbans_per_variable(self):
        config = benchmark.load_config(ROOT / 'config/benchmark_full.json')
        config['training_cap_pooled'] = 1
        records, fixtures = [], {}
        for station, wban in [('SYNTHETIC_A', '11111'), ('SYNTHETIC_B', '22222')]:
            index = pd.date_range('2013-01-01T00:05Z', periods=72, freq='5min')
            high = pd.DataFrame({v: np.ones(len(index)) for v in benchmark.VARIABLES}, index=index)
            if station == 'SYNTHETIC_B':
                high['relative_humidity'] = np.nan
            pairs = []
            for end in index.ceil('h').unique():
                for variable in benchmark.VARIABLES:
                    coarse = 12. if variable == 'precipitation' else 1.
                    pairs.append({'time_end_utc': end, 'variable': variable, 'LATITUDE': 40.,
                                  'LONGITUDE': -100., 'derived_hourly': coarse, 'official_hourly': coarse})
            fixtures[station] = (high, pd.DataFrame(pairs))
            records.append({'station': station, 'station_wban': wban, 'station_split': 'train', 'year': 2013})
        # A non-training record must never reach read_record or median fitting.
        records.append({'station': 'SYNTHETIC_HELD_OUT', 'station_wban': '33333',
                        'station_split': 'test', 'year': 2025})
        with tempfile.TemporaryDirectory(prefix='benchmark_SYNTHETIC_TEST_') as temp:
            with mock.patch.object(benchmark, 'read_record', side_effect=lambda record: fixtures[record['station']]):
                with contextlib.redirect_stdout(io.StringIO()):
                    metadata = benchmark.fit_models(records, config, Path(temp))
            self.assertEqual(len(metadata), 8)
            for model_info in metadata:
                self.assertEqual(model_info['training_rows'], 1)
                self.assertEqual(len(model_info['training_wbans']), 1)
                self.assertEqual(sum(model_info['training_wban_row_counts'].values()), 1)
                self.assertNotIn('33333', model_info['training_wbans'])
                if model_info['variable'] == 'relative_humidity':
                    self.assertEqual(model_info['training_wbans'], ['11111'])
                bundle = joblib.load(model_info['path'])
                self.assertEqual(bundle['training_wbans'], model_info['training_wbans'])
                self.assertEqual(bundle['training_wban_row_counts'], model_info['training_wban_row_counts'])


if __name__ == '__main__':
    unittest.main()
