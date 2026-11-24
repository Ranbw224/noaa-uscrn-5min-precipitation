"""Regression checks for the frozen fitting and partition entry points."""

import contextlib
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from benchmark_full import VARIABLES, fit_models, load_config
from build_precipitation_event_library import read_station_year
from build_uscrn_release import read_json


def test_wet_hour_fitting_applies_to_all_configured_variables(tmp_path):
    config = load_config(ROOT / 'config/benchmark_full.json')
    config['train_years'] = [2013]
    config['evaluation_years'] = [2021]
    config['training_wet_variables'] = VARIABLES
    stamps = pd.date_range('2013-01-01T00:05Z', periods=72, freq='5min')
    high = pd.DataFrame({variable: np.ones(len(stamps)) for variable in VARIABLES}, index=stamps)
    hours = stamps.ceil('h').unique()
    wet = np.array([False, False, True, True, False, False])
    high['precipitation'] = np.repeat(wet.astype(float), 12)
    rows = []
    for i, end in enumerate(hours):
        for variable in VARIABLES:
            coarse = float(wet[i]) * 12 if variable == 'precipitation' else 1.0
            rows.append({'time_end_utc': end, 'variable': variable, 'LATITUDE': 34.66,
                         'LONGITUDE': -87.35, 'official_hourly': coarse, 'derived_hourly': coarse})
    pairs = pd.DataFrame(rows)
    records = [{'station': 'SYNTHETIC', 'station_wban': '63868', 'station_split': 'train', 'year': 2013}]
    with contextlib.redirect_stdout(io.StringIO()):
        models = fit_models(records, config, tmp_path, reader=lambda _record: (high, pairs))
    assert len(models) == 8
    assert all(model['training_rows'] == 2 for model in models)


BASE_MANIFEST = ROOT / 'reports/uscrn_2007_2025/base/full_build_manifest.json'


@pytest.mark.skipif(not BASE_MANIFEST.is_file(), reason='Requires local base data')
def test_in_memory_event_reader_matches_base_partitions():
    import pyarrow.parquet as pq

    records = [record for record in read_json(BASE_MANIFEST)['records']
               if record['year'] == 2013 and record['station_wban'] in {'63868', '53131'}]
    assert len(records) == 2
    for record in records:
        fine = pq.ParquetFile(ROOT / record['outputs']['high_frequency']['path']).read().to_pandas()
        hourly = pq.ParquetFile(ROOT / record['outputs']['hourly_pairs']['path']).read().to_pandas()
        original = read_station_year(record, None)
        supplied = read_station_year(record, None, fine_frame=fine, hourly_frame=hourly)
        for field in ('amounts', 'quality', 'hour_complete', 'hour_context', 'hour_current', 'eligible'):
            np.testing.assert_array_equal(getattr(original, field), getattr(supplied, field))
        assert original.counts == supplied.counts
