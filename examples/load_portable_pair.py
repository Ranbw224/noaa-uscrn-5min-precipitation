"""Inspect an indexed pair and the matching five-minute observations."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def open_hive_layer(release, layer):
    year_type = {'pairs': pa.int16(), 'annotations': pa.int64()}[layer]
    keys = pa.schema([('variable', pa.large_string()), ('source_year', year_type)])
    return ds.dataset(release / 'data' / layer, format='parquet',
                      partitioning=ds.partitioning(keys, flavor='hive'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--wban', default='04990')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--hour-end', default='2025-06-03T08:00Z')
    parser.add_argument('--hive', action='store_true', help='Open the pair layer with explicit Hive key types')
    args = parser.parse_args()
    end = pd.to_datetime(args.hour_end, utc=True)
    index = json.loads((args.release / 'data/portable_index.json').read_bytes())['records']
    pair = next(row['outputs']['pairs'] for row in index if row.get('source_year') == args.year
                and row.get('variable') == 'precipitation')
    fine = next(row['outputs']['high_frequency'] for row in index
                if row.get('source_year') == args.year and row.get('station_wban') == args.wban)
    data = open_hive_layer(args.release, 'pairs') if args.hive else ds.dataset(
        args.release / pair['path'], format='parquet')
    target = data.to_table(columns=['official_hourly', 'derived_hourly', 'is_valid_target'],
                           filter=(ds.field('station_wban') == args.wban)
                           & (ds.field('source_year') == args.year)
                           & (ds.field('variable') == 'precipitation')
                           & (ds.field('time_end_utc') == end.to_pydatetime())).to_pandas()
    if len(target) != 1 or not target.is_valid_target.iloc[0]:
        raise ValueError('The requested hour does not have a complete precipitation target')
    observed = pq.ParquetFile(args.release / fine['path']).read(
        columns=['time_end_utc', 'precipitation']).to_pandas()
    depths = observed.loc[(observed.time_end_utc > end - pd.Timedelta(hours=1))
                          & (observed.time_end_utc <= end), 'precipitation']
    expected = target.derived_hourly.iloc[0]
    if len(depths) != 12 or depths.isna().any() or abs(depths.sum() - expected) > 1e-10:
        raise ValueError('The indexed observations disagree with the exact aggregate')
    print(json.dumps({'wban': args.wban, 'hour_end_utc': end.isoformat(),
                      'official_total_mm': target.official_hourly.iloc[0],
                      'derived_total_mm': expected, 'observed_5min_depths_mm': depths.tolist()}, indent=2))


if __name__ == '__main__':
    main()
