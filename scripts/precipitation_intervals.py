"""Read checked station-year partitions and group wet intervals."""

import hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

DRY_STEPS = 72

def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read_partition(record, layer, columns, grid):
    spec = record['outputs'][layer]
    path = ROOT / spec['path']
    if sha256(path) != spec['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
    if len(frame) != spec['rows']:
        raise ValueError(f'Row count mismatch: {path}')
    if layer == 'hourly_pairs':
        frame = frame.loc[frame['variable'].eq('precipitation')]
    if set(frame['station_wban']) != {record['station_wban']}:
        raise ValueError(f'WBAN mismatch: {path}')
    frame = frame.set_index('time_end_utc')
    if frame.index.has_duplicates or not frame.index.isin(grid).all():
        raise ValueError(f'Duplicate or off-grid endpoint: {path}')
    return frame.reindex(grid)

def event_spans(values):
    """Group wet endpoints across short observed dry periods, breaking at unknowns."""
    wet = np.flatnonzero(values > 0)
    if len(wet) == 0:
        return []
    unknown_prefix = np.r_[0, np.cumsum(~np.isfinite(values), dtype=np.int64)]
    gap_has_unknown = unknown_prefix[wet[1:]] != unknown_prefix[wet[:-1] + 1]
    breaks = (np.diff(wet) > DRY_STEPS) | gap_has_unknown
    return list(zip(wet[np.r_[True, breaks]], wet[np.r_[breaks, True]], strict=True))
def rolling_amounts(values, width):
    return np.lib.stride_tricks.sliding_window_view(values, width).sum(axis=-1)
