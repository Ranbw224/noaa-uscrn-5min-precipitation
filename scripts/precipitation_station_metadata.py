"""Verify and sample the frozen historical Koppen-Geiger data."""

import hashlib
from pathlib import Path
import re
import shutil
from zipfile import ZipFile
import numpy as np
import pandas as pd
import rasterio
import requests
from precipitation_intervals import sha256

ROOT = Path(__file__).resolve().parents[1]

CLIMATE = ROOT / 'data/reference/beck_koppen_geiger'

def verify_zip(path, config):
    if path.stat().st_size != config['climate_zip_bytes'] or sha256(path) != config['climate_zip_sha256']:
        raise ValueError(f'Beck archive size or SHA-256 differs from the frozen source: {path}')
    with path.open('rb') as handle:
        if hashlib.file_digest(handle, 'md5').hexdigest() != config['climate_zip_md5']:
            raise ValueError('Beck archive does not match the Figshare MD5')

def stage_climate_files(config, archive_path):
    CLIMATE.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        pending = archive_path.with_suffix('.zip.part')
        with requests.get(config['climate_download_url'], stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with pending.open('wb') as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
        verify_zip(pending, config)
        pending.replace(archive_path)
    verify_zip(archive_path, config)
    paths = []
    with ZipFile(archive_path) as archive:
        for key in ('climate_raster_member', 'climate_legend_member'):
            member = config[key]
            target = CLIMATE / Path(member).name
            pending = target.with_suffix('.part')
            with archive.open(member) as source, pending.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            pending.replace(target)
            paths.append(target)
    return paths

def sample_climate(stations, raster_path, legend_path):
    legend = {int(code): label for code, label in re.findall(
        r'^\s*(\d+):\s*([A-Z][A-Za-z]+)\b', legend_path.read_text(encoding='utf-8'), re.M)}
    if len(legend) != 30:
        raise ValueError('Unexpected Beck climate legend')
    rows = []
    with rasterio.open(raster_path) as raster:
        if (raster.crs.to_epsg() != 4326 or raster.width != 36000 or raster.height != 18000
                or raster.res != (0.01, 0.01)):
            raise ValueError('Unexpected raster grid for the 0.01-degree climate product')
        for station in stations.itertuples(index=False):
            base = {'station_wban': station.station_wban, 'kg_raster_row': None,
                    'kg_raster_col': None, 'kg_numeric': None, 'kg_class': None, 'kg_group': None}
            if pd.isna(station.longitude) or pd.isna(station.latitude):
                rows.append({**base, 'kg_match_status': 'missing_coordinates'})
                continue
            row, col = raster.index(station.longitude, station.latitude)
            value = next(raster.sample([(station.longitude, station.latitude)], masked=True))[0]
            in_bounds = 0 <= row < raster.height and 0 <= col < raster.width
            if not in_bounds or np.ma.is_masked(value) or int(value) not in legend:
                rows.append({**base, 'kg_raster_row': row, 'kg_raster_col': col,
                             'kg_match_status': 'no_class_at_catalogue_coordinates'})
                continue
            code = legend[int(value)]
            rows.append({**base, 'kg_raster_row': row, 'kg_raster_col': col, 'kg_numeric': int(value),
                         'kg_class': code, 'kg_group': code[0], 'kg_match_status': 'matched_cell'})
        grid = {'crs': raster.crs.to_string(), 'pixel_size_degrees': raster.res,
                'height': raster.height, 'width': raster.width, 'nodata': raster.nodata}
    return stations.merge(pd.DataFrame(rows), on='station_wban', validate='one_to_one'), grid

if __name__ == '__main__':
    import json

    protocol = ROOT / 'config/precipitation_event_protocol.json'
    config = json.loads(protocol.read_bytes())['metadata']
    provenance = json.loads((CLIMATE / 'source.json').read_bytes())
    if provenance['archive_sha256'] != config['climate_zip_sha256']:
        raise ValueError('The climate provenance differs from the frozen archive')
    files = stage_climate_files(config, CLIMATE / 'koppen_geiger_tif.zip')
    for path in files:
        if sha256(path) != provenance['extracted_sha256'][path.name]:
            raise ValueError(f'Climate data differ from the frozen extract: {path.name}')
    print('Climate raster and legend verified')
