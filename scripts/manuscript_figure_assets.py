"""Load and verify map assets and export manuscript figures."""

import hashlib
import json
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.io.shapereader import BasicReader
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

MAPS = ROOT / 'data/reference/natural_earth'

FIGURES = ROOT / 'figures'

INK = '#1D2429'

LONLAT = ccrs.PlateCarree()

def read_json(path):
    return json.loads(path.read_bytes())

def basemap_geometry():
    geometry = {}
    for name, item in read_json(MAPS / 'sources.json')['assets'].items():
        path = MAPS / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Natural Earth archive changed: {name}')
        reader = BasicReader(str(path))
        if 'admin_1' in name:
            geometry[name] = tuple(r.geometry for r in reader.records() if r.attributes['ADM0_A3'] == 'USA')
        else:
            geometry[name] = tuple(reader.geometries())
        reader.close()
        if not geometry[name]:
            raise ValueError(f'Natural Earth archive is empty: {name}')
    return geometry

def export(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'{stem}.pdf', metadata={'CreationDate': None})
    fig.savefig(FIGURES / f'{stem}.svg', metadata={'Date': None})
    fig.savefig(FIGURES / f'{stem}.png', dpi=450)
    plt.close(fig)
