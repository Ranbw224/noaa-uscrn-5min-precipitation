"""Read and screen NOAA USCRN native observation rows."""

import io
import numpy as np
import pandas as pd
from source_text import read_clean_source

SH = ('WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE '
      'AIR_TEMPERATURE PRECIPITATION SOLAR_RADIATION SR_FLAG SURFACE_TEMPERATURE '
      'ST_TYPE ST_FLAG RELATIVE_HUMIDITY RH_FLAG SOIL_MOISTURE_5 SOIL_TEMPERATURE_5 '
      'WETNESS WET_FLAG WIND_1_5 WIND_FLAG').split()

HH = ('WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE '
      'T_CALC T_HR_AVG T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG SOLARAD_MAX '
      'SOLARAD_MAX_FLAG SOLARAD_MIN SOLARAD_MIN_FLAG SUR_TEMP_TYPE SUR_TEMP '
      'SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG SUR_TEMP_MIN SUR_TEMP_MIN_FLAG '
      'RH_HR_AVG RH_HR_AVG_FLAG SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 '
      'SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 SOIL_TEMP_20 '
      'SOIL_TEMP_50 SOIL_TEMP_100').split()

VARS = {
    'air_temperature': ('AIR_TEMPERATURE', 'T_HR_AVG', None, None, 'degC', .100001),
    'precipitation': ('PRECIPITATION', 'P_CALC', None, None, 'mm_per_interval', .100001),
    'solar_radiation': ('SOLAR_RADIATION', 'SOLARAD', 'SR_FLAG', 'SOLARAD_FLAG', 'W_m-2', 1.000001),
    'relative_humidity': ('RELATIVE_HUMIDITY', 'RH_HR_AVG', 'RH_FLAG', 'RH_HR_AVG_FLAG', 'percent', 1.000001),
}

def read_product(path, subhourly, _clean_source=None):
    cols = SH if subhourly else HH
    text, byte_audit = _clean_source if _clean_source is not None else read_clean_source(path, len(cols))
    source = io.StringIO(text) if byte_audit['nul_byte_count'] else path
    df = pd.read_csv(source, sep=r'\s+', header=None,
                     dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str})
    if df.shape[1] != len(cols):
        raise ValueError(f'Bad schema: {path}')
    df.columns = cols
    if df.isna().any().any():
        raise ValueError(f'Unexpected blank field instead of documented sentinel: {path}')
    stamp = pd.to_datetime(df.UTC_DATE + df.UTC_TIME.str.zfill(4), format='%Y%m%d%H%M', utc=True)
    df.index = pd.DatetimeIndex(stamp, name='time_end_utc')
    df.attrs['source_text_audit'] = byte_audit
    return df

def good_values(df, field, flag):
    raw = pd.to_numeric(df[field], errors='raise')
    # Used fields have -9999 or (6-character solar integer) -99999 sentinels.
    value = raw.mask(raw <= -9990)
    if flag:
        value = value.mask(df[flag] != 0)
    return value.astype(float)

def interval_grid(subhourly, year):
    return pd.date_range(f'{year}-01-01T00:05Z' if subhourly else f'{year}-01-01T01:00Z',
                         f'{year+1}-01-01T00:00Z', freq='5min' if subhourly else 'h',
                         name='time_end_utc')

def characterize(df, subhourly, station, year):
    expected = interval_grid(subhourly, year)
    record = {'station': station, 'year': year, 'product': 'subhourly01' if subhourly else 'hourly02',
              'rows': len(df), 'expected_rows': len(expected),
              'duplicate_timestamps': int(df.index.duplicated().sum()),
              'absent_timestamps': len(expected.difference(df.index)),
              'unexpected_timestamps': len(df.index.difference(expected)),
              'first_timestamp_utc': str(df.index.min()), 'last_timestamp_utc': str(df.index.max()),
              'datalogger_versions': sorted(df.CRX_VN.unique().tolist()),
              'wban': str(df.WBANNO.iloc[0]),
              'longitude': float(df.LONGITUDE.iloc[0]), 'latitude': float(df.LATITUDE.iloc[0])}
    if record['duplicate_timestamps']:
        raise ValueError('Duplicate times require explicit adjudication')
    records = []
    for var, (sf, hf, sq, hq, units, _) in VARS.items():
        field, flag = (sf, sq) if subhourly else (hf, hq)
        val = good_values(df, field, flag).reindex(expected)
        invalid = val.isna().to_numpy()
        edges = np.diff(np.r_[False, invalid, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        records.append({**record, 'variable': var, 'units': units,
                        'missing_or_bad_count': int(val.isna().sum()),
                        'missing_or_bad_fraction': float(val.isna().mean()),
                        'source_nonzero_qc_count': int((df[flag] != 0).sum()) if flag else 0,
                        'gap_runs': len(starts), 'longest_gap_steps': int((ends-starts).max()) if len(starts) else 0,
                        'valid_min': float(val.min()), 'valid_max': float(val.max())})
    return records
