"""Summarize catalogue composition and replay differences."""

import numpy as np
import pandas as pd

PROFILES = ('front_loaded', 'mid_loaded', 'back_loaded')

METHODS = ('uniform_within_hour', 'linear_bounded_mean_constrained', 'extra_trees_constrained')

def climate(values):
    return values.fillna('Unclassified')

def count_tiers(frame):
    return {f'tier_{tier}_events': int(frame.usability_tier.eq(tier).sum()) for tier in (1, 2, 3)}

def event_coverage(events, sites, station_years, release):
    metadata = sites.copy()
    metadata['climate'] = climate(metadata.kg_group)
    elevation = metadata.elevation_m.to_numpy()
    metadata['elevation'] = np.select(
        [~np.isfinite(elevation), elevation < 500, elevation <= 1500],
        ['Unclassified', '<500 m', '500–1500 m'], default='>1500 m')
    data = events.merge(metadata[['station_wban', 'climate', 'elevation']], on='station_wban', validate='many_to_one')
    table = []
    for dimension, labels in {
        'climate': ['A', 'B', 'C', 'D', 'E', 'Unclassified'],
        'elevation': ['<500 m', '500–1500 m', '>1500 m', 'Unclassified'],
        'network': list(sorted(metadata.network.unique())),
    }.items():
        for label in labels:
            subset = data[data[dimension].eq(label)]
            site_group = metadata[metadata[dimension].eq(label)]
            table.append({'dimension': dimension, 'group': label, 'events': len(subset),
                          'sites_with_events': subset.station_wban.nunique(), 'metadata_sites': len(site_group),
                          'valid_precipitation_hours': station_years.loc[
                              station_years.station_wban.isin(site_group.station_wban), 'valid_observation_hours'].sum(),
                          **count_tiers(subset)})
    pd.DataFrame(table).to_csv(release / 'technical_validation/catalogue_coverage.csv', index=False)
    joint = []
    for (network, code, band), subset in data.groupby(['network', 'climate', 'elevation'], sort=True):
        joint.append({'network': network, 'climate': code, 'elevation': band, 'events': len(subset),
                      'sites_with_events': subset.station_wban.nunique(), **count_tiers(subset)})
    pd.DataFrame(joint).to_csv(release / 'technical_validation/catalogue_joint_coverage.csv', index=False)

def spread(values):
    quantiles = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75], method='linear')
    q1, median, q3 = [float(value) for value in quantiles]
    return {'median': median, 'q1': q1, 'q3': q3, 'iqr': q3 - q1}
