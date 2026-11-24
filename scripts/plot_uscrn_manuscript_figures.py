"""Draw archive coverage and the verified official-hourly example for the 2007–2025 manuscript."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import joblib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from benchmark_full import ROOT, all_predictions, save_json, sha256
from manuscript_figure_assets import INK, LONLAT, MAPS, basemap_geometry, export, plt, read_json
from replay_uscrn_release import eligible_release_samples, release_reader


NETWORKS = {
    'USCRN': ('#0072B2', 'o'),
    'USRCRN': ('#AB5915', '^'),
    'Alabama-USRCRN': ('#712C7E', 's'),
}
VARIABLES = {
    'air_temperature': 'Air temperature',
    'precipitation': 'Precipitation',
    'solar_radiation': 'Solar radiation',
    'relative_humidity': 'Relative humidity',
}
RECONSTRUCTIONS = {
    'uniform_within_hour': 'Uniform',
    'linear_bounded_mean_constrained': 'Constrained linear',
    'extra_trees_constrained': 'ExtraTrees',
}


def draw_map(fig, rect, projection, bounds, sites, basemap, place=None):
    ax = fig.add_axes(rect, projection=projection)
    ax.set_extent(bounds, crs=LONLAT)
    ax.set_facecolor('#F3F7F9')
    ax.spines['geo'].set_edgecolor('#B1BABF')
    ax.add_geometries(basemap['ne_50m_land'], LONLAT, facecolor='#E7E8E6',
                      edgecolor='#899399', linewidth=.35)
    for layer in ('ne_50m_admin_0_boundary_lines_land', 'ne_50m_admin_1_states_provinces_lines'):
        ax.add_geometries(basemap[layer], LONLAT, facecolor='none',
                         edgecolor='#A5ABAD', linewidth=.32)
    xy = projection.transform_points(LONLAT, sites.longitude.to_numpy(), sites.latitude.to_numpy())
    if not (np.all((xy[:, 0] > ax.get_xlim()[0]) & (xy[:, 0] < ax.get_xlim()[1]))
            and np.all((xy[:, 1] > ax.get_ylim()[0]) & (xy[:, 1] < ax.get_ylim()[1]))):
        raise ValueError(f'Map extent omits stations in {place or "the main map"}')
    for network, (color, marker) in NETWORKS.items():
        group = sites.loc[sites.network.eq(network)]
        ax.scatter(group.longitude, group.latitude, color=color, marker=marker, s=15,
                   edgecolors='white', linewidths=.30, zorder=4, transform=LONLAT)
    if place:
        ax.text(.03, .95, place, ha='left', va='top', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .90, 'pad': 1.5})


def coverage_figure(stations, availability, rows):
    basemap = basemap_geometry()
    fig = plt.figure(figsize=(183 / 25.4, 152 / 25.4))
    alaska = stations.state.eq('AK') & stations.country.eq('US')
    hawaii = stations.state.eq('HI') & stations.country.eq('US')
    russia = stations.country.eq('RS')
    mainland = ~alaska & ~hawaii & ~russia
    draw_map(fig, [.120, .505, .594, .428], ccrs.LambertConformal(),
             [-126, -65, 23, 50], stations[mainland], basemap)
    draw_map(fig, [.733, .735, .229, .186],
             ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=56, standard_parallels=(55, 65)),
             [-174, -130, 51, 72], stations[alaska], basemap, 'Alaska')
    draw_map(fig, [.733, .630, .229, .091], LONLAT,
             [-160.5, -154, 18.5, 22.5], stations[hawaii], basemap, 'Hawaii')
    draw_map(fig, [.733, .528, .229, .091], LONLAT,
             [124, 134, 69, 74], stations[russia], basemap, 'Russia')
    fig.text(.04, .956, 'a', fontsize=9, fontweight='bold')
    fig.text(.085, .956, 'Station locations', fontsize=9, fontweight='bold')
    legend = [Line2D([], [], color=color, marker=marker, linestyle='none',
                     label=network, markersize=4.8) for network, (color, marker) in NETWORKS.items()]
    fig.legend(handles=legend, loc='center', bbox_to_anchor=(.49, .500),
               ncol=len(legend), frameon=False, handletextpad=.35)

    years = availability.index.to_numpy()
    first, last = int(years[0]), int(years[-1])
    ticks = np.unique([first, *range(first + 3, last, 3), 2015, last])
    limits = (first - .5, last + .5)
    station_ax = fig.add_axes([.177, .327, .772, .119])
    station_ax.plot(years, availability.paired_station_years, marker='o',
                    markersize=2.8, color=INK, linewidth=1.05)
    station_ax.set(xlim=limits, xticks=ticks, ylabel='Paired stations')
    station_ax.tick_params(axis='x', labelbottom=False)
    station_ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    station_ax.grid(axis='y', color='#D9DFE2', linewidth=.45)
    station_ax.axvline(2014.5, color='#A34D28', linestyle='--', linewidth=.74)
    for year, xytext, align in [(2014, (-5, 4), 'right'), (2015, (3, -4), 'left')]:
        n = availability.loc[year, 'paired_station_years']
        station_ax.annotate(str(n), (year, n), xytext=xytext, textcoords='offset points',
                            ha=align, va='bottom' if year == 2014 else 'top', color='#8E411E')
    fig.text(.04, .462, 'b', fontsize=9, fontweight='bold')
    fig.text(.085, .462, 'Paired station inventory', fontsize=9, fontweight='bold')

    ylabels = list(VARIABLES.values())
    for label, product, y in [('c', 'subhourly01', .181), ('d', 'hourly02', .052)]:
        ax = fig.add_axes([.177, y, .772, .089])
        subset = rows.loc[rows['product'].eq(product)]
        values = subset.pivot(index='variable', columns='year', values='available_pct').reindex(
            index=list(VARIABLES), columns=years)
        if values.isna().any().any():
            raise ValueError(f'Availability years or variables are missing for {product}')
        raster = ax.pcolormesh(np.arange(first - .5, last + 1.5), np.arange(len(ylabels) + 1),
                               values, cmap='cividis', vmin=0, vmax=100,
                               linewidth=.19, edgecolors='#F5F6F7')
        ax.set(xlim=limits, ylim=(len(ylabels), 0), xticks=ticks,
               yticks=np.arange(len(ylabels)) + .5, yticklabels=ylabels)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelbottom=label == 'd', length=2)
        ax.set_title(product, loc='left', pad=4, fontsize=8)
        fig.text(.04, y + .107, label, fontsize=9, fontweight='bold')
        ax.axvline(2014.5, color='#FFFFFF', linestyle='--', linewidth=.75)
    bar = fig.colorbar(raster, cax=fig.add_axes([.668, .290, .210, .011]),
                       orientation='horizontal', ticks=[0, 25, 50, 75, 100])
    bar.ax.xaxis.set_ticks_position('top')
    bar.ax.tick_params(length=1.6, pad=1)
    bar.outline.set_linewidth(.35)
    fig.text(.653, .292, 'Accepted calendar intervals (%)', ha='right', va='center')
    return fig


def verify_example(release, protocol, config, records):
    replay = protocol['envelope_replay']
    event_id = replay['example_event_id']
    match = [('event_id', '=', event_id)]
    event_frame = pd.read_parquet(release / 'data/events.parquet', filters=match)
    if len(event_frame) != 1:
        raise ValueError(f'Expected one catalogue entry for {event_id}')
    event = next(event_frame.itertuples(index=False))
    if not (event.usability_tier == 1 and event.dataset_split == replay['station_split']):
        raise ValueError(f'Example is outside the official replay cohort: {event_id}')
    track, variable = replay['track'], replay['variable']
    model = next(item for item in read_json(release / 'model_metadata.json')
                 if (item['track'], item['variable']) == (track, variable))
    model_path = release / model['path']
    model_hash = sha256(model_path)
    if model_hash != model['sha256']:
        raise ValueError('The official-hourly example model failed its checksum')
    bundle = joblib.load(model_path)
    first, last = config['fitting_years']
    if (bundle['settings']['train_years'] != list(range(first, last + 1))
            or event.station_wban in bundle['training_wbans']):
        raise ValueError('The example model does not match the frozen fitting protocol')
    record = next(record for record in records if record['station_wban'] == event.station_wban
                  and record['year'] == event.first_source_year)
    high, pairs = release_reader({**record, 'release_root': str(release)})
    ends, x, targets, _ = eligible_release_samples(high, pairs, track, variable)
    hours = pd.date_range(event.first_covering_hour_end_utc, event.last_covering_hour_end_utc, freq='h')
    index = ends.get_indexer(hours)
    if (index < 0).any():
        raise ValueError('The official inputs do not cover the example envelope')
    step = pd.Timedelta(minutes=protocol['interval_minutes'])
    times = pd.date_range(hours[0] - pd.Timedelta(hours=1) + step, hours[-1], freq=step)
    series_path = release / 'data/envelope_timeseries.parquet'
    series = pd.read_parquet(series_path, filters=match).sort_values('time_end_utc')
    if (not pd.DatetimeIndex(series.time_end_utc).equals(times)
            or not np.array_equal(series.inside_observed_event, (times > event.event_start_utc) & (times <= event.event_end_utc))
            or len(series) != event.covering_hours * int(pd.Timedelta(hours=1) / step)):
        raise ValueError('The example archive does not match the expected complete-hour grid')
    predicted = all_predictions(bundle, x[index], variable)
    recomputed = {'observed': targets[index].ravel(),
                  **{method: predicted[method].ravel() for method in RECONSTRUCTIONS}}
    errors = {}
    tolerance = protocol['peak_depth_tolerance_mm']
    for key, values in recomputed.items():
        if not np.allclose(series[key], values, atol=tolerance, rtol=0):
            raise ValueError(f'Example differs from the verified model/observation for {key}')
        errors[key] = float(np.abs(series[key] - values).max())
    return event, series, {
        'event_id': event_id,
        'station_wban': event.station_wban,
        'envelope_start_utc': (times[0] - step).isoformat(),
        'envelope_end_utc': times[-1].isoformat(),
        'observed_start_utc': event.event_start_utc.isoformat(),
        'observed_end_utc': event.event_end_utc.isoformat(),
        'five_minute_bins': len(series),
        'fit_years': [first, last],
        'model': model['path'],
        'model_sha256': model_hash,
        'comparison_tolerance_mm': tolerance,
        'max_absolute_difference_mm': errors,
    }


def replay_figure(event, series, step_minutes):
    step = pd.Timedelta(minutes=step_minutes)
    edges = pd.DatetimeIndex([series.time_end_utc.iloc[0] - step, *series.time_end_utc])
    top = series[['observed', *RECONSTRUCTIONS]].max().max() * 1.12
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(183 / 25.4, 107 / 25.4))
    fig.subplots_adjust(left=.133, right=.974, top=.859, bottom=.148, hspace=.44)
    fig.supylabel(f'Precipitation (mm per {step_minutes} min)', x=.027, size=8.5)
    for i, (ax, (method, label)) in enumerate(zip(axes, RECONSTRUCTIONS.items())):
        ax.stairs(series.observed, edges, baseline=0, fill=True, linewidth=.5,
                  color='#B9C3C9', label='Observed', zorder=1)
        ax.stairs(series[method], edges, color='#0069A3', linewidth=1.1,
                  label='Reconstructed', zorder=3)
        for time in (event.event_start_utc, event.event_end_utc):
            ax.axvline(time, color='#454B50', linestyle=(0, (3, 2)), linewidth=.85, zorder=4)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(axis='x', color='#C9D1D6', linewidth=.52)
        ax.tick_params(axis='x', length=2)
        ax.set_title(label, loc='left', fontsize=8, pad=4)
        ax.text(-.095, 1.045, chr(ord('a') + i), transform=ax.transAxes,
                fontweight='bold', fontsize=9, va='bottom')
    axes[-1].set_xlabel('UTC')
    fig.legend(handles=[
        Patch(facecolor='#B9C3C9', edgecolor='none', label='Observed'),
        Line2D([], [], color='#0069A3', linewidth=1.1, label='Reconstructed'),
        Line2D([], [], color='#454B50', linewidth=.85, linestyle=(0, (3, 2)), label='Observed event bounds'),
    ], loc='upper center', bbox_to_anchor=(.53, .982), ncol=3, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    args = parser.parse_args()
    release = args.release.resolve()
    audit = read_json(release / 'audit_log/validation_summary.json')
    if (audit['status'] != 'passed' or set(audit['assertions']) != set('123456')
            or set(audit['assertions'].values()) != {'passed'}
            or sha256(release / 'data/portable_index.json') != audit['portable_index_sha256']):
        raise ValueError('Figures require the validated M4 archive')
    config = read_json(release / 'provenance/release_configuration.json')
    protocol = read_json(release / 'provenance/event_protocol.json')
    base = read_json(ROOT / config['base_manifest'])['records']
    stations_path = release / 'data/station_metadata.csv'
    availability_path = release / 'technical_validation/annual_availability.csv'
    missingness_path = release / 'technical_validation/annual_variable_missingness.csv'
    stations = pd.read_csv(stations_path, dtype={'station_wban': str})
    availability = pd.read_csv(availability_path).set_index('year').sort_index()
    rows = pd.read_csv(missingness_path)
    if (not stations.station_wban.is_unique or set(stations.station_wban) != {r['station_wban'] for r in base}
            or set(stations.network) != set(NETWORKS) or list(availability.index) != protocol['source_years']
            or not availability.paired_station_years.eq(pd.DataFrame(base).groupby('year').station_wban.nunique()).all()
            or not rows.missing_or_bad_rows.between(0, rows.calendar_rows).all()):
        raise ValueError('The map roster or yearly coverage differs from the release')
    rows['available_pct'] = 100 * (rows.calendar_rows - rows.missing_or_bad_rows) / rows.calendar_rows
    event, series, example = verify_example(release, protocol, config, base)

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'axes.linewidth': .55,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': .55, 'ytick.major.width': .55,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    stems = ('full_dataset_coverage_2007_2025', 'precipitation_hourly_envelope_2007_2025')
    export(coverage_figure(stations, availability, rows), stems[0])
    export(replay_figure(event, series, protocol['interval_minutes']), stems[1])
    report = ROOT / config['reports_directory'] / 'm5'
    report.mkdir(parents=True, exist_ok=True)
    series.to_csv(report / 'figure_2_plot_data.csv', index=False, lineterminator='\n')
    inputs = [stations_path, availability_path, missingness_path, release / 'data/envelope_timeseries.parquet',
              release / 'provenance/event_protocol.json', MAPS / 'sources.json']
    outputs = [ROOT / 'figures' / f'{stem}.{ext}' for stem in stems for ext in ('pdf', 'svg', 'png')]
    save_json(report / 'figure_sources.json', {
        'inputs_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
        'outputs_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs},
        'coverage': {
            'station_counts_by_network': stations.network.value_counts().to_dict(),
            'stations': len(stations),
            'paired_stations_by_year': availability.paired_station_years.to_dict(),
            'variable_year_product_cells': len(rows),
            'availability_denominator': 'Calendar intervals pooled over paired station-years per product and year',
        },
        'example': example,
    })
    print(f'Figure 1: {len(stations)} sites and {len(availability)} years')
    print(f'Figure 2: {example["event_id"]}, {len(series)} bins, max differences {example["max_absolute_difference_mm"]}')


if __name__ == '__main__':
    main()
