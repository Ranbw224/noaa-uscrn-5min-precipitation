"""Draw Figure 1 from the manuscript's verified archive and replay counts."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, OCHRE, SLATE = '#1F4E78', '#2E7D6B', '#B87333', '#5A6578'
INK, GRID = '#1A202C', '#CBD5E1'
HOUR, OBSERVED = '#E4EEF7', '#647D98'
STYLE = {'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'figure.facecolor': 'white'}


def scalar_macros(path):
    return dict(re.findall(r'^\\newcommand\{\\([A-Za-z]+)\}\{([^{}\n]*)\}$',
                           path.read_text(encoding='utf-8-sig'), re.M))


def card(ax, x, y, width, height, title, subtitle, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0,rounding_size=1',
                               facecolor=fill, edgecolor=edge, lw=1.4, zorder=2))
    ax.plot([x, x + width], [y + height - 3.2] * 2, color=edge, lw=.8, zorder=3)
    ax.text(x + width / 2, y + height - 1.6, title, color=edge, fontsize=9.1, weight='bold',
            ha='center', va='center', zorder=4)
    ax.text(x + 1.0, y + height - 4.2, subtitle, color=INK, fontsize=8.0, va='top',
            linespacing=1.22, zorder=4)


def connection(ax, source, target, color=SLATE, curve='arc3', two_way=False):
    ax.annotate('', xy=target, xytext=source,
                arrowprops=dict(arrowstyle='<|-|>' if two_way else '-|>',
                                color=color, lw=1.15, connectionstyle=curve))


def timeline(ax, dry_hours):
    ax.add_patch(FancyBboxPatch((1.5, 51.4), 97, 23.7, boxstyle='round,pad=0,rounding_size=1',
                               facecolor='#FAFCFF', edgecolor=GRID, lw=1.15))
    ax.text(3, 72.5, 'Temporal support alignment and complete-hour replay (schematic)',
            color=NAVY, weight='bold', fontsize=9)

    steps_per_hour = 12
    observed = np.array([
        0, 0, 0, 0, 0, .8, 1.4, .7, .3, 0, 0, 0,
        1.4, 3.1, 4.3, 2.9, 1.4, .6, .1, 0, .4, .5, .2, 0,
        0, .4, 1.2, .8, .7, .4, 0, 0, 0, 0, 0, 0,
    ])
    ends = np.linspace(15.4, 47.7, observed.size + 1)
    hour_ends = ends[::steps_per_hour]
    baseline = 59.1
    for left, right in zip(hour_ends[:-1], hour_ends[1:]):
        ax.add_patch(Rectangle((left, baseline), right - left, 9.0, facecolor=HOUR,
                               edgecolor='#A5BCD1', linewidth=.65, zorder=2))
    ax.bar((ends[:-1] + ends[1:]) / 2, observed * 1.55, bottom=baseline,
           width=np.diff(ends) * .86, color=OBSERVED, align='center', zorder=3)
    hourly_average = observed.reshape(-1, steps_per_hour).mean(axis=1)
    ax.stairs(baseline + 1.55 * hourly_average, hour_ends, baseline=None,
              color=NAVY, linewidth=1.35, zorder=4)
    wet = np.flatnonzero(observed)
    for x in ends[[wet[0], wet[-1] + 1]]:
        ax.plot([x, x], [baseline - .12, 68.2], color=TEAL, lw=1.2, ls='--', zorder=5)
    ax.text((ends[wet[0]] + ends[wet[-1] + 1]) / 2, 69.2,
            r'Observed wet span $(t_{\mathrm{start}},t_{\mathrm{end}}]$',
            fontsize=8.0, weight='bold', color=TEAL, ha='center')
    for x, label in zip(hour_ends, (r'$h-1$', r'$h$', r'$h+1$', r'$h+2$')):
        ax.text(x, 57.0, label, fontsize=7.8, color=SLATE, ha='center')
    dry_label = f'{dry_hours:g}-h dry buffer\n({dry_hours * steps_per_hour:g} accepted zeros)'
    for x in (8.5, 54.5):
        ax.text(x, 62.4, dry_label, fontsize=7.5, color=TEAL, ha='center')

    y = 53.5
    ax.add_patch(Rectangle((3, y - .55), 1.3, 1.1, facecolor=HOUR, edgecolor='#A5BCD1', linewidth=.7))
    ax.add_patch(Rectangle((15.7, y - .55), 1.25, 1.1, facecolor=OBSERVED, edgecolor='none'))
    ax.plot([32.7, 34.2], [y] * 2, color=NAVY, lw=1.35)
    ax.plot([50.7, 52.2], [y] * 2, color=TEAL, lw=1.2, ls='--')
    for x, label in ((4.8, 'Official hour'), (17.5, '5-min observation'),
                     (34.7, 'Uniform reconstruction'), (52.7, 'Event bounds')):
        ax.text(x, y, label, fontsize=7.3, color=SLATE, va='center')


def workflow_figure(values, dry_hours=6):
    year_range = lambda key: values[key].replace('--', '–')
    millions = lambda key: f"{int(values[key].replace(',', '')) / 1e6:.1f}M"

    def archive(filename, field):
        value = values.get(f'Release{field}ArchiveGB')
        return f'{filename} ({value} GB)' if value else filename

    count = values.get('ReleaseObservationArchiveCount')
    size = values.get('ReleaseObservationsArchiveGB')
    observation_archives = (f'{count} ZIPs by source year ({size} GB total)'
                            if count and size else 'ZIPs grouped by source year')

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10.6, 7.45))
        ax = fig.add_axes([.012, .012, .976, .976], xlim=(0, 100), ylim=(0, 100))
        ax.set_axis_off()
        ax.text(1, 97.5, 'a   Archive construction workflow and temporal support alignment',
                fontsize=11.5, weight='bold', color=INK)
        stages = [
            ('NOAA snapshots', f'• subhourly01 and hourly02\n• {values["StationCount"]} WBANs ({year_range("ArchivePeriod")})'
             '\n• Source URLs and SHA-256\n• Raw byte audit', SLATE, '#F4F6F9'),
            ('Parse and screen', '• Right-closed UTC intervals\n• Leading NUL prefixes\n• Source flags and physical bounds'
             '\n• T_CALC endpoint check', NAVY, '#EEF4F9'),
            ('Dual hourly tracks', '• Official and exact-derived\n• Rain, air temp., solar and RH\n• Adjacent-hour input masks'
             '\n• SHA-256 station split', NAVY, '#EEF4F9'),
            ('Event catalogue', f'• {values["EventCount"]} wet segments\n• {values["ConfirmedCount"]} confirmed events'
             '\n• 5/15/30-min rolling peaks\n• Profile, phase, soil and wind', TEAL, '#EBF5F3'),
        ]
        for x, (title, subtitle, edge, fill) in zip((1.5, 26.5, 51.5, 76.5), stages):
            card(ax, x, 81.5, 21 if x < 76 else 22, 13.5, title, subtitle, edge, fill)
        for x in (22.7, 47.7, 72.7):
            connection(ax, (x, 88.5), (x + 3.6, 88.5))

        timeline(ax, dry_hours)
        card(ax, 63.8, 55.1, 33.2, 15.4, f'Held-out complete-hour replay ({year_range("ReplayPeriod")})',
             f'• {values["ReplayEvents"]} events ({values["MainEvents"]} at ≥{values["ReplayPrimaryPeakMin"]} mm; '
             f'{values["HighIntensityEvents"]} at ≥{values["HighCutoff"]} mm)\n'
             '• K complete hours and 12K five-minute steps\n'
             '• Uniform allocation, constrained linear and\n'
             f'  ExtraTrees (fitted {year_range("FittingPeriod")})\n'
             '• 15-min underestimation and 5/15/30-min errors',
             OCHRE, '#FDF6EE')

        ax.plot([1, 99], [49.4] * 2, color=GRID, lw=.85, ls='--')
        ax.text(1, 46.0, 'b   Released archive tables, row counts, join keys and package layout',
                fontsize=11.5, weight='bold', color=INK)
        for x, height, fill, edge in ((1, 28, '#F8FAFC', '#94A3B8'), (34.5, 28, '#F8FAFC', '#94A3B8'),
                                      (68, 42.5, '#FBF9F5', '#CDB99F')):
            ax.add_patch(FancyBboxPatch((x, 1.5), 31, height, boxstyle='round,pad=0,rounding_size=1',
                                       facecolor=fill, edgecolor=edge, lw=1.0, ls='--', zorder=1))
        for x, y, label, color in (
            (16.5, 4.6, 'Five-minute observations', SLATE),
            (16.5, 2.3, observation_archives, SLATE),
            (50, 4.6, archive('precipitation_pairs.zip', 'Precipitation'), SLATE),
            (50, 2.3, archive('covariate_pairs.zip', 'Covariates'), SLATE),
            (83.5, 2.6, archive('core_events.zip', 'Core'), OCHRE),
        ):
            ax.text(x, y, label, fontsize=7.5, weight='bold', color=color, ha='center')
        card(ax, 35.5, 32.4, 29, 11.6, f'station_metadata.csv ({values["StationCount"]} rows)',
             'PK: station_wban\n• Coordinates, elevation, climate and network\n'
             '• Station split\n• Included in core_events.zip', SLATE, 'white')
        card(ax, 2.5, 13, 28, 14, f'observations/ ({millions("FineCount")} rows)',
             'PK: station_wban, source_year,\n    time_end_utc\n• Hive-partitioned Parquet\n'
             '• Accepted and source five-minute values\n• Source flags and exclusion codes', NAVY, 'white')
        card(ax, 35.5, 17.5, 29, 10.5, f'pairs/ ({millions("PairCount")} rows)',
             'PK: station_wban, source_year,\n    time_end_utc, variable\n'
             '• Official and derived hourly values and masks', NAVY, 'white')
        card(ax, 35.5, 7.1, 29, 7.6, f'annotations/ ({millions("PairCount")} rows)',
             'PK: Same composite key as pairs/\n• Completion classes and cross-variable flag', NAVY, 'white')
        card(ax, 69.5, 28, 28, 14, f'events.parquet ({values["EventCount"]} rows)',
             'PK: event_id   FK: station_wban\n• UTC bounds and usability tier\n• 5/15/30-min peaks and P95 reference'
             '\n• Centroid, earliest peak, phase, soil and wind\n• Antecedent dry duration and censoring', TEAL, 'white')
        card(ax, 69.5, 8.5, 28, 12, 'envelope_timeseries.parquet',
             f'PK: event_id, time_end_utc ({values["ReplayBins"]} rows)\n'
             f'• {values["ReplayEvents"]} held-out events\n• Shared five-minute observations\n'
             '• Three frozen baseline reconstructions', OCHRE, 'white')
        connection(ax, (35.5, 39.5), (16.5, 27.2), curve='angle,angleA=180,angleB=90,rad=5')
        connection(ax, (50, 32.3), (50, 28.2))
        connection(ax, (64.7, 39.5), (69.3, 36))
        connection(ax, (50, 17.3), (50, 14.9), color=NAVY, two_way=True)
        connection(ax, (83.5, 27.8), (83.5, 20.7), color=OCHRE)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript-dir', type=Path, default=ROOT / 'manuscript')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/revision_workflow_schema.pdf')
    args = parser.parse_args()
    macros = {}
    for file in ('station_disaggregation_results.tex', 'station_disaggregation_revision_results.tex',
                 'station_disaggregation_deposit_sizes.tex'):
        macros.update(scalar_macros(args.manuscript_dir / file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = workflow_figure(macros)
    figure.savefig(args.output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(figure)
