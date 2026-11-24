# NOAA five-minute precipitation catalogue and disaggregation code

This repository contains the Python workflows for the 2007–2025 NOAA USCRN/USRCRN precipitation event catalogue and complete-hour disaggregation benchmark. The workflows screen and align NOAA hourly and five-minute records, build the event catalogue, fit the fixed-parameter baselines, and evaluate reconstructions against the observed five-minute series.

The released data, model artifacts and validation records are available from [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.24455451). The [field dictionary](docs/station_disaggregation_release.md) describes the files, keys and quality fields.

## Environment

Use Python 3.13 in a virtual environment. Run commands from the repository root.

    python -m pip install -r requirements.txt
    python -m pytest -q tests

The unit tests use small fixtures. Two tests use the local NOAA base build and are skipped in a clean checkout. Install `requirements-precipitation-library.txt` for raw-data rebuilding and `requirements-manuscript-figures.txt` for the station coverage map and Sioux Falls figure.

## Use the published data

Download the required ZIP archives from Zenodo, verify them with the accompanying SHA256SUMS.txt, and extract them under a common root. The commands below assume that this root is `output/station_disaggregation_release_2007_2025/`.

| Task | Archives |
| --- | --- |
| Query events, calculate a local P95, inspect the replay or load an archived model | `core_events.zip` |
| Read the default Sioux Falls precipitation pair and its observations | `core_events.zip`, `precipitation_pairs.zip` and `five_minute_observations_2024_2025.zip` |
| Read temperature, radiation or humidity pairs | Add `covariate_pairs.zip` |
| Look up five-minute observations for another period | Use the observation ZIPs that cover the required source years |

    python examples/station_disaggregation_event_use.py output/station_disaggregation_release_2007_2025
    python examples/station_disaggregation_event_use.py output/station_disaggregation_release_2007_2025 --local-p95 kg_group
    python examples/load_portable_pair.py output/station_disaggregation_release_2007_2025 --hive
    python examples/load_saved_model.py output/station_disaggregation_release_2007_2025

The pair example preserves partition field types and uses WBAN codes as strings. Load model artifacts only from the verified release.

## Reproduce the frozen benchmark summaries

The diagnostic script reads the published observations and replay with the frozen manifests. It uses all 12 ZIPs extracted under the common root and recreates the workflow, event overview and replay-stratum figures, the supporting CSV tables and the revision result macros.

    python scripts/build_uscrn_manuscript_revision.py output/station_disaggregation_release_2007_2025 --figure-dir output/diagnostics/figures --report-dir output/diagnostics/reports --macros-out output/diagnostics/station_disaggregation_revision_results.tex

The generated tables and figures are derived from the archived predictions. The original LaTeX result macros and byte-verified configuration files are included as inputs.

## Rebuild from NOAA source records

For a rebuild against the published snapshot, place the released `provenance/source_manifest.json` at `data/uscrn_2007_2025/source_manifest.json` and restore the verified NOAA source files to the relative paths listed in the manifest. A fresh run with no local source manifest inventories the current NOAA archive. Current source files and listings can differ from the published snapshot.

    python -m pip install -r requirements-precipitation-library.txt -r requirements-manuscript-figures.txt
    python scripts/run_uscrn_ingest.py
    python scripts/summarize_uscrn_quality.py
    python scripts/precipitation_station_metadata.py
    python scripts/build_uscrn_release.py --workers 2
    python scripts/summarize_uscrn_events.py
    python scripts/replay_uscrn_release.py
    python scripts/plot_uscrn_manuscript_figures.py output/station_disaggregation_release_2007_2025

The ingest command builds the base Parquet partitions. The climate preparation script downloads the Beck et al. archive when required and verifies the pinned checksums. The replay command fits the fixed-parameter models and derives the benchmark output. The map and Sioux Falls figure use the base-build manifest as well as the processed release. Run a rebuild in a separate checkout from an extracted published release, since the default output paths coincide.

The `config/` directory records the event definitions, paired-source rules, input policy and frozen model configuration. Natural Earth basemaps, with source and license information in `data/reference/natural_earth/sources.json`, are included for coverage map reproduction.

## Data citation

Wan, F., Ran, B., Yang, H. & Zou, Y. (2026). *A five-minute precipitation event catalogue and disaggregation benchmark for NOAA reference stations* (Version 1.0.0). Zenodo. [10.5281/zenodo.24455451](https://doi.org/10.5281/zenodo.24455451).
