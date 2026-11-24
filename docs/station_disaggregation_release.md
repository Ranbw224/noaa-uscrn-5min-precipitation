# Station disaggregation release, 2007–2025

After extracting the archive, read the [release README](../README.md) for schemas, event eligibility and UTC support. The relative file paths, row counts and SHA-256 digests for primary data are recorded in `data/portable_index.json`.

| Archive group | Files and purpose |
|---|---|
| Core | Event catalogue, site metadata, envelope time series, scores, provenance, diagnostic tables, the field dictionary and saved models |
| Precipitation | `data/pairs/variable=precipitation/` and the corresponding annotation partitions |
| Covariates | Pair and annotation partitions for air temperature, solar radiation and relative humidity |
| Observations | Complete five-minute observation partitions by source year and station |

Archive paths are relative to the same release root, so extracted groups can be combined. Choose paths from the index for the group installed. In PyArrow, the partition key `variable` has type `large_string`. `source_year` uses `int16` in pairs and `int64` in annotations. Supply these types explicitly for Hive directory reads, or open the indexed Parquet paths directly.

The core package also contains source citations and audit evidence. The numerical tables supporting the manuscript figures and sensitivity analyses are under `supplementary/manuscript/`.

## Event catalogue data dictionary

| Field group | Columns in `data/events.parquet` | Meaning and reuse |
|---|---|---|
| Eligibility | `usability_tier`, `boundary_confirmed`, `hourly_benchmark_eligible`, `comparison_ready`, `ineligible_covering_hours`, `ineligible_source_years` | Tier 1 has confirmed boundaries and eligible official covering hours. Tier 2 has confirmed boundaries and at least one ineligible official covering hour. Tier 3 has unconfirmed boundaries. `ineligible_covering_hours` counts failures of `track_eligible_official`. Zero is equivalent to `hourly_benchmark_eligible` being true. `ineligible_source_years` lists distinct source years with a failed covering hour and is empty when all hours pass. |
| Year boundary | `cross_year` | True when the right-closed observed span extended by a nominal six hours at each end intersects multiple UTC years. For Tier 3 the extended window may include unconfirmed buffers. Consult the boundary-status fields for observed buffer support. |
| Storm profile | `temporal_centroid_ratio`, `storm_profile_tercile`, `earliest_peak_ratio`, `has_max_interval_tie` | The ratios use normalized event time. The earliest-peak sensitivity analysis uses the same cutpoints as the centroid. Values below 0.33 are front-loaded, values in [0.33, 0.67] are mid-loaded, and values above 0.67 are back-loaded. These fields are missing in Tier 3. |
| Phase and temperature | `event_temp_min_c`, `event_temp_mean_c`, `event_temp_max_c`, `precip_phase_flag`, `event_wetbulb_mean_c` | The phase screen uses the 1.5 m `AIR_TEMPERATURE` summaries. The separate wet-bulb estimate uses standard sea-level pressure and requires accepted temperature and humidity at the same timestamp. These fields are missing in Tier 3. |
| Soil moisture | `soil_moisture_pre_*cm`, `soil_moisture_missing_reason_*cm`, `soil_moisture_pre_event_offset_min` | Hourly volumetric content in m³/m³ comes from the corresponding `hourly02` depth at 5, 10, 20, 50 or 100 cm. The screening excludes sentinels, nonfinite values and values outside [0, 1]. There is no independent source QC flag. The shared offset is populated when at least one depth has a valid candidate. The observation layer keeps the separate five-minute 5 cm field. These event fields are missing in Tier 3. |
| Wind | `event_wind_mean_ms` | Mean accepted `WIND_1_5` in m/s over the observed event. Accepted readings require `WIND_FLAG` equal to zero. Recomputing the diagnostic requires the full five-minute observation package. This field is missing in Tier 3. |
| Antecedent dry duration | `antecedent_dry_hours_lower_bound`, `antecedent_dry_censored` | Available for all segments. The stored duration gives a lower bound for confirmed events. Tier 3 starts are provisional and may coincide with gaps. Use those rows for gap tracing and exclude them from duration distributions, moisture stratification and frequency estimation. For the observed dry-duration distribution, select Tier 1 or Tier 2 and `antecedent_dry_censored` false. Report coverage and censoring by station, source year or climate group as applicable. |

The soil-moisture reason `valid` indicates an accepted candidate. `no_sensor_prior_to_deployment` covers event starts before the first valid record at that station and depth, including depths with no valid record. `transient_missing_after_deployment` covers rejected or missing candidates at or after the first valid record. The first valid record is an observational availability boundary. Summarize these reason codes for confirmed events in Tiers 1 and 2.

## Annotation and endpoint audit fields

For the exploratory `cross_var_consistency_flag`, use current exact-derived precipitation, relative humidity and radiation. The first matching rule wins.

1. `missing_covariate` if any of the three values is missing.
2. `rain_low_rh` if precipitation is above zero and relative humidity is below 60%.
3. `rain_high_rad` if precipitation is above zero, radiation exceeds 600 W/m² and the UTC hour ending is in 16:00–22:00, inclusive.
4. `consistent` otherwise.

The archived UTC window is fixed. For local diagnostics, convert the hour ending to approximate local mean solar time as `(decimal UTC hour + east-positive longitude in degrees / 15) % 24`. Choose and report the window in that time convention. The archived flag is excluded from quality filtering and model features. Use one row per WBAN, source year and hourly endpoint when tabulating the flag.

The temperature endpoint audit is stored in `provenance/m2_summary.json`. `T_CALC_endpoint.compared_hours` counts recorded hours with a usable value on both products. Hours missing one or both values are excluded.
