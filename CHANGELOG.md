# Changelog

All notable changes to GWBASE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-06

Round-1 revision release for the GWBASE manuscript. Contains the library
changes behind the revised results, plus the analysis scripts and outputs
generated in response to reviewer comments.

### Changed

- **Mann-Kendall trend test now uses the Hamed-Rao modification.**
  `gwbase.analysis._mk_row` calls `mk.hamed_rao_modification_test` instead of
  `mk.original_test`. The Hamed-Rao variant corrects the variance of the MK
  statistic for serial correlation, which is present in annual mean water-table
  and discharge series. This changes reported trend significance; trend
  directions and Sen's slopes are unaffected.
- **`aggregate_streamflow_monthly_bfd` returns an additional `n_bfd_days`
  column** giving the number of baseflow-dominated days each monthly mean rests
  on. Callers that select columns positionally should be checked.
- Scatter plots from `plot_delta_scatter` are written as
  `gage_<id>_general_scatter.png` rather than `gage_<id>.png`, and no longer
  carry an embedded title (titles are supplied by the manuscript captions).

### Added

- **`min_bfd_days` parameter** on
  `gwbase.pairing.aggregate_streamflow_monthly_bfd`, with a matching
  `streamflow.min_bfd_days` key in `config.yaml`. Months containing fewer than
  this many BFD days are dropped before pairing. The default of `1` reproduces
  v0.1.0 behaviour, in which a month represented by a single BFD day carried the
  same weight as a fully baseflow-dominated month.
- Analysis scripts under `notebooks/` supporting the round-1 revision:
  Hamed-Rao trend testing, deseasonalised within-well regression, basin-sum
  over the four retained catchments, descriptive statistics tables, lake-reach
  verification, and Spanish-Fork-excluded recomputations.
- Generated outputs for the above under `results/round1_revision/`, with a
  README describing each folder.

### Fixed

- **scipy >= 1.12 compatibility.** `calculate_well_metrics` and
  `compute_delta_metrics` now coerce their input columns with
  `pd.to_numeric(..., errors='raise')` before use. Object-dtype columns arriving
  from upstream merges previously raised
  `data type dtype('O') not compatible with finfo` on newer scipy, where older
  versions coerced silently. Values are unchanged.
- Lake-crossing reach artifact: GEOGLOWS draws synthetic reaches through lake
  polygons to preserve network topology. Reaches contained entirely within the
  lake polygon are now dropped, while reaches that only clip its edge (real
  river mouths, such as the Bear River outlet into Bear River Bay) are retained.

## [0.1.0] - 2026-06-19

Initial release accompanying the submitted manuscript.

[0.2.0]: https://github.com/BYU-Hydroinformatics/gwbase/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/BYU-Hydroinformatics/gwbase/releases/tag/v0.1.0
