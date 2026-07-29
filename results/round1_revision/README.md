# Round-1 revision — computed results

Generated 2026-07-29, answering the five `[TO COMPUTE]` / `[TO VERIFY]` markers in
`reviewer_1_response.md`, `reviewer_2_response.md`, and `reviewer_3_response.md`
(paper repo: `~/2026_xueyi_GSLB_paper/review1/`).

Source data: `results/features/data_with_deltas.csv` (the delivered/verified data —
per `HANDOFF.md` this reproduces the published Table 2 to floating-point precision,
so it stands in for a fresh `result/` pipeline run). Generating scripts live in
`notebooks/round1_*.py`, run in the order below (each depends on the previous).

**Headline finding that affects all four numeric items: the basin-scale number's
own 95% CI crosses zero.** Individual gages are mostly solid (3 of 4 significant),
but summing them for one basin-scale figure reintroduces enough uncertainty
(driven by Bear River's wide CI) that the point estimate cannot be presented as
if it were established with confidence. See item 2.

---

## 1. Deseasonalised within-well spec (`01_deseasonalised_within_well/`)

Script: `notebooks/round1_deseasonalised_fe_full.py`

**The draft numbers in `reviewer_1_response.md:118-122` are confirmed exactly**
against a live run (zero lag, deseasonalised, well FE, clustered by well):

| Gage | Slope (cfs/ft) | 95% CI | p | R² | n (wells) |
|---|---|---|---|---|---|
| Bear River | +2.268 | [−3.25, +7.79] | 0.415 n.s. | 0.0008 | 22,783 (99) |
| Weber River | +0.188 | [+0.03, +0.35] | 0.018 * | 0.0014 | 18,174 (106) |
| Spanish Fork | +1.480 | [−1.56, +4.52] | 0.233 n.s. | 0.0264 | 1,686 (7) — **excluded, <10 wells** |
| Provo River | +0.687 | [+0.26, +1.11] | 0.0011 ** | 0.0039 | 10,314 (47) |
| Little Cottonwood | +0.808 | [+0.58, +1.04] | <0.001 *** | 0.1120 | 2,479 (16) |

New (not in the draft): 95% CIs, within-R², and the normalized (ΔQ/Q₀) slopes,
e.g. Little Cottonwood +0.1237 ft⁻¹ (p<0.001), Weber +0.0034 ft⁻¹ (p=0.024) —
`deseasonalised_fe_full.csv` has all lags (0/3/6/12) × all three series (raw,
deseasonalised, deseasonalised-normalized) × all 5 gages.

This is ready to go into Section 5 as its own subsection.

## 2. Basin sum over 4 catchments (`02_basin_sum_four_catchments/`)

Script: `notebooks/round1_basin_sum_four_catchments.py`

Sums the four retained gages' zero-lag deseasonalised slopes from item 1:

- **Point estimate: +3.95 cfs/ft** (within the old 3.4–9.8 range, coincidentally)
- **95% CI (propagated across catchments): [−2.00, +9.90] cfs/ft**

Per-gage share of the sum: Bear River 57%, Little Cottonwood 20%, Provo 17%,
Weber 5% — Bear River still dominates, but it's also the gage carrying almost
all of the basin CI's width (its own CI is [−3.25, +7.79]).

**This is the number that replaces 3.4–9.8 everywhere** (Abstract, §5.5, §6.3,
Conclusions, Figures 14/16/17/18) — but it can't be reported as a clean point
estimate the way 3.4–9.8 was, because **the basin-sum CI includes zero.** The
honest framing is something like "a point estimate of ~4 cfs/ft per foot of
basin-averaged decline, not distinguishable from zero at the 95% level once
catchment-level uncertainty is propagated to the basin scale" — which is a
materially more hedged claim than the submitted paper made, and is consistent
with the letter's own instruction not to state a direction until the numbers
exist. Figures 14/16/17/18 have **not** been regenerated yet — that's plotting
work on top of these numbers, not done here.

## 3. Within-well R², n, clustered 95% CI (`03_r2_n_ci_table/`)

Script: `notebooks/round1_r2_n_ci_table.py`

One table per retained gage (+ Spanish Fork, flagged) with three estimators
side by side: naive pooled OLS, pooled OLS clustered by well, and well-FE
clustered (both raw and deseasonalised). Reproduces the letter's illustrative
example almost exactly: Bear River's naive CI [+3.79, +7.23] widens to
**[−15.63, +26.64]** once clustered (letter says [−15.37, +26.38] — the ~0.3%
difference is a dof-convention rounding artifact, not a discrepancy worth
chasing). Full numbers in `r1_comment1_full_table.csv`.

## 4. A6 baseflow-magnitude comparison (`04_a6_baseflow_context/`)

Script: `notebooks/round1_a6_baseflow_context.py`

Updated to 4 gages and the new sensitivity range from item 2:

- Basin baseflow (sum of per-gage mean BFD-month q): **670 cfs**
- New sensitivity as % of baseflow: point estimate **0.59%** per ft, CI-based
  range **[−0.30%, +1.48%]** per ft (old framing was 0.47–1.35%, 5 gages)
- At the **observed** basin-median decline rate among *declining* wells
  (−0.267 ft/yr, 153 of 273 wells), 30 years of decline (8.0 ft) implies a
  point-estimate reduction of **31.6 cfs (4.7% of baseflow)**, CI-based range
  [−16.0, +79.3] cfs
- At the basin-median rate across *all* wells (−0.024 ft/yr, i.e. including
  recovering wells), 30 years implies only **2.9 cfs (0.43% of baseflow)**

**Per the letter's conditional instruction:** the "wrong denominator" pushback
against Reviewer 3 still holds directionally (baseflow, not peak flow, is the
right comparison, and 0.6–4.7% of baseflow is not "≪1%" the way peak-flow
framing made it look) — but it is **not a strong pushback**, since the
basin-sum CI crossing zero means we can't rule out the sensitivity being
genuinely negligible. The management framing should soften accordingly, as the
letter itself anticipated.

The 10 km³ 2011–2016 drought storage-loss figure (manuscript §2, ref35) was
**not** converted into a cfs comparison — there's no basin specific-yield value
in this project to convert a GPS-deformation volume estimate into feet of WTE
decline, so it remains background motivation rather than an input to this
arithmetic. The Sen's-slope-based observed-decline numbers above are what
actually answers "express it against an observed decline."

## 5. Lake-reach verification (`05_lake_reach_verification/`)

Script: `notebooks/round1_lake_reach_verification.py`

Checked `gsl_lake.shp` against `gslb_stream.shp` (GEOGLOWS reaches, field
`LINKNO`) for all 275 retained wells (`well_reach_relationships.csv`) and all
5 terminal gages (reach via `COMID_v2` in `gsl_nwm.csv`).

- **Well–gage pairing claim: VERIFIED.** 0 of 275 retained wells are paired to
  a reach that intersects the lake polygon. Steps 1–3 and 6 are unaffected —
  this part of the letter can be stated as fact.
- **Terminal-gage claim: NOT LITERALLY TRUE as currently worded.** Bear
  River's own outlet reach (LINKNO 710640970, also the catchment ID used
  throughout for well pairing) does intersect the lake polygon — about 13% of
  its length lies inside it. This is expected, not a bug: Bear River
  discharges into Bear River Bay, part of the Great Salt Lake, so its terminal
  reach necessarily meets the lake it drains to. It's a different phenomenon
  from the routing artifact Reviewer 2 asked about (unrelated reaches drawn
  crossing *through* the lake in Figs. 3/10). No well is paired to this reach,
  so nothing downstream of the pairing step is affected.

**Action needed:** reword `reviewer_2_response.md`'s blanket claim ("no
terminal gage sits on such a reach") to something precise — e.g., no retained
well is paired to a lake-crossing reach, and Bear River's own terminal reach
meeting Bear River Bay is the expected behavior of a lake-draining outlet, not
a pairing error. Don't submit the current wording unchanged.

---

## What's still not done

- Figures 14, 16, 17, 18 have not been regenerated with the new basin-sum
  number (item 2) — plotting work, not run here.
- The ten-well catchment threshold is still a manual choice (these 4 gages
  happen to clear it), not an enforced config/code parameter.
- No R1/R3 wording has been edited yet — these are numbers only.
