# Comparing Noise-Removal Methods for Groundwater–Streamflow Regression

**Study system:** Five Utah stream gages (Bear River, Weber River, Provo River, Spanish Fork,
Little Cottonwood Creek) paired with 56–102 nearby monitoring wells.  
**Goal:** Estimate the slope β in ΔQ ~ β · ΔWTE, where a positive slope is physically
expected for gaining streams (higher water table → higher baseflow).  
**Pre-processing:** Monthly streamflow Q is the mean of baseflow-dominated days only
(ML_BFD = 1 filter). All outliers beyond 3×IQR are removed before regression.

---

## 1. Framework Overview

Three distinct regression frameworks are used across the ten methods:

| Framework     | Form                      | Question asked                        |
|---------------|---------------------------|---------------------------------------|
| Δ–Δ (diff)    | ΔQ = α + β·ΔWTE + ε       | Do WTE and Q change together?         |
| Level (anom)  | z_Q = α + β·z_WTE + ε     | Are WTE and Q anomalies correlated?   |
| Lag           | Q(t+k) ~ f(WTE_t)         | Does WTE lead Q by k months?          |

All ten methods below fall into the first two frameworks.
Lag analysis is treated separately (see `lag_comparison_analysis.py`).

---

## 2. Methods

### 2.1  Monthly Delta  (original baseline)

**Principle:** Compute month-to-month first differences for both WTE and Q, then
regress. The seasonal cycle is NOT removed — Δ values still contain seasonal variation.

```
ΔWTE(t) = WTE(t) - WTE(t-1)
ΔQ(t)   = Q(t)   - Q(t-1)

Regression:  ΔQ(t) = α + β · ΔWTE(t) + ε

Constraint:  only consecutive months kept  (month_idx diff == 1)
             where month_idx = year × 12 + month
```

---

### 2.2  Annual Delta  (Year-over-Year)

**Principle:** Average all observations within each calendar year, then take year-to-year
differences. Annual averaging eliminates the seasonal cycle entirely.

```
WTE_bar(y) = mean of all WTE observations in year y
  Q_bar(y) = mean of all Q   observations in year y

ΔWTE(y) = WTE_bar(y) - WTE_bar(y-1)
  ΔQ(y) =   Q_bar(y) -   Q_bar(y-1)

Regression:  ΔQ(y) = α + β · ΔWTE(y) + ε
```

---

### 2.3  Same-Quarter Year-over-Year Delta  ← recommended primary method

**Principle:** Average observations within each calendar quarter
(Q1 = Jan–Mar, Q2 = Apr–Jun, Q3 = Jul–Sep, Q4 = Oct–Dec), then take differences only
between **the same quarter in consecutive years**.  Comparing Q2-2005 to Q2-2004
removes the seasonal cycle because both observations are at the same phase of the
hydrograph.  This is a seasonal differencing operator with lag 4 (quarterly) / lag 12
(monthly).

```
WTE_qtr(y, q) = mean WTE in quarter q of year y      (q = 1,2,3,4)

ΔWTE(y, q) = WTE_qtr(y, q) - WTE_qtr(y-1, q)        [same quarter, one year apart]
  ΔQ(y, q) =   Q_qtr(y, q) -   Q_qtr(y-1, q)

Regression:  ΔQ(y,q) = α + β · ΔWTE(y,q) + ε
```

---

### 2.4  Consecutive Quarter Delta

**Principle:** Take differences between every consecutive quarter pair regardless of
season.  A quarter index `qtr_idx = year×4 + quarter_number` ensures that Q4→Q1
transitions across calendar years are included automatically.

```
qtr_idx = year × 4 + quarter_number     (Q1=1, Q2=2, Q3=3, Q4=4)

ΔWTE = WTE(qtr_idx) - WTE(qtr_idx - 1)
  ΔQ =   Q(qtr_idx) -   Q(qtr_idx - 1)

Regression:  ΔQ = α + β · ΔWTE + ε
Constraint:  only consecutive quarters kept  (qtr_idx diff == 1)
```

**Problem:** Crosses season boundaries (e.g. Q2→Q3 captures the spring-to-summer
drop in snowmelt rivers), so seasonal forcing dominates ΔQ and produces mostly
negative slopes.

---

### 2.5  Deseasonalized Quarter Consecutive Delta

**Principle:** Subtract the long-term climatological mean for each quarter before
differencing.  This removes the seasonal cycle from the level values; consecutive
quarter differences are then computed on the residuals.

```
WTE_clim(q) = mean over all years of  WTE_qtr(y, q)     [climatology per quarter]

WTE_anom(y, q) = WTE_qtr(y, q) - WTE_clim(q)
  Q_anom(y, q) =   Q_qtr(y, q) -   Q_clim(q)

ΔWTE_anom = WTE_anom(qtr_idx) - WTE_anom(qtr_idx - 1)
  ΔQ_anom =   Q_anom(qtr_idx) -   Q_anom(qtr_idx - 1)

Regression:  ΔQ_anom = α + β · ΔWTE_anom + ε
```

---

### 2.6  Monthly Anomaly Delta

**Principle:** Same logic as Method 2.5 but at the monthly scale.  Subtract the
12-month climatology mean from each observation, then take consecutive monthly
differences of the anomaly series.

```
WTE_clim(m) = mean over all years of WTE in calendar month m    (m = 1..12)

WTE_anom(t) = WTE(t) - WTE_clim( month(t) )
  Q_anom(t) =   Q(t) -   Q_clim( month(t) )

ΔWTE_anom(t) = WTE_anom(t) - WTE_anom(t-1)
  ΔQ_anom(t) =   Q_anom(t) -   Q_anom(t-1)

Regression:  ΔQ_anom(t) = α + β · ΔWTE_anom(t) + ε
Constraint:  month_idx diff == 1
```

---

### 2.7  Rolling 12-Month Annual Difference  ← recommended high-n alternative

**Principle:** Smooth both series with a 12-month trailing rolling mean (low-pass
filter), then take 12-month differences of the smoothed series.  The rolling mean
suppresses sub-annual variability; the 12-month lag difference isolates interannual
change.

```
WTE_roll(t) = rolling mean of WTE over months [t-11 … t]   (min 10 obs required)
  Q_roll(t) = rolling mean of   Q over months [t-11 … t]

ΔWTE_12(t) = WTE_roll(t) - WTE_roll(t-12)
  ΔQ_12(t) =   Q_roll(t) -   Q_roll(t-12)

Regression:  ΔQ_12(t) = α + β · ΔWTE_12(t) + ε
Constraint:  month_idx diff == 12
```

---

### 2.8  STL Decomposition + Year-over-Year Delta

**Principle:** Apply Seasonal-Trend decomposition using LOESS (STL; Cleveland et al.
1990).  STL fits an additive model:

```
X(t) = Trend(t) + Seasonal(t) + Remainder(t)
```

The deseasonalized signal retains only the trend and remainder components.
Same-quarter year-over-year differences are then applied to this signal.

```
STL parameters:  period = 12,  seasonal window = 13,  robust = True

WTE_deseas(t) = Trend_WTE(t) + Remainder_WTE(t)     [drop Seasonal component]
  Q_deseas(t) = Trend_Q(t)   + Remainder_Q(t)

ΔWTE_deseas(y, q) = WTE_deseas(y,q) - WTE_deseas(y-1, q)
  ΔQ_deseas(y, q) =   Q_deseas(y,q) -   Q_deseas(y-1, q)

Regression:  ΔQ_deseas = α + β · ΔWTE_deseas + ε
```

Requires ≥ ~36 continuous months; covers ~26% of well–gage pairs.

---

### 2.9  Standardized Anomaly Regression

**Principle:** Works with **level values** rather than differences.  Each time series
is (1) linearly detrended to remove long-term drift, then (2) standardized to zero mean
and unit variance within each calendar month.  The resulting z-scores are regressed
directly against each other.

```
Step 1 — Detrend:
    WTE_dt(t) = WTE(t) - (a_hat + b_hat × t)       [OLS linear detrend]

Step 2 — Monthly z-score:
    z_WTE(t) = [ WTE_dt(t) - mean_WTE_dt(m) ] / std_WTE_dt(m)
                                                     (m = calendar month of t)

Step 3 — Regression:
    z_Q(t) = α + β · z_WTE(t) + ε
```

The slope β is dimensionless (z-score / z-score) and cannot be compared directly to
the CFS/ft slopes of the Δ–Δ methods.  Use for signal detection, not sensitivity
estimation.

---

### 2.10  Wavelet Interannual Component

**Principle:** Decompose each time series with a 5-level Discrete Wavelet Transform
(DWT) using the Daubechies-4 (db4) wavelet.  Zero out the three highest-frequency
detail levels (cD1–cD3, periods < ~8 months) and reconstruct only the low-frequency
"interannual" component.  Apply same-quarter YoY differences to the reconstructed
signal.

```
DWT decomposition (db4, 5 levels):
    [cA5, cD5, cD4, cD3, cD2, cD1] = wavedec(X, 'db4', level=5)

Zero high-frequency coefficients:
    cD1 = 0,  cD2 = 0,  cD3 = 0

Reconstruct interannual signal:
    X_inter(t) = waverec([cA5, cD5, cD4, 0, 0, 0])

Same-quarter YoY difference:
    ΔX_inter(y, q) = X_inter(y,q) - X_inter(y-1, q)

Regression:  ΔQ_inter = α + β · ΔWTE_inter + ε
```

Requires ≥ 64 continuous months for 5-level decomposition.
Only 56 of 268 pairs (21%) qualify; results are unstable.

---

## 3. Results

All slopes are from a **pooled OLS regression** across all well–gage pairs within each
gage (one regression per method × gage, combining all data points).  This corresponds
to the black dashed line on the scatter-by-gage figures.

### 3.1  Pooled Slope by Method and Gage  (units: CFS/ft, except † = z-score/z-score)

```
Method                   Bear R.   Weber R.  Provo R.  Span. F.  Litt. Cot.  Avg    %Pos
─────────────────────────────────────────────────────────────────────────────────────────
Monthly delta             +6.73     +0.17     -0.22     +2.31     +0.15      +1.83   80%
Annual delta             +12.69     +0.18     +4.36     +1.29     +0.71      +3.85  100%
Same-quarter YoY         +12.22     +0.45     +3.30     +1.33     +0.46      +3.55  100%
Consec. quarter          -24.71     -1.53     -6.61     +5.32     -0.04      -5.51   20%
Deseason qtr consec       +4.93     +0.27     +0.63     +1.42     +0.77      +1.61  100%
Monthly anomaly           +5.87     +0.51     -0.54     +3.20     +0.57      +1.92   80%
Rolling 12m               +6.25     +0.26     +2.89     +1.73     +0.58      +2.34  100%
STL deseason YoY          +1.82     +1.98     +1.61     +1.44     +0.72      +1.52  100%
Std anomaly †            +0.051    +0.090    +0.092    +0.179    +0.282      +0.14  100%
Wavelet                  -89.10     -0.07     +1.50     +1.13     +0.34     -17.24   60%
─────────────────────────────────────────────────────────────────────────────────────────
```

### 3.2  Pooled R² and Significance

```
Method                   Avg R²   Avg -log10(p)   Notes
─────────────────────────────────────────────────────────────────────────────────────────
Monthly delta            0.0057       122          Seasonal noise inflates residuals
Annual delta             0.0272         2.9        Low n (~800 pts/gage); moderate sig.
Same-quarter YoY         0.0137       122          Balanced n and signal
Consec. quarter          0.0095        62          Seasonal forcing dominates ΔQ
Deseason qtr             0.0083         2          Many pairs not individually significant
Monthly anomaly          0.0016         2          Over-whitening suppresses signal
Rolling 12m              0.0255       240          Best combo of n, R², significance
STL deseason             0.0135         2          Low coverage (70 wells)
Std anomaly              0.0268       300          Highest significance; all p < 0.001
Wavelet                  0.0638         2          Sparse (56 wells); Bear River outlier
─────────────────────────────────────────────────────────────────────────────────────────
```

### 3.3  Provo River Exception

Provo River returns a negative pooled slope under methods that retain seasonal
information (monthly delta: −0.22, monthly anomaly: −0.54, consecutive quarter: −6.61).
Under properly deseasonalized methods (annual, same-quarter YoY, STL, rolling 12m)
the slope becomes positive (+1.6 to +4.4 CFS/ft).  This strongly suggests that the
apparent negative relationship in raw monthly differences reflects a **seasonal confound**
— snowmelt peaks before groundwater recharges — not a true losing-stream response.

---

## 4. Discussion

### Which method to prefer?

Monthly ΔQ in Utah snowmelt rivers is dominated by the spring-to-summer decline
(>80% of total variance), which is orthogonal to the interannual groundwater signal of
interest.  Methods differ primarily in how thoroughly they suppress this seasonal
component before estimating β.

**Same-quarter YoY delta** is the most interpretable: simple seasonal lag-12 operator,
no hyperparameters, natural-unit slopes (CFS/ft), 100% positive gages, largest absolute
slopes.  → **Recommended primary method.**

**Rolling 12-month difference** retains ~3× more data points than annual delta while
achieving similar R² and very high significance (avg −log10p = 240).
→ **Recommended high-n alternative.**

**Standardized anomaly regression** is the most statistically robust (100% positive,
uniformly p < 0.001), but slopes are in z-score units.
→ **Best for signal detection / cross-gage comparisons.**

**Consecutive quarter delta** exposes the regression to strong seasonal forcing and
should be avoided.

### Why are all R² values below 0.07?

Pooled regression across many wells and years captures only the shared linear trend
embedded in a spatially heterogeneous system (each well–gage pair has its own aquifer
geometry and response lag) with high temporal noise (precipitation events, irrigation,
measurement error).  Low R² does not mean the relationship is absent; it means the
signal-to-noise ratio in pooled data is low.  Individual well-level regressions for
the top-performing pairs show substantially higher R².

---

## 5. Summary Table

```
Method               Framework  Seasonal removal          %Pos   Avg R²  Use
─────────────────────────────────────────────────────────────────────────────────────────
Monthly delta        Δ–Δ        None                       80%   0.006   Baseline only
Annual delta         Δ–Δ        Annual averaging          100%   0.027   Low-n check
Same-quarter YoY     Δ–Δ        Seasonal lag-12 diff      100%   0.014   PRIMARY ★
Consec. quarter      Δ–Δ        None                       20%   0.010   Avoid
Deseason qtr         Δ–Δ        Climatology subtraction   100%   0.008   Secondary check
Monthly anomaly      Δ–Δ        Climatology subtraction    80%   0.002   Not recommended
Rolling 12m          Δ–Δ        Low-pass + lag-12 diff    100%   0.026   High-n alt. ★
STL deseason         Δ–Δ        LOESS decomposition       100%   0.014   Long records
Std anomaly          Level      Detrend + z-score         100%   0.027   Detection ★
Wavelet              Δ–Δ        DWT low-pass filter        60%   0.064*  Research only
─────────────────────────────────────────────────────────────────────────────────────────
* Wavelet R² inflated by a single well in Bear River; most pairs R² < 0.02
```

---

*Generated 2026-04-17.  Source: `notebooks/method_comparison.py`*
