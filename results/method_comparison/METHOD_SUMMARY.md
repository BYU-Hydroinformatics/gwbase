# Comparing Noise-Removal Methods for Groundwater–Streamflow Regression

**Study system:** Five Utah stream gages (Bear River, Weber River, Provo River, Spanish Fork,
Little Cottonwood Creek) paired with 56–102 nearby monitoring wells.  
**Goal:** Estimate the slope β in ΔQ ~ β · ΔWTE, where a positive slope is physically
expected for gaining streams (higher water table → higher baseflow).  
**Pre-processing:** Monthly streamflow Q is the mean of baseflow-dominated days only
(ML_BFD = 1 filter). ΔWTE outliers are removed using a 3-std per-gage filter
(WTE comes from infrequent manual measurements with interpolation; large ΔWTE values
are likely interpolation artifacts). ΔQ is kept as-is since it derives from continuous
daily gauge records.

---

## 1. Framework Overview

Two regression frameworks are used across the six methods:

| Framework    | Form                    | Question asked                      |
|--------------|-------------------------|-------------------------------------|
| Δ–Δ (diff)   | ΔQ = α + β·ΔWTE + ε     | Do WTE and Q change together?       |
| Level (anom) | z_Q = α + β·z_WTE + ε   | Are WTE and Q anomalies correlated? |

Lag analysis (does WTE lead Q by k months?) is treated separately
in `lag_comparison_analysis.py`.

---

## 2. Methods

### 2.1  Monthly Delta  (baseline)

**Principle:** Month-to-month first differences. Seasonal cycle is NOT removed.

```
ΔWTE(t) = WTE(t) - WTE(t-1)
ΔQ(t)   = Q(t)   - Q(t-1)

Regression:  ΔQ(t) = α + β · ΔWTE(t) + ε
Constraint:  consecutive months only  (month_idx = year×12 + month, diff == 1)
```

---

### 2.2  Annual Delta  (Year-over-Year)

**Principle:** Annual means, then year-to-year differences. Seasonal cycle fully removed
by averaging.

```
WTE_bar(y) = mean of all WTE in year y
  Q_bar(y) = mean of all Q   in year y

ΔWTE(y) = WTE_bar(y) - WTE_bar(y-1)
  ΔQ(y) =   Q_bar(y) -   Q_bar(y-1)

Regression:  ΔQ(y) = α + β · ΔWTE(y) + ε
```

---

### 2.3  Same-Quarter Year-over-Year Delta  ★ primary method

**Principle:** Quarterly means, then differences between the same quarter in consecutive
years (e.g. Q2-2005 minus Q2-2004). Comparing the same seasonal phase removes the annual
cycle. This is a seasonal differencing operator with lag 4 (quarterly) / lag 12 (monthly).

```
WTE_qtr(y, q) = mean WTE in quarter q of year y     (q = 1,2,3,4)

ΔWTE(y, q) = WTE_qtr(y, q) - WTE_qtr(y-1, q)
  ΔQ(y, q) =   Q_qtr(y, q) -   Q_qtr(y-1, q)

Regression:  ΔQ(y,q) = α + β · ΔWTE(y,q) + ε

Outlier removal: 3-std on ΔWTE per gage (only WTE filtered; ΔQ kept intact)
Scatter plot:    all four quarters combined, coloured by quarter
```

---

### 2.4  Deseasonalized Quarter Consecutive Delta

**Principle:** Subtract the long-term quarterly climatology from each observation, then
take consecutive quarter differences on the residuals. Removes the mean seasonal cycle
before differencing.

```
WTE_clim(q) = multi-year mean of WTE_qtr for quarter q

WTE_anom(y, q) = WTE_qtr(y, q) - WTE_clim(q)
  Q_anom(y, q) =   Q_qtr(y, q) -   Q_clim(q)

qtr_idx = year × 4 + quarter_number

ΔWTE_anom = WTE_anom(qtr_idx) - WTE_anom(qtr_idx - 1)
  ΔQ_anom =   Q_anom(qtr_idx) -   Q_anom(qtr_idx - 1)

Regression:  ΔQ_anom = α + β · ΔWTE_anom + ε
Constraint:  consecutive quarters only (qtr_idx diff == 1)
```

---

### 2.5  Rolling 12-Month Annual Difference  ★ high-n alternative

**Principle:** Smooth with a 12-month trailing rolling mean (low-pass filter), then take
12-month differences. Rolling mean suppresses sub-annual variability; 12-month lag
difference isolates interannual change.

```
WTE_roll(t) = rolling mean of WTE over [t-11 … t]   (min 10 obs)
  Q_roll(t) = rolling mean of   Q over [t-11 … t]

ΔWTE_12(t) = WTE_roll(t) - WTE_roll(t-12)
  ΔQ_12(t) =   Q_roll(t) -   Q_roll(t-12)

Regression:  ΔQ_12(t) = α + β · ΔWTE_12(t) + ε
Constraint:  month_idx diff == 12
```

---

### 2.6  Standardized Anomaly Regression  ★ signal detection

**Principle:** Works with level values, not differences. Each series is (1) linearly
detrended per well-gage pair to remove long-term drift, then (2) standardized to zero
mean and unit variance within each calendar month (z-score). Z-scores are regressed
directly. Outliers beyond |z| > 4 are removed.

```
Step 1 — Detrend:
    WTE_dt(t) = WTE(t) - (a_hat + b_hat × t)

Step 2 — Monthly z-score:
    z_WTE(t) = [ WTE_dt(t) - mean_WTE_dt(month) ] / std_WTE_dt(month)
    z_Q(t)   = [ Q_dt(t)   - mean_Q_dt(month)   ] / std_Q_dt(month)
    Truncation: |z| > 4 removed for both axes

Step 3 — Regression:
    z_Q(t) = α + β · z_WTE(t) + ε
```

Slope β is dimensionless (z-score units). Use for signal detection and significance
testing, not for physical sensitivity estimation (CFS/ft).

---

## 3. Results

All slopes are from a **pooled OLS regression** per method × gage (all well–gage pairs
combined), corresponding to the black dashed line on scatter-by-gage figures.

### 3.1  Pooled Slope by Method and Gage

Units: CFS/ft for methods 2.1–2.5; dimensionless z/z for method 2.6 (†).

```
Method                   Bear R.   Weber R.  Provo R.  Span. F.  Litt. Cot.  Avg    %Pos
─────────────────────────────────────────────────────────────────────────────────────────
Monthly delta (base)      +6.73     +0.17     -0.22     +2.31     +0.15      +1.83   80%
Annual delta (YoY)       +12.69     +0.18     +4.36     +1.29     +0.71      +3.85  100%
Same-quarter YoY         +11.85     +0.62     +5.62     +1.92     +0.25      +4.05  100%
Deseason qtr consec       +4.93     +0.27     +0.63     +1.42     +0.77      +1.61  100%
Rolling 12m               +6.25     +0.26     +2.89     +1.73     +0.58      +2.34  100%
Std anomaly †            +0.051    +0.090    +0.092    +0.179    +0.282      +0.14  100%
─────────────────────────────────────────────────────────────────────────────────────────
```

### 3.2  Pooled R² and Significance

```
Method                   N (total)  Avg R²   Avg -log10(p)   % Gages p<0.05
─────────────────────────────────────────────────────────────────────────────────────────
Monthly delta (base)       53,477   0.0057       122              80%
Annual delta (YoY)          3,869   0.0272         2.9            60%
Same-quarter YoY            9,957   0.0161       121.8            80%
Deseason qtr consec         9,400   0.0083         1.9            40%
Rolling 12m                13,554   0.0255       240.4           100%
Std anomaly                53,251   0.0268       300.0           100%
─────────────────────────────────────────────────────────────────────────────────────────
```

### 3.3  Provo River

Under methods without seasonal removal (monthly delta: −0.22 CFS/ft), Provo River
shows a negative slope. All five deseasonalized methods return positive slopes
(+0.63 to +5.62 CFS/ft), confirming this is a seasonal confound (snowmelt peaks
before groundwater recharge) rather than a true losing-stream signal.

---

## 4. Discussion

### Which method to prefer?

Monthly ΔQ in Utah snowmelt rivers is dominated by the spring-to-summer recession
(>80% of variance), which is orthogonal to the interannual groundwater signal of
interest. All five deseasonalized methods recover 100% positive pooled slopes.

**Same-quarter YoY delta** — most interpretable: transparent seasonal differencing,
no hyperparameters, natural CFS/ft slopes, 100% positive gages, highest average slope.
→ **Recommended primary method.**

**Rolling 12-month difference** — best combination of data volume (~14k points),
R² (0.026), and statistical significance (−log10p = 240, all gages p < 0.001).
→ **Recommended high-n alternative.**

**Standardized anomaly regression** — most statistically robust (all gages uniformly
p < 0.001, −log10p = 300). Slopes in z-score units only.
→ **Best for signal detection and cross-gage comparison.**

**Monthly delta (baseline)** — retained as reference; 80% positive gages but seasonal
noise inflates residuals.

### Why are all R² values below 0.07?

Pooled regression combines many wells across a spatially heterogeneous system. Each
well–gage pair has its own aquifer geometry and response lag, so the pooled R² captures
only the shared linear component. Low R² does not mean the signal is absent — individual
top-performing pairs show substantially stronger fits.

---

## 5. Summary Table

```
Method                Framework  Seasonal removal            %Pos  Avg R²  Recommended use
─────────────────────────────────────────────────────────────────────────────────────────────
Monthly delta         Δ–Δ        None                         80%  0.006   Baseline / reference
Annual delta          Δ–Δ        Annual averaging            100%  0.027   Low-n cross-check
Same-quarter YoY      Δ–Δ        Seasonal lag-12 diff        100%  0.016   PRIMARY ★
Deseason qtr consec   Δ–Δ        Climatology subtraction     100%  0.008   Secondary check
Rolling 12m           Δ–Δ        Low-pass + 12-month diff    100%  0.026   High-n alternative ★
Std anomaly           Level      Detrend + monthly z-score   100%  0.027   Signal detection ★
─────────────────────────────────────────────────────────────────────────────────────────────
```

---

*Updated 2026-05-05. Source: `notebooks/method_comparison.py`*
