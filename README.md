# Forecasting Monthly Temperature Anomalies with LSTM and Classical Machine Learning

**MSc dissertation — MSc in IT for Business Data Analytics, International Business School (May 2023)**

Author: Alireza Esmaeily Brojerdi · Supervisor: Zsofia Gyarmathy

Full dissertation: [`Esmaeily_Brojerdi_MSc_Thesis_Climate_Anomaly_Forecasting.pdf`](Esmaeily_Brojerdi_MSc_Thesis_Climate_Anomaly_Forecasting.pdf) (77 pages)

---

## What the project does

Monthly **surface temperature anomaly** — the deviation of a month's temperature from a long-run
reference average — is the target variable. The project builds an end-to-end pipeline that cleans a
monthly climate dataset, removes outliers, selects features, benchmarks five classical regression
models, and trains an **LSTM** network that forecasts the anomaly value **three months ahead**.

The motivation is practical rather than climatological: anomaly forecasts feed planning decisions in
agriculture, energy and environmental operations, so the work is framed as a full data-science
pipeline — exploration → preprocessing → feature selection → modelling → evaluation — rather than as
a single model.

## Data

A monthly panel of **1,458 rows / 23 columns** combining temperature-anomaly records, city-level
average temperatures and greenhouse-gas emissions.

| Column | Meaning |
|---|---|
| `date` | end-of-month date of the observation |
| `anomaly_value` | **target** — monthly surface temperature anomaly |
| `upper_95_ci`, `lower_95_ci` | 95 % confidence bounds around the anomaly value |
| `AverageTemperature` | monthly average temperature for a city |
| `AverageTemperatureUncertainty` | reported uncertainty of that average |
| `City`, `Country` | geography of the temperature series |
| `emission` | annual greenhouse-gas emissions (tonnes), broadcast to months |
| + calendar and event features | year, month, day, leap-year flag, natural-event columns |

## Pipeline

**1 · Data quality**

Missing-value and duplicate audit (0 duplicates), dtype correction, `datetime` parsing, removal of
unused columns, and reconciliation of **leap years and 28/29 February** so every year contributes a
comparable monthly series → 1,458 → **1,429 rows**.

**2 · Exploration**

Correlation heatmap, **autocorrelation and lag analysis** of the anomaly series, and distribution /
time-series plots per city and country.

**3 · Scaling**

`MinMaxScaler` and `StandardScaler` compared; min–max scaling carried forward for the sequence model.

**4 · Outlier handling — three methods compared**

Z-score (|z| > 2), **IQR** (1.5 × IQR fences) and the robust **modified Z-score**
(0.6745 · (x − median) / MAD), which is less distorted by extreme values because it uses the median
and MAD instead of mean and standard deviation. The modified Z-score set was used for modelling →
**1,308 rows**.

**5 · Feature selection**

Random-Forest feature importance on the cleaned frame (`upper_95_ci` ≈ 0.60, `lower_95_ci` ≈ 0.32
dominate; calendar flags ≈ 0).

**6 · Models**

80/20 train–test split, MSE and R² on a held-out set.

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 2.29 × 10⁻⁵ | 0.9997 |
| Random Forest Regression | 3.32 × 10⁻⁵ | 0.9996 |
| Gradient Boosting Regression | 4.75 × 10⁻⁵ | 0.9994 |
| Decision Tree Regression | 1.35 × 10⁻⁴ | 0.9983 |
| Neural Network Regression (MLP) | 2.88 × 10¹⁴ | diverged — unscaled inputs |

**7 · LSTM (Keras / TensorFlow)**

Sequences of **3 time steps** (three preceding months) over the scaled feature matrix, an
`LSTM(64)` layer with a dense regression head, MSE loss, Adam optimiser, **early stopping**
(patience 5) on a 20 % validation split, 50 epochs max, batch size 32 — then a three-month-ahead
forecast from the final sequence, plotted against the actual series.

## Honest reading of the results

The near-perfect R² of the classical models is **not** evidence of forecasting skill. The feature
matrix still contains `upper_95_ci` and `lower_95_ci`, which are constructed *around* the target:
they bracket `anomaly_value` by definition, so a model that learns their midpoint reproduces the
target almost exactly. Random-Forest importance confirms this — those two columns carry ~92 % of the
total importance.

A clean re-run would drop both confidence-interval columns and re-benchmark on genuinely predictive
features (lagged anomalies, emissions, city temperature), and would evaluate with a **chronological**
split rather than a random one, since a random split on a time series lets future months inform the
training set. The LSTM section already uses a chronological 80/20 split; the classical benchmark does
not.

Keeping this limitation visible was the most useful outcome of the project: it is the difference
between a metric that looks good and a metric that means something.

## Tools

`Python` · `pandas` · `NumPy` · `scikit-learn` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn` ·
Jupyter / Google Colab

## Repository contents

```
README.md                                              this file
Esmaeily_Brojerdi_MSc_Thesis_Climate_Anomaly_Forecasting.pdf
                                                       full dissertation, incl. all code and figures
```

The dissertation is literate-programming style: every step is shown as code, output and commentary,
so the notebook logic can be read directly from the PDF.

## Author

**Alireza Esmaeily Brojerdi** — Data Analyst, Munich

MSc IT for Business Data Analytics · BSc Information Technology Engineering

More projects: [github.com/AESMAEILY](https://github.com/AESMAEILY)
