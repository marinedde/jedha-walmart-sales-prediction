# 🛒 Walmart Weekly Sales Prediction

> **Jedha Fullstack Data Science** — Supervised Machine Learning  
> Certification CDSD · Linear Regression & Regularization

---

## 📋 Project Overview

The Walmart marketing team wants to predict **weekly sales** for each store based on economic indicators (temperature, fuel price, CPI, unemployment rate).

This is a **supervised regression** task: the target variable `Weekly_Sales` is continuous.

**Dataset:** 150 observations · 8 columns · sourced from a Kaggle competition

---

## 🎯 Objectives

| Step | Description |
|------|-------------|
| Part 1 | EDA + preprocessing pipeline |
| Part 2 | Linear regression baseline |
| Part 3 | Regularized models (Ridge, Lasso) + GridSearchCV |

---

## 📊 Dataset

| Variable | Type | Description |
|----------|------|-------------|
| `Store` | Categorical | Store identifier (1–20) |
| `Date` | Date | Week date → decomposed into Year/Month/Day/DayOfWeek |
| `Weekly_Sales` | **Target** | Weekly sales in USD |
| `Holiday_Flag` | Binary | Holiday week indicator |
| `Temperature` | Numerical | Average temperature (°F) |
| `Fuel_Price` | Numerical | Fuel price (USD) |
| `CPI` | Numerical | Consumer Price Index |
| `Unemployment` | Numerical | Unemployment rate (%) |

---

## ⚙️ Methodology

### Preprocessing
- Dropped rows where `Weekly_Sales` is null (never impute the target variable)
- Feature engineering on `Date` → `Year`, `Month`, `Day`, `DayOfWeek`
- Outlier removal using ±3σ rule on `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
- **scikit-learn Pipeline**: `SimpleImputer(median)` + `StandardScaler` for numericals, `OneHotEncoder` for categoricals
- Train/test split: 80/20 with `random_state=42`

### Models

| Model | RMSE Test | R² Test |
|-------|-----------|---------|
| LinearRegression | 188,478 $ | 0.898 |
| Ridge (α=1) | 216,126 $ | 0.866 |
| Lasso (α=1) | 188,476 $ | 0.898 |
| **Ridge (α=0.01, GridSearch)** | **189,230 $** | **0.897** |

---

## 🔍 Key Findings

**1. Store identity dominates predictions**  
The top coefficients are all `Store_*` variables (±900K$). The model essentially learned the size/volume of each store rather than the economic relationships.

**2. Slight overfitting**  
R² train (0.978) >> R² test (0.898) — gap of 0.08. Acceptable but worth noting.

**3. Regularization not helpful here**  
Optimal Ridge alpha = 0.01 (near-zero regularization). The real issue is the **low data/feature ratio** (104 rows × 29 features), not noisy coefficients.

**4. Dataset limitations**  
With only 131 rows after cleaning, results are not statistically robust. The full Kaggle dataset (421,570 rows) would give very different conclusions.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/jedha-walmart-sales-prediction.git
cd jedha-walmart-sales-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook Walmart_ML_Jedha.ipynb
```

Or open directly in **Google Colab** → File → Upload notebook

---

## 📁 File Structure

```
jedha-walmart-sales-prediction/
│
├── Walmart_ML_Jedha.ipynb        # Main notebook
├── Walmart_Store_sales.csv       # Dataset
└── README.md
```

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-green)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange)

---

## 📚 Context

Part of the **Jedha Fullstack Data Science** certification (CDSD).  
Topic: Supervised Machine Learning — Regression.
