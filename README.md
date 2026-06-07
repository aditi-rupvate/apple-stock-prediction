
Predicting Apple's weekly stock price using technical indicators and forex macroeconomic data with machine learning

This end-to-end data science project predicts Apple Inc. (AAPL) stock closing price 
one week (5 trading days) ahead using supervised machine learning. What makes this 
project unique is the integration of forex macroeconomic data — the Nominal Broad 
U.S. Dollar Index (DTWEXBGS) from the Federal Reserve — alongside traditional 
technical indicators.

Built with Python | scikit-learn | XGBoost | yfinance | FRED API
=======
# 🍎 Apple Weekly Stock Price Prediction
### Using Machine Learning + Forex Macroeconomic Indicators

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

This is an end-to-end machine learning project that predicts **Apple Inc. (AAPL) stock closing price approximately one week (5 trading days) into the future**.

What makes this project stand out from a standard stock predictor is the integration of **forex macroeconomic data** — the Nominal Broad U.S. Dollar Index (DTWEXBGS) from the Federal Reserve (FRED). Since Apple earns revenue globally, a stronger U.S. dollar reduces the value of international sales when converted back into dollars — directly influencing investor expectations and the stock price.

---

## 🎯 Problem Statement

> *"Can historical stock behaviour combined with forex macroeconomic indicators help estimate Apple's short-term price movement?"*

This is a **supervised regression problem** — the model predicts a continuous numerical value (next week's closing price), not a category.

---

## 📊 Datasets

| Dataset | Source | Description | Rows |
|--------|--------|-------------|------|
| Apple Stock (AAPL) | Yahoo Finance via `yfinance` | Daily Open, High, Low, Close, Volume from 2015 to present | ~2,500+ |
| USD Dollar Index (DTWEXBGS) | Federal Reserve (FRED API) | Nominal Broad U.S. Dollar Index — daily from 2015 to present | ~2,500+ |

The two datasets are merged on the **Date** column. Missing USD values (due to different holiday calendars) are filled using **forward-fill (ffill)**.

---

## ⚙️ Feature Engineering

26 features are engineered across 6 categories:

| Category | Features |
|----------|----------|
| Raw Price | Open, High, Low, Close, Volume |
| Returns & Behaviour | Daily_Return, Price_Range, Open_Close_Diff, High_Low_Pct |
| Moving Averages | MA_5, MA_10, MA_20, MA_50, Volume_MA_10 |
| Volatility | Volatility_5, Volatility_10, Volatility_20 |
| Technical Indicators | RSI_14, MACD, MACD_Signal, MACD_Histogram, EMA_12, EMA_26 |
| Forex / Macro | USD_Index, USD_Return, USD_MA_5, USD_MA_10, USD_MA_20 |

> **RSI** is calculated using **Wilder's Smoothed Moving Average** (EWM with `com=window-1`) — the industry standard, not a simple rolling mean.

---

## 🤖 Models Used

| Model | Type | Scaling |
|-------|------|---------|
| Linear Regression | Linear | ✅ StandardScaler Pipeline |
| Ridge Regression | Linear + L2 Regularisation | ✅ StandardScaler Pipeline |
| Random Forest | Ensemble (Bagging) | ❌ Not needed |
| Gradient Boosting | Ensemble (Boosting) | ❌ Not needed |
| XGBoost | Optimised Boosting | ❌ Not needed |

Linear models are wrapped in a **scikit-learn Pipeline** with `StandardScaler` to handle the large difference in scale between features like Volume (millions) and Daily_Return (tiny decimals).

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Average dollar error in predictions |
| RMSE | Like MAE but penalises large errors more — used to select best model |
| R² Score | How much price variation the model explains (1.0 = perfect) |
| Direction Accuracy | Did the model correctly predict UP or DOWN? |

---

## 📉 Business Signal

After prediction, the percentage change from today's price is converted into a signal:

| Signal | Condition |
|--------|-----------|
| 📈 Bullish | Predicted change > +2% |
| 📉 Bearish | Predicted change < -2% |
| ➡️ Neutral | Between -2% and +2% |

---

## 🗂️ Project Structure

```
apple-stock-prediction/
│
├── apple_weekly_stock_prediction_forex_features.ipynb   # Main notebook
├── .gitignore                                           # Ignores .env and junk files
└── README.md                                            # This file
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/aditi-rupvate/apple-stock-prediction.git
cd apple-stock-prediction
```

### 2. Install dependencies
```bash
pip install yfinance pandas numpy matplotlib scikit-learn xgboost pandas_datareader python-dotenv requests -q
```

### 3. Get a free FRED API key
1. Go to [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account
3. Request an API key (instant)

### 4. Set up your API key

**Option A — Local (.env file):**
Create a `.env` file in the project folder:
```
FRED_API_KEY=your_key_here
```

**Option B — Google Colab or anywhere else:**
When you run the notebook, it will prompt you to enter your key securely (input is hidden).

### 5. Run the notebook
Open `apple_weekly_stock_prediction_forex_features.ipynb` in Jupyter or Google Colab and run all cells from top to bottom.

---

## 🔍 Key Design Decisions

**Why chronological train/test split?**
Time-series data must never be randomly shuffled. A random split would allow future data into the training set, causing data leakage and unreliable results. The first 80% of rows are used for training, the last 20% for testing.

**Why StandardScaler only for linear models?**
Linear models are sensitive to feature scale. Tree-based models split on thresholds and are completely unaffected by scale — so no scaler is needed.

**Why the USD Dollar Index?**
Apple earns revenue in dozens of currencies globally. When a stronger dollar reduces the value of international revenue, it can affect investor expectations and the stock price. This macroeconomic context is what differentiates this project from a standard price-only predictor.

---

## ⚠️ Limitations & Disclaimer

- This project is for **educational and portfolio purposes only**
- It is **not financial advice** and should not be used for real trading decisions
- The model does not account for breaking news, earnings surprises, or fundamental company changes
- High R² when predicting price levels is expected — direction accuracy is the more meaningful metric
- Walk-forward validation would give more robust performance estimates than a single 80/20 split

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — data manipulation
- **numpy** — numerical computing
- **matplotlib** — visualisation
- **scikit-learn** — machine learning models, Pipeline, StandardScaler
- **XGBoost** — gradient boosting
- **yfinance** — Apple stock data
- **requests** — FRED API calls
- **python-dotenv** — secure API key management

---
