# chain.ai
a Machine Learning system for building and evaluating algorithmic cryptocurrency trading strategies. It automates the entire process, including multi-timeframe data acquisition, 
feature engineering, and training Random Forest models to accurately predict market direction. The system uses strict time-based evaluation to maintain data integrity, and while 
it currently focuses on Bitcoin (BTC/USD) for live signal generation, it is actively under development with plans to integrate support for other altcoins soon.

## Features
#### Current (Implemented):
- **Multi-Timeframe Analysis**: Supports concurrent analysis and modeling across 8 timeframes (including API-fetched `1m`, `5m`, `15m`, `30m`, `1h`, `6h`, `1d`, and a custom-resampled `10m` chart).
- **Incremental Data Management**: Uses **CCXT** to safely fetch and incrementally update historical OHLCV data from **Coinbase**, featuring robust de-duplication logic to maintain dataset integrity.
- **Advanced Feature Engineering**: Automatically generates a comprehensive suite of Technical Analysis (TA) indicators (`RSI`, `MACD`, `Bollinger Bands`, `MA`, `ROC`) vital for model training.
- **Time-Series Integrity**: Employs a critical **time-based data split** (70% Train, 15% Validation, 15% Test) across all models to eliminate look-ahead bias and ensure reliable evaluation.
- **Specialized Model Training**: Trains dedicated Random Forest Classifiers for each of the 8 timeframes, utilizing class weighting to better handle imbalanced prediction targets (`UP`/`DOWN`/`FLAT`).
- **Rigorous Backtesting**: Separately simulates trading performance on the unseen **Test Set** with realistic parameters, including initial capital, trade size, and **0.05% transaction costs**.
- **Actionable Signal Generation**: Produces a live, machine-readable `signal_report.json` file containing a prediction and a **Confidence Score** (based on the model's UP/DOWN prediction precision) for external trading bot consumption.
### Future (Planned):
- trading bot
- gui
- Others are to be announced...

## Setup and Running the Program
### 1. Create and active virtual environment 
```sh
python -m venv .venv
```
Active
```sh
.venv\Scripts\activate
```
### 2. Update pip
```sh
python.exe -m pip install --upgrade pip
```
### 3. Install dependencies 
```sh
pip install -r requirements.txt
```
### 4. Run pipeline
```sh
python main.py
```

## Folder Structure 
```
[chain.ai/]
├── main.py  #Orchestrates the entire workflow (Fetch -> Process -> Train -> Predict).
├── signal_report.json  #The machine-readable file containing the latest trading signals and confidence scores for the external bot.   
├── data/ 
│   ├── raw/ #Incremental historical OHLCV data fetched directly from Coinbase via CCXT.
│   └── processed/ #DataFrames containing technical indicators and the engineered prediction target.
├── models/ #Saved joblib files for each timeframe, plus a JSON summary of training metrics.
├── reports/ #Detailed classification reports and confusion matrices for the model training phase.
└── src/ #All core logic for data, modeling, and prediction.
```
