import pandas as pd
import numpy as np
import os
import joblib
import ta
from sklearn.preprocessing import StandardScaler
import json 

# --- Configuration ---
# LIST OF ALL TIMEFRAMES TO PROCESS
ALL_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '6h', '1d', '10m']

# Paths
RAW_DATA_PATH_TEMPLATE = 'data/raw/BTC_USD_Coinbase_raw_{timeframe}.csv'
MODEL_PATH_TEMPLATE = 'models/random_forest_{timeframe}.joblib'
# NEW PATH: Where the prediction data will be saved for the bot
OUTPUT_SIGNAL_PATH = 'signal_report.json' 

# NEW PATH: Location to load the dynamic metrics from src/train.py
MODEL_METRICS_PATH = 'models/metrics_summary.json'

# --- MODEL METRICS ---
# REMOVED: Hardcoded MODEL_ACCURACIES dictionary is now replaced by a dynamic load.
# -----------------------

# Feature Calculation Constants (MUST match src/process.py)
SHORT_WINDOW = 10
LONG_WINDOW = 30
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20

# Define features
FEATURES = [
    'MA_10', 'MA_30', 'RSI', 'MACD', 'MACD_Signal', 
    'BB_High', 'BB_Low', 'BB_Middle', 'BB_Band_Width', 'ROC' 
]

# ADDED: Function to dynamically load metrics
def load_model_metrics() -> dict:
    """Loads model performance metrics from the JSON file created by src/train.py."""
    if not os.path.exists(MODEL_METRICS_PATH):
        print(f"⚠️ Warning: Model metrics file not found at {MODEL_METRICS_PATH}. Cannot display confidence.")
        return {}
    try:
        with open(MODEL_METRICS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading metrics JSON from {MODEL_METRICS_PATH}: {e}")
        return {}


def generate_ohlc_features_for_live_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the exact technical features needed for prediction.
    """
    
    df = df.copy()

    # 1.1 Moving Averages (MA)
    df[f'MA_{SHORT_WINDOW}'] = df['close'].rolling(window=SHORT_WINDOW).mean()
    df[f'MA_{LONG_WINDOW}'] = df['close'].rolling(window=LONG_WINDOW).mean()

    # 1.2 Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=RSI_WINDOW).rsi()
    
    # 1.3 Moving Average Convergence Divergence (MACD)
    macd_indicator = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    
    # 1.4 Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=BOLLINGER_WINDOW)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Band_Width'] = bollinger.bollinger_wband()

    # 1.5 Rate of Change (Momentum)
    df['ROC'] = df['close'].pct_change(periods=5) 
    
    # We only need the features from the VERY LAST row for prediction
    return df[FEATURES].dropna().iloc[-1:]


def get_live_prediction(timeframe: str):
    """
    Loads data, calculates features on the latest candle, and predicts the outcome.
    Includes special logic for the 10m custom timeframe.
    """
    model_path = MODEL_PATH_TEMPLATE.format(timeframe=timeframe)
    
    # 1. Load Model
    if not os.path.exists(model_path):
        return 'Model Not Found', None

    model = joblib.load(model_path)
    
    # 2. Load Data (with 10m special case)
    if timeframe == '10m':
        # 10m was generated from 1m, so we load the 1m raw data
        source_timeframe = '1m'
        source_raw_path = RAW_DATA_PATH_TEMPLATE.format(timeframe=source_timeframe)
        
        if not os.path.exists(source_raw_path):
             return 'Source Data Not Found', None
        
        df_raw = pd.read_csv(source_raw_path, index_col='timestamp', parse_dates=True).sort_index()
        
        ohlcv_dict = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }
        df_resampled = df_raw.resample('10min').agg(ohlcv_dict).dropna()
        df_recent = df_resampled.iloc[-100:]
        
    else:
        # Normal API timeframe loading
        raw_path = RAW_DATA_PATH_TEMPLATE.format(timeframe=timeframe)
        if not os.path.exists(raw_path):
            return 'Raw Data Not Found', None
            
        df_raw = pd.read_csv(raw_path, index_col='timestamp', parse_dates=True).sort_index()
        df_recent = df_raw.iloc[-100:]

    # 3. Calculate Features
    X_live = generate_ohlc_features_for_live_data(df_recent)
    
    if X_live.empty:
        return 'Insufficient Data', None
        
    # 4. Scale and Predict
    scaler = StandardScaler()
    X_live_scaled = scaler.fit_transform(X_live)
    
    prediction = model.predict(X_live_scaled)[0]
    
    return prediction, X_live.index[0]


def main():
    """Loops through all timeframes, generates predictions, and prints a final table."""
    
    print("="*60)
    print("--- FULL PROJECT SCOPE PREDICTION REPORT ---")
    print(f"Date of Report Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ADDED: Load metrics dynamically at the start
    loaded_metrics = load_model_metrics()
    
    results = []
    
    for tf in ALL_TIMEFRAMES:
        prediction, timestamp = get_live_prediction(tf)
        
        signal = prediction
        accuracy_str = "N/A" # Default for failed runs
        
        # Convert timestamp to string for JSON serialization
        timestamp_str = str(timestamp) if timestamp is not None else "N/A"
        
        # UPDATED: Get confidence metric (Average Precision) if the prediction was successful
        if prediction in ['UP', 'DOWN', 'FLAT']:
            tf_metrics = loaded_metrics.get(tf, {})
            
            # Calculate the relevant trading confidence: Average of UP and DOWN precision
            up_prec = tf_metrics.get('up_precision', 0.0)
            down_prec = tf_metrics.get('down_precision', 0.0)
            
            if up_prec > 0.0 and down_prec > 0.0:
                confidence = (up_prec + down_prec) / 2
                accuracy_str = f"{confidence * 100:.2f}%"
            else:
                accuracy_str = "Low Confidence/Missing"


        results.append({
            'timeframe': tf, 
            'last_candle': timestamp_str, 
            'signal': signal, 
            'accuracy': accuracy_str,
        })
        
        # Print a simple status during the loop instead of full output
        print(f"Checking {tf}: Status - {'Success' if prediction in ['UP', 'DOWN', 'FLAT'] else 'Failed'}")

    # --- SAVE RESULTS TO JSON FILE ---
    try:
        # We save the JSON file to the project root directory
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', OUTPUT_SIGNAL_PATH)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n[INFO] All signals saved to: {OUTPUT_SIGNAL_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save signal JSON: {e}")


    # Adjusted separator length for the reduced column count
    print("\n" + "="*70)
    print("--- FINAL TRADING SIGNAL SUMMARY ---")
    print("="*70)
    
    # Create the final table using pandas for terminal output
    summary_df = pd.DataFrame(results)
    
    # Rename columns for the final display and set the specific order
    summary_df.columns = [
        'Timeframe', 'Last Candle', 'Raw Signal', 'Confidence (Avg. Prec.)' # Renamed column
    ]
    
    # Ensure 'Last Candle' is string for consistent table width
    summary_df['Last Candle'] = summary_df['Last Candle'].astype(str)
    
    # Use 'pipe' formatting and ensure all columns align left (Markdown default)
    print(summary_df.to_markdown(index=False, numalign="left", stralign="left"))


if __name__ == '__main__':
    main()