import pandas as pd
import numpy as np
import os
import ta # Technical Analysis library

# --- Configuration ---
# All file paths assume execution from the project root directory
RAW_DATA_PATH_TEMPLATE = 'data/raw/BTC_USD_Coinbase_raw_{timeframe}.csv'
PROCESSED_DATA_PATH_TEMPLATE = 'data/processed/BTC_USD_Coinbase_processed_{timeframe}.csv'

# Timeframes to load directly (now including 15m, 30m, 6h)
API_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '6h', '1d'] 

# Custom timeframes to be generated from the 1m data (Resampling)
CUSTOM_TIMEFRAMES = ['10m'] 

# Standard Window sizes for feature calculation
SHORT_WINDOW = 10
LONG_WINDOW = 30
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20

# Look-ahead window for the target variable (3 candles ahead)
TARGET_LOOKAHEAD = 3 
# Target Threshold (e.g., a 0.1% change is considered a move)
TARGET_THRESHOLD = 0.001 


def generate_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all technical indicators and the target variable.
    Dynamically adjusts window sizes for very small datasets.
    """
    
    # --- 0. Dynamic Window Adjustment ---
    # Keeping this safety check for very short datasets.
    if len(df) < 50: 
        print(f"   -> Reducing feature windows for small dataset (len={len(df)})")
        short_w = 5
        long_w = 10
        rsi_w = 5
        bb_w = 10
        roc_p = 3
    else:
        # Use standard windows for all other larger datasets
        short_w = SHORT_WINDOW
        long_w = LONG_WINDOW
        rsi_w = RSI_WINDOW
        bb_w = BOLLINGER_WINDOW
        roc_p = 5

    # --- 1. Technical Indicators (Features) ---
    
    # 1.1 Moving Averages (MA)
    df[f'MA_{short_w}'] = df['close'].rolling(window=short_w).mean()
    df[f'MA_{long_w}'] = df['close'].rolling(window=long_w).mean()

    # 1.2 Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_w).rsi()
    
    # 1.3 Moving Average Convergence Divergence (MACD)
    macd_indicator = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    
    # 1.4 Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=bb_w)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Band_Width'] = bollinger.bollinger_wband()

    # 1.5 Rate of Change (Momentum)
    df['ROC'] = df['close'].pct_change(periods=roc_p) 
    
    # --- 2. Target Variable (The value to predict) ---
    
    # Calculate the price change 'TARGET_LOOKAHEAD' periods into the future
    future_close = df['close'].shift(-TARGET_LOOKAHEAD)
    df['Future_Return'] = (future_close - df['close']) / df['close']
    
    # Create the categorical target (UP/DOWN/FLAT)
    df['Target'] = np.select(
        [
            df['Future_Return'] > TARGET_THRESHOLD,  # Move up by at least threshold
            df['Future_Return'] < -TARGET_THRESHOLD  # Move down by at least threshold
        ],
        [
            'UP', 
            'DOWN'
        ],
        default='FLAT' # Default to FLAT if change is within the threshold
    )

    # --- 3. Cleanup ---
    df.dropna(inplace=True) 
    df.drop(columns=['Future_Return'], inplace=True)
    
    return df


def resample_and_process(df_source: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples the source data to a higher timeframe and processes it."""
    
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Use 'T' for minute-based resampling to avoid FutureWarning
    if timeframe.endswith('m') and timeframe != '1m':
        resample_freq = timeframe.replace('m', 'T')
    else:
        resample_freq = timeframe
    
    resampled_df = df_source.resample(resample_freq).agg(ohlcv_dict).dropna()
    resampled_df = generate_ohlc_features(resampled_df)
    
    return resampled_df


def process_raw_data():
    """Main function to load, process, and save all dataframes."""
    
    print("--- Starting Data Processing (src/process.py) ---")
    
    os.makedirs(os.path.dirname('data/processed/'), exist_ok=True)
    
    # --- Load Raw Data Sources ---
    raw_1m_path = RAW_DATA_PATH_TEMPLATE.format(timeframe='1m')
    
    if not os.path.exists(raw_1m_path):
        print(f"🛑 ERROR: Required raw 1m data not found at {raw_1m_path}")
        print("Please ensure 'python src/fetch.py' ran successfully.")
        return
        
    df_1m_raw = pd.read_csv(raw_1m_path)
    df_1m_raw['timestamp'] = pd.to_datetime(df_1m_raw['timestamp'])
    df_1m_raw.set_index('timestamp', inplace=True) 
    
    all_timeframes_to_process = API_TIMEFRAMES + CUSTOM_TIMEFRAMES
    
    for tf in all_timeframes_to_process:
        processed_df = pd.DataFrame() 
        
        if tf in API_TIMEFRAMES:
            # --- Case 1: Load and process API-fetched timeframes (1m, 5m, 15m, 30m, 1h, 6h, 1d) ---
            print(f"\n[1/2] Processing API Timeframe: {tf}")
            if tf == '1m':
                df_to_process = df_1m_raw.copy()
            else:
                raw_path = RAW_DATA_PATH_TEMPLATE.format(timeframe=tf)
                
                if not os.path.exists(raw_path):
                    # This check is crucial since you haven't run fetch.py with the new list yet
                    print(f"🛑 ERROR: Required raw data for {tf} not found at {raw_path}. Skipping. (You need to run fetch.py first.)")
                    continue
                    
                df_to_process = pd.read_csv(raw_path)
                df_to_process['timestamp'] = pd.to_datetime(df_to_process['timestamp'])
                df_to_process.set_index('timestamp', inplace=True)
                
            processed_df = generate_ohlc_features(df_to_process)
            
        else:
            # --- Case 2: Generate and process custom timeframes (Only 10m remains) ---
            
            df_source = df_1m_raw
            print(f"\n[2/2] Generating and Processing Custom Timeframe: {tf} (Source: 1m)")
                
            print(f"   -> Resampling raw data to custom {tf}...")
            processed_df = resample_and_process(df_source, tf)

        # Save and summarize the final processed dataframe
        final_path = PROCESSED_DATA_PATH_TEMPLATE.format(timeframe=tf)
        
        if processed_df.empty:
            print(f"❌ Could not generate final dataset for {tf}. DataFrame is empty after processing.")
        else:
            if processed_df.index.isna().any():
                 print(f"⚠️ Warning: NaT values found in the index for {tf}. Skipping save.")
                 continue

            processed_df.to_csv(final_path, index=True) 
            
            # Print summary
            start_date = processed_df.index.min().strftime('%Y-%m-%d')
            end_date = processed_df.index.max().strftime('%Y-%m-%d')
            print(f"✅ Saved final dataset for {tf} to: {final_path}")
            print(f"   Shape: {processed_df.shape}")
            print(f"   Date Range: {start_date} to {end_date}")
            print(f"   Target Distribution: \n{processed_df['Target'].value_counts(normalize=True).mul(100).round(2)}%")


if __name__ == '__main__':
    process_raw_data()
    print("\n--- All Data Processing Complete ---")
    print("You now have 8 potential ready-to-train datasets.")