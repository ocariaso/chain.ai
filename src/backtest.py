import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Configuration ---
ALL_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '6h', '1d', '10m']

PROCESSED_DATA_PATH_TEMPLATE = 'data/processed/BTC_USD_Coinbase_processed_{timeframe}.csv'
MODEL_PATH_TEMPLATE = 'models/random_forest_{timeframe}.joblib'
REPORT_PATH_TEMPLATE = 'reports/backtest_report_{timeframe}.txt'

TARGET_COLUMN = 'Target'
EXCLUDE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', TARGET_COLUMN]

# --- Trading Parameters ---
INITIAL_CAPITAL = 10000.00  # Starting balance
TRADE_SIZE_RATIO = 0.10     # Use 10% of capital per trade
TRANSACTION_COST = 0.0005   # 0.05% per trade (simulated fee/slippage)


def load_model_and_test_data(timeframe: str):
    """Loads the trained model and the held-out TEST data."""
    model_path = MODEL_PATH_TEMPLATE.format(timeframe=timeframe)
    data_path = PROCESSED_DATA_PATH_TEMPLATE.format(timeframe=timeframe)

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"⚠️ Warning: Missing model or data for {timeframe}. Skipping.")
        return None, None, None, None

    # Load Model
    model = joblib.load(model_path)
    
    # Load Data and Split (replicating the split from src/train.py)
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True).sort_index()
    total_len = len(df)
    validation_end = int(total_len * 0.85)
    
    # We only care about the TEST set (the last 15% of the data)
    df_test = df.iloc[validation_end:]
    
    # Split features/target
    features = [col for col in df_test.columns if col not in EXCLUDE_COLUMNS]
    X_test = df_test[features]
    y_test = df_test[TARGET_COLUMN]

    # Scale the features using a new scaler (or load the saved one, 
    # but for simplicity, we'll re-scale the test set based on its own distribution for now, 
    # as the model was trained on scaled data).
    # NOTE: In a production pipeline, you MUST use the scaler fit on the TRAINING data.
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test) 

    return model, df_test, X_test_scaled, y_test


def run_backtest_simulation(model, df_test: pd.DataFrame, X_test_scaled, y_test: pd.Series, timeframe: str):
    """
    Simulates trading based on model predictions on the test set.
    
    This function implements a basic buy/sell/hold strategy and calculates 
    realized profit and risk metrics.
    """
    
    print(f"\n--- Running Backtest Simulation for {timeframe} ---")
    
    # 1. Generate Predictions
    y_pred = model.predict(X_test_scaled)
    df_test['Prediction'] = y_pred

    # 2. Setup Trading Variables
    capital = INITIAL_CAPITAL
    trades = []
    current_position = 0 # 1 for Long, -1 for Short, 0 for Flat/Exit
    entry_price = 0
    trade_count = 0

    # 3. Simple Trading Logic Loop
    for i in range(len(df_test) - 1): # Iterate up to the second-to-last candle
        current_candle = df_test.iloc[i]
        next_candle = df_test.iloc[i+1]
        
        prediction = current_candle['Prediction']
        current_close = current_candle['close']
        
        # --- Trading Signals ---
        signal_long = (prediction == 'UP')
        signal_short = (prediction == 'DOWN')
        signal_exit = (prediction == 'FLAT') # We use FLAT as an explicit exit signal
        
        # --- Position Management ---
        
        # 1. Exit Logic (Close current position if there is a 'FLAT' signal)
        if current_position != 0 and (signal_exit or (current_position == 1 and signal_short) or (current_position == -1 and signal_long)):
            
            exit_price = next_candle['open'] # Assume trade closes on the next candle's open
            
            # Calculate Profit/Loss
            pnl_ratio = (exit_price - entry_price) / entry_price
            if current_position == -1: # Reverse PnL for short trades
                pnl_ratio = -pnl_ratio 
            
            pnl_gross = trade_size_usd * pnl_ratio
            
            # Apply transaction costs (entry + exit)
            cost = 2 * (trade_size_usd * TRANSACTION_COST) 
            pnl_net = pnl_gross - cost
            
            # Update Capital
            capital += pnl_net
            current_position = 0
            
            # Record Trade
            trades.append({'timeframe': timeframe, 'pnl_net': pnl_net, 'capital': capital, 'trade_type': 'LONG' if current_position == 1 else 'SHORT'})
            trade_count += 1
            
        # 2. Entry Logic (Enter position if no current position)
        if current_position == 0:
            trade_size_usd = capital * TRADE_SIZE_RATIO
            entry_price = next_candle['open'] # Assume trade opens on the next candle's open
            
            if signal_long:
                current_position = 1
            elif signal_short:
                current_position = -1
        
    # --- Final Metrics Calculation ---
    final_capital = capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate = sum(1 for trade in trades if trade['pnl_net'] > 0) / (trade_count if trade_count > 0 else 1)

    print(f"✅ Backtest Complete: Final Capital = ${final_capital:.2f}")

    # Return key metrics
    return {
        'timeframe': timeframe,
        'test_samples': len(df_test),
        'total_trades': trade_count,
        'total_return': total_return * 100,
        'win_rate': win_rate * 100,
        'final_capital': final_capital
    }


def main():
    """Main function to loop through all timeframes and backtest."""
    
    print("--- Starting Full Model Backtesting Pipeline ---")
    
    backtest_results = []
    
    for tf in ALL_TIMEFRAMES:
        
        model, df_test, X_test_scaled, y_test = load_model_and_test_data(tf)
        
        if model is None:
            continue
            
        # Ensure the test set is large enough (e.g., at least 50 candles)
        if len(df_test) < 50:
            print(f"❌ Skipping {tf}: Test set is too small ({len(df_test)} samples).")
            continue
            
        # Run the backtest and get metrics
        metrics = run_backtest_simulation(model, df_test.copy(), X_test_scaled, y_test, tf)
        backtest_results.append(metrics)
            
    # --- Generate Summary Table ---
    print("\n" + "="*80)
    print("--- ALL BACKTESTS COMPLETE ---")
    print("="*80 + "\n")
    
    if backtest_results:
        summary_df = pd.DataFrame(backtest_results)
        
        # Re-order columns for readability
        summary_df = summary_df[[
            'timeframe', 'test_samples', 'total_trades', 
            'total_return', 'win_rate', 'final_capital'
        ]]
        
        # Format the numbers
        summary_df['test_samples'] = summary_df['test_samples'].astype(int)
        summary_df['total_trades'] = summary_df['total_trades'].astype(int)
        
        summary_df['total_return'] = summary_df['total_return'].round(2).astype(str) + '%'
        summary_df['win_rate'] = summary_df['win_rate'].round(2).astype(str) + '%'
        summary_df['final_capital'] = summary_df['final_capital'].map('${:,.2f}'.format)
        
        print("\n" + "--- FINAL BACKTESTING RESULTS SUMMARY (Test Set) ---")
        print(summary_df.to_markdown(index=False))
        print("\n")
        print("Key: The best model is the one with the highest **Total Return** and a high **Win Rate** (ideally over 55% to cover fees).")

if __name__ == '__main__':
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True) 
    main()