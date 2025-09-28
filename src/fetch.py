import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOL = 'BTC/USD'
EXCHANGE_ID = 'coinbase'
# All timeframes to be fetched and maintained as separate files
API_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '6h', '1d'] 

# Path configuration
DATA_DIR = 'data/raw'

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Helper function to convert CCXT timeframe string to milliseconds
def get_timeframe_in_milliseconds(timeframe: str) -> int:
    """Converts a CCXT timeframe string to milliseconds."""
    units = {'m': 60, 'h': 3600, 'd': 86400}
    num = int(timeframe[:-1])
    unit = timeframe[-1]
    
    seconds = num * units.get(unit, 0)
    return seconds * 1000

def get_file_path(timeframe: str) -> str:
    """Generates the file path for a specific timeframe."""
    return os.path.join(DATA_DIR, f"{SYMBOL.replace('/', '_')}_{EXCHANGE_ID}_raw_{timeframe}.csv")

def get_last_timestamp(path: str, timeframe: str) -> int:
    """
    Checks the local CSV file and returns the timestamp (in milliseconds) 
    of the *start* of the candle to fetch next.
    
    CRITICAL FIX: We now return the last saved timestamp, not the next one.
    This guarantees the 'since' is not in the future. We rely on the 
    API to send the last candle again, which we will handle (de-duplicate) 
    by index later.
    """
    
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            # Read the last row of the existing DataFrame
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Use the index to avoid potential errors from reading data columns
            last_timestamp_ms = df.index[-1].value // 10**6 

            # --- FIX APPLIED HERE ---
            # next_timestamp_ms = last_timestamp_ms + timeframe_ms  <-- REMOVED
            # We return the last saved timestamp.
            next_since_ms = last_timestamp_ms
            
            # Convert milliseconds to ISO 8601 string for clean display
            since_dt = datetime.fromtimestamp(next_since_ms / 1000)
            print(f"File found for {timeframe}. Last candle saved was: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}. Fetching from (re-fetching last candle): {since_dt.isoformat()}Z")
            
            return next_since_ms
            
        except Exception as e:
            # If the file is corrupt or empty, fall back to a recent start time
            print(f"Error reading last timestamp from {path}: {e}. Falling back to 7 days ago.")
            pass # Continue to the fallback logic below
    
    # Default start: Fetch enough data to cover a significant history if the file is new/empty
    days_to_fetch = 7 
    if timeframe == '1d':
        days_to_fetch = 365 # Fetch one year for daily data if it's brand new
    
    start_ms = int((datetime.utcnow() - timedelta(days=days_to_fetch)).timestamp() * 1000)
    
    start_dt = datetime.fromtimestamp(start_ms / 1000)
    print(f"File for {timeframe} not found. Starting full fetch from: {start_dt.isoformat()}Z")
    
    return start_ms

def fetch_and_append_ohlcv_update(exchange, symbol: str, timeframe: str):
    """
    Fetches new OHLCV data starting from the last saved timestamp for a specific timeframe,
    and SAFELY APPENDS it to the existing file.
    """
    path = get_file_path(timeframe)
    next_since_ms = get_last_timestamp(path, timeframe)
    
    # Fetch new data (limit=None means fetch all available until now)
    try:
        new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=next_since_ms, limit=None)
    except Exception as e:
        print(f"!!! Error fetching {timeframe} data from API: {e}")
        return

    if not new_ohlcv:
        print(f"No new {timeframe} data to append.")
        return

    # Convert the new data to a DataFrame
    df_new = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
    df_new = df_new.set_index('timestamp')

    # --- DE-DUPLICATION LOGIC (to handle re-fetching the last candle) ---
    # 1. Load the existing file to get all timestamps
    if os.path.exists(path):
        df_existing = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        # 2. Filter the new data, keeping only rows whose index is NOT in the existing index
        df_to_append = df_new[~df_new.index.isin(df_existing.index)]
    else:
        # If the file does not exist, append all new data
        df_to_append = df_new
    # ------------------------------------------------------------------

    if len(df_to_append) == 0:
        print(f"No truly new {timeframe} candles to append after de-duplication.")
        return

    # Save the new data (append to existing file or write if new)
    header = not os.path.exists(path) # Write header only if the file is new
    df_to_append.to_csv(path, mode='a', header=header) # mode='a' is the key for safe appending
    print(f"✅ Successfully appended {len(df_to_append)} truly new {timeframe} candles to {os.path.basename(path)}")


def main():
    # 1. Instantiate the exchange
    exchange = ccxt.coinbase({
        'rateLimit': True, 
        'timeout': 30000 
    })
    
    # 2. Loop through ALL timeframes to fetch and append updates
    for tf in API_TIMEFRAMES:
        print(f"\n--- Processing {tf} Timeframe ---")
        fetch_and_append_ohlcv_update(exchange, SYMBOL, tf)

if __name__ == '__main__':
    main()