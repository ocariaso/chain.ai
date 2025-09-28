import subprocess
import os
import sys
import pandas as pd 

def run_pipeline():
    """
    Executes the core project workflow: Fetch Data -> Process Data -> Train Models -> Predict Signal.
    """
    print("=========================================================")
    print("--- Starting bitStats Modeling and Prediction Pipeline ---")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================================")
    
    steps = [
        # 1. FETCH LIVE DATA 
        ("Data Fetching", "src/fetch.py"),
        
        # 2. Process Data (Create Features and Targets)
        ("Data Processing", "src/process.py"),
        
        # 3. Train Models (Generate/Update all 8 Random Forest models)
        ("Model Training", "src/train.py"),
        
        # 4. Generate Live Predictions (Creates signal_report.json for the bot)
        ("Prediction Report", "src/predict.py"),
    ]

    for step_name, script_path in steps:
        print(f"\n[STEP START] Running {step_name} ({script_path})...")
        
        # Execute the script
        result = subprocess.run([sys.executable, script_path], capture_output=False, text=True, check=False)
        
        if result.returncode != 0:
            print(f"\n[STEP FAILED] {step_name} encountered an error! Exiting pipeline.")
            sys.exit(1)
        
        print(f"[STEP COMPLETE] {step_name} finished successfully.")

    print("\n=========================================================")
    print("--- Pipeline Finished Successfully ---")
    print("=========================================================")


if __name__ == "__main__":
    try:
        import pandas as pd 
    except ImportError:
        print("Error: pandas library is required. Please install it (pip install pandas).")
        sys.exit(1)
        
    run_pipeline()