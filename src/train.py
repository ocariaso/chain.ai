import pandas as pd
import numpy as np
import os
import joblib # For saving the model
import json # ADDED: For saving metrics dynamically
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Configuration ---
# LIST OF ALL TIMEFRAMES TO PROCESS
ALL_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '6h', '1d', '10m']

PROCESSED_DATA_PATH_TEMPLATE = 'data/processed/BTC_USD_Coinbase_processed_{timeframe}.csv'
MODEL_PATH_TEMPLATE = 'models/random_forest_{timeframe}.joblib'
REPORT_PATH_TEMPLATE = 'reports/training_report_{timeframe}.txt'

# NEW PATH: Location to save the summary of all metrics for the predict script
METRICS_SUMMARY_PATH = 'models/metrics_summary.json'

# Define the target (y) and features (X) to exclude
TARGET_COLUMN = 'Target'
EXCLUDE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', TARGET_COLUMN]


def load_and_prepare_data(timeframe: str):
    """Loads processed data, sets index, and splits features/target."""
    path = PROCESSED_DATA_PATH_TEMPLATE.format(timeframe=timeframe)
    print(f"\n--- Starting Data Processing for {timeframe} ---")
    print(f"Loading data from: {path}")
    
    if not os.path.exists(path):
        print(f"🛑 ERROR: Processed data file not found at {path}. Skipping this timeframe.")
        return None, None, None
        
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    
    # Define features (X) and target (y)
    features = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[features]
    y = df[TARGET_COLUMN]
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) distribution:\n{y.value_counts(normalize=True).mul(100).round(2)}%")
    
    return X, y, timeframe


def split_data_by_time(X, y):
    """
    Splits data into train (70%), validation (15%), and test (15%) sets 
    using a CRITICAL time-based split.
    """
    
    # 1. Sort by time to ensure a proper split
    X = X.sort_index()
    y = y.sort_index()
    total_len = len(X)
    
    # Define split points (70% Train, 15% Validation, 15% Test)
    train_end = int(total_len * 0.70)
    validation_end = int(total_len * 0.85)
    
    # Create the sets
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:validation_end]
    y_val = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]
    
    print("\n--- Time-Based Data Split ---")
    print(f"Train Period: {X_train.index.min().date()} to {X_train.index.max().date()} ({len(X_train)} samples)")
    print(f"Validation Period: {X_val.index.min().date()} to {X_val.index.max().date()} ({len(X_val)} samples)")
    print(f"Test Period: {X_test.index.min().date()} to {X_test.index.max().date()} ({len(X_test)} samples)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, timeframe):
    """Scales data, trains a Random Forest model, and evaluates performance."""
    
    model_path = MODEL_PATH_TEMPLATE.format(timeframe=timeframe)
    report_path = REPORT_PATH_TEMPLATE.format(timeframe=timeframe)
    
    # 1. Feature Scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 2. Model Training 
    print(f"\n--- Training Random Forest Classifier for {timeframe} ---")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        # BEST MOVE: 'balanced' class_weight to prioritize rare UP/DOWN classes
        class_weight='balanced', 
        n_jobs=-1
    )
    
    # Check if we have enough data to train
    if len(X_train) < 100:
        print(f"❌ Skipping {timeframe}: Not enough training data ({len(X_train)} samples).")
        return None, None
        
    # Ensure all target classes are present in y_train
    train_classes = y_train.unique()
    if len(train_classes) < 2:
        print(f"❌ Skipping {timeframe}: Only one class present in training data: {train_classes}. Cannot train classifier.")
        return None, None
      
    model.fit(X_train_scaled, y_train)
    print("✅ Training complete.")

    # 3. Model Evaluation on the Validation Set
    print("\n--- Evaluating on Validation Set ---")
    y_pred = model.predict(X_val_scaled)
    
    # Get all labels and find which ones are NOT in the validation set predictions (to avoid UndefinedMetricWarning)
    target_names = ['DOWN', 'FLAT', 'UP']
    # Identify labels that are actually present in the validation set AND predicted labels
    unique_labels = np.unique(np.concatenate([y_val, y_pred]))
    # Filter target names to only include those present in the data to avoid errors
    present_labels = [label for label in target_names if label in unique_labels]

    try:
        # Calculate metrics for the summary table
        acc = accuracy_score(y_val, y_pred)
        report_dict = classification_report(y_val, y_pred, output_dict=True, labels=present_labels, zero_division=0)
        
        # Safely extract Precision and Recall for the key trading classes (UP and DOWN)
        up_prec = report_dict.get('UP', {}).get('precision', 0.0)
        down_prec = report_dict.get('DOWN', {}).get('precision', 0.0)
        up_rec = report_dict.get('UP', {}).get('recall', 0.0)
        down_rec = report_dict.get('DOWN', {}).get('recall', 0.0)
        
        # Generate full report string for file
        report = f"--- Training Timeframe: {timeframe} ---\n"
        report += "-"*40 + "\n"
        report += "Evaluation on Validation Set\n"
        report += "-"*40 + "\n"
        report += f"Accuracy: {acc:.4f}\n\n"
        report += "Classification Report:\n"
        # Use the standard classification report for the file
        report += classification_report(y_val, y_pred, zero_division=0)
        report += "\nConfusion Matrix:\n"
        report += str(confusion_matrix(y_val, y_pred, labels=target_names))
    
    except ValueError as e:
        print(f"⚠️ Warning: Could not generate full classification report for {timeframe} due to missing classes. Error: {e}")
        acc, up_prec, down_prec, up_rec, down_rec = 0.0, 0.0, 0.0, 0.0, 0.0
        report = f"--- Training Timeframe: {timeframe} ---\nEvaluation failed due to missing classes in validation set. Try simplifying target for this timeframe."
    
    print(report)
    
    # Save the model and report
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Training report saved to: {report_path}")
    
    # Return key metrics for the summary table
    metrics = {
        'timeframe': timeframe,
        'accuracy': acc,
        'up_precision': up_prec,
        'down_precision': down_prec,
        'up_recall': up_rec,
        'down_recall': down_rec,
        'samples': len(X_train)
    }
    
    return model, metrics


if __name__ == '__main__':
    print("--- Starting Full Model Training Pipeline ---")
    
    training_results = []
    
    for tf in ALL_TIMEFRAMES:
        # 1. Load Data
        X, y, timeframe = load_and_prepare_data(tf)
        
        if X is None:
            continue
            
        # 2. Split Data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_time(X, y)
        
        # 3. Train and Evaluate
        trained_model, metrics = train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, timeframe)
        
        if metrics:
            training_results.append(metrics)
            
    print("\n" + "="*80)
    print("--- ALL Model Training Complete ---")
    print(f"Successfully trained and saved {len(training_results)} models.")
    print("="*80 + "\n")
    
    # --- Generate Summary Table ---
    if training_results:
        summary_df = pd.DataFrame(training_results)
        
        # Re-order columns for readability
        summary_df = summary_df[[
            'timeframe', 'samples', 'accuracy', 
            'up_precision', 'down_precision', 
            'up_recall', 'down_recall'
        ]]
        
        # Format the numbers
        summary_df['samples'] = summary_df['samples'].astype(int)
        for col in summary_df.columns[2:]:
            summary_df[col] = (summary_df[col] * 100).round(2).astype(str) + '%'
        
        print("\n" + "--- TRAINING PERFORMANCE SUMMARY (Validation Set) ---")
        print(summary_df.to_markdown(index=False))
        print("\n")
        print("Key: The model with the highest **UP Precision** and **DOWN Precision** is typically best for generating high-confidence trade signals.")

    # ADDED: Block to save all summary metrics to a JSON file for src/predict.py
    if training_results:
        try:
            # Convert list of dicts to a single dict keyed by timeframe for easy lookup
            metrics_dict = {item['timeframe']: item for item in training_results}
            
            os.makedirs(os.path.dirname(METRICS_SUMMARY_PATH), exist_ok=True)
            with open(METRICS_SUMMARY_PATH, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"✅ All training metrics saved to: {METRICS_SUMMARY_PATH}")
        except Exception as e:
            print(f"❌ ERROR saving metrics JSON: {e}")