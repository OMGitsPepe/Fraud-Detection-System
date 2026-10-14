# Imports
import os
import json
import datetime
import time
import traceback
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from modules.data.raw_data_handler import RawDataHandler
from modules.model import RandomForestModel
from typing import Optional, Union

# Configuration
STORAGE_PATH = 'storage'
CLEAN_DATABASE_CSV = 'storage/database/clean_database.csv'
DATASET_PATH = 'storage/datasets'
MODEL_SAVE_PATH = 'storage/saved_models'
LOG_PATH = 'logs'
DEFAULT_MODEL_FILENAME = "default_model.joblib"
DEFAULT_MODEL_PATH = 'default_model.joblib'

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Global Variables / App State
app = Flask(__name__)
default_model: Optional[RandomForestModel] = None

# Store last transaction times
LAST_TRANSACTION_TIMES = {}

# List Define of exact feature default model was trained on (order matters)
EXPECTED_MODEL_FEATURES = [
    'month_sin',
    'month_cos',
    'log_time_diff',
    'amt_zscore',
    'time_segment_daytime',
    'time_segment_late_night',
    'day_of_week_Friday',
    'day_of_week_Monday',
    'day_of_week_Saturday',
    'day_of_week_Sunday',
    'day_of_week_Thursday',
    'day_of_week_Tuesday',
    'day_of_week_Wednesday'
]


# Helper functions

# JSON serialization
def make_serializable(data):
    """
    Recursively converts numpy types and other non-serializables to primitive types.
    :param data: Data that was given from curl command.
    :return: Converted data.
    """
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(i) for i in data]
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
        return int(data)  # Convert numpy ints to python ints
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)  # Convert numpy floats to python floats
    elif isinstance(data, (np.ndarray,)):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, (np.bool_)):
        return bool(data)  # Convert numpy bools to python bools
    elif isinstance(data, (datetime.date, datetime.datetime)):
        return data.isoformat()  # Convert datetimes to string
    elif pd.isna(data):  # Handle pandas NaT or NaN specifically
        return None
    else:
        return data


def log_event(log_data: dict):
    """
    Logs a dictionary as a JSON object to a timestamped file.
    :param log_data: Data and metadata to be logged.
    :return:
    """
    """Logs a dictionary as a JSON object to a timestamped file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(LOG_PATH, f"log_{timestamp}.json")
    log_data['log_timestamp_utc'] = datetime.datetime.utcnow().isoformat()

    try:
        # Convert data before dumping
        serializable_data = make_serializable(log_data)

        with open(log_file_path, 'w') as f:
            # Dump the serializable data
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        # Log error to console if logging itself fails
        print(f"CRITICAL: Error writing log file {log_file_path}: {e}")


# Preprocessing function for prediction
def preprocess_predict_data(transaction_data: dict) -> Optional[pd.DataFrame]:
    """
    Preprocesses a single transaction dictionary for prediction.
    :param transaction_data: Dictionary containing the raw transaction data.
    :return: A single row DataFrame with the exact features the model expects,
    or None if preprocessing fails.
    """
    # Check for feature to be used for prediction
    required_keys = ['trans_date_trans_time', 'cc_num', 'category', 'amt']
    if not all(key in transaction_data for key in required_keys):
        log_event({"event": "preprocessing_error",
                   "error": "Missing required keys", "data": transaction_data})
        return None

    keys_to_keep = list(
        set(required_keys + ['lat', 'long', 'city_pop', 'merch_lat',
                             'merch_long']))
    processed_data = {k: transaction_data.get(k) for k in keys_to_keep if
                      k in transaction_data}
    try:
        df = pd.DataFrame([processed_data])
        if df.empty: raise ValueError("DataFrame creation failed.")
    except Exception as e:
        log_event({"event": "preprocessing_error",
                   "error": f"Failed to create DataFrame: {e}",
                   "data": processed_data})
        return None

    # Convert dates into distinct columns
    try:
        handler = RawDataHandler(storage_path=STORAGE_PATH,
                                 save_path=STORAGE_PATH)
        df = handler.convert_dates(df.copy())
        if df is None or df.empty: raise ValueError(
            "RawDataHandler.convert_dates returned empty/None.")
        if not all(col in df.columns for col in
                   ['hour', 'month_num', 'day_of_week']):
            raise ValueError(
                "Required date columns not found after convert_dates.")
        current_time = df['trans_date_trans_time'].iloc[0]
    except Exception as e:
        log_event({"event": "preprocessing_error",
                   "error": f"Call to RawDataHandler.convert_dates failed: {e}",
                   "trace": traceback.format_exc(), "data": transaction_data})
        return None

    # Feature engineer
    try:
        # Create Time Segment based on hour
        df['time_segment'] = np.where((df['hour'] <= 3) | (df['hour'] >= 22),
                                      'late_night', 'daytime')

        # Cyclical encoding for month using month_num
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

        # Compute time difference between consecutive transactions per customer
        time_diff_minutes = 0.0
        cc_num = df['cc_num'].iloc[0]
        try:
            if os.path.exists(CLEAN_DATABASE_CSV):
                historical_db_td = pd.read_csv(CLEAN_DATABASE_CSV,
                                               usecols=['cc_num',
                                                        'trans_date_trans_time'],
                                               parse_dates=[
                                                   'trans_date_trans_time'])
                customer_history_td = historical_db_td[
                    (historical_db_td['cc_num'] == cc_num) & (historical_db_td[
                                                                  'trans_date_trans_time'] < current_time)]
                if not customer_history_td.empty:
                    last_transaction_time = customer_history_td[
                        'trans_date_trans_time'].max()
                    time_delta = current_time - last_transaction_time
                    time_diff_minutes = time_delta.total_seconds() / 60.0
        except Exception as lookup_error:
            log_event({"event": "feature_engineering_error",
                       "warning": f"Failed during CSV lookup for time_diff (defaulting to 0): {lookup_error}",
                       "cc_num": cc_num})
        df['time_diff'] = time_diff_minutes
        df['log_time_diff'] = np.log(df['time_diff'] + 1.0)
        if not os.path.exists(CLEAN_DATABASE_CSV) and time_diff_minutes == 0.0:
            log_event({"event": "feature_engineering_warning",
                       "message": f"Historical database CSV not found at {CLEAN_DATABASE_CSV}. Assuming time_diff = 0.",
                       "cc_num": cc_num})
        elif time_diff_minutes == 0.0:
            log_event({"event": "feature_engineering_info",
                       "message": "No prior transaction history found in CSV for cc_num or error occurred. Assuming first transaction (time_diff=0).",
                       "cc_num": cc_num})

        # Amount Z-Score
        amt_zscore = 0.0
        current_category = df['category'].iloc[0].lower()
        current_amt = df['amt'].iloc[0]
        stats_calculated = False
        try:
            start_calc_time = time.time()
            if not os.path.exists(CLEAN_DATABASE_CSV):
                log_event({"event": "feature_engineering_warning",
                           "message": f"Cannot calculate Z-score: Clean database CSV not found at {CLEAN_DATABASE_CSV}",
                           "category": current_category})
            else:
                # Load category and amount columns
                db_df = pd.read_csv(CLEAN_DATABASE_CSV,
                                    usecols=['category', 'amt'])
                # Calculate stats for all categories
                category_stats_df = db_df.groupby('category')['amt'].agg(
                    ['mean', 'std']).reset_index()
                category_stats_df['std'] = category_stats_df['std'].fillna(
                    0)  # Handle single-entry categories
                # Convert to dictionary for lookup
                stats_dict = category_stats_df.set_index('category').to_dict(
                    'index')

                # Look up stats for the current category
                stats = stats_dict.get(current_category)

                if stats:
                    if stats['std'] > 0:
                        amt_zscore = (current_amt - stats['mean']) / stats[
                            'std']
                        stats_calculated = True
                    elif stats['std'] == 0:
                        amt_zscore = 0.0
                        stats_calculated = True
                else:
                    # Category not found in historical data
                    log_event({"event": "feature_engineering_warning",
                               "message": "Category not found in historical data for Z-score calculation",
                               "category": current_category})

            calc_duration = time.time() - start_calc_time
            if stats_calculated:
                log_event({"event": "feature_engineering_info",
                           "message": "Calculated Z-score using on-the-fly stats from CSV.",
                           "category": current_category,
                           "amt_zscore": amt_zscore,
                           "calc_duration_seconds": round(calc_duration, 4)})
            else:
                log_event({"event": "feature_engineering_warning",
                           "message": "Using default Z-score=0 due to missing stats or calculation issue.",
                           "category": current_category,
                           "calc_duration_seconds": round(calc_duration, 4)})

        except Exception as zscore_error:
            log_event({"event": "feature_engineering_error",
                       "error": f"Failed during on-the-fly Z-score calculation: {zscore_error}",
                       "trace": traceback.format_exc(),
                       "category": current_category})

        df['amt_zscore'] = amt_zscore

        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['time_segment'], prefix='time_segment', drop_first=False, dummy_na=False)

        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=all_days)
        df = pd.get_dummies(df, columns=['day_of_week'], prefix='day_of_week', dummy_na=False, drop_first=False)
        base_day_col = 'day_of_week_Friday'
        if base_day_col in EXPECTED_MODEL_FEATURES:
            pass
        elif base_day_col in df.columns:
            df = df.drop(columns=[base_day_col])  # Drop if exists and not expected

        expected_day_cols = [f'day_of_week_{day}' for day in all_days if
                             f'day_of_week_{day}' in EXPECTED_MODEL_FEATURES]
        for col in expected_day_cols:
            if col not in df.columns: df[col] = 0


    except Exception as e:
        log_event({"event": "preprocessing_error",
                   "error": f"Feature engineering failed: {e}",
                   "trace": traceback.format_exc(),
                   "data": df.to_dict() if isinstance(df,
                                                      pd.DataFrame) else str(
                       df)})
        return None

    # Check if the feature created are used for the model
    try:
        missing_cols = []
        for col in EXPECTED_MODEL_FEATURES:
            if col not in df.columns:
                missing_cols.append(col)
                df[col] = 0
        if missing_cols:
            log_event({"event": "feature_warning",
                       "message": f"Expected features not generated, adding with default value 0: {missing_cols}"})
        final_features = df[EXPECTED_MODEL_FEATURES]
        unexpected_cols = [col for col in df.columns if
                           col not in EXPECTED_MODEL_FEATURES]
        if unexpected_cols:
            log_event({"event": "feature_warning",
                       "message": f"Unexpected features present after processing (will be dropped): {unexpected_cols}"})
    except Exception as e:
        log_event({"event": "preprocessing_error",
                   "error": f"Final feature selection failed: {e}",
                   "trace": traceback.format_exc(),
                   "available_columns": df.columns.tolist(),
                   "expected_columns": EXPECTED_MODEL_FEATURES})
        return None

    return final_features


# Load default model
try:
    if os.path.exists(DEFAULT_MODEL_PATH):
        default_model = RandomForestModel.load_model(DEFAULT_MODEL_PATH)
        log_event({"event": "server_startup", "status": "success",
                   "message": f"Default model loaded successfully from {DEFAULT_MODEL_PATH}"})
    else:
        log_event({"event": "server_startup", "status": "error",
                   "message": f"Default model file not found at {DEFAULT_MODEL_PATH}. /predict endpoint will not work."})
except Exception as e:
    log_event({"event": "server_startup", "status": "error",
               "message": f"Failed to load default model: {e}",
               "trace": traceback.format_exc()})
    default_model = None


# Flask endpoints

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict fraud for a single transaction.
    :return: Prediction result.
    """
    endpoint_start_time = time.time()
    log_data = {"event": "prediction_request", "endpoint": "/predict"}

    if default_model is None or not default_model._is_trained:
        log_data.update(
            {"status": "error", "message": "Model not loaded or not trained"})
        log_event(log_data)
        return jsonify(
            {"error": "Model not available"}), 503  # Service Unavailable

    if not request.is_json:
        log_data.update({"status": "error", "message": "Request must be JSON"})
        log_event(log_data)
        return jsonify({"error": "Request must be JSON"}), 400  # Bad Request

    data = request.get_json()
    log_data["request_data"] = data

    # Preprocess data
    processed_features = preprocess_predict_data(data)

    if processed_features is None:
        log_data.update({"status": "error",
                         "message": "Preprocessing failed"})
        log_event(log_data)
        return jsonify({
                           "error": "Invalid transaction data or preprocessing failure"}), 400

    # Predict
    try:
        prediction_result = default_model.predict(processed_features)
        prediction = int(prediction_result[0])  # Get the single prediction
        prediction_label = "fraudulent" if prediction == 1 else "legitimate"

        log_data.update({"status": "success",
                         "prediction_raw": prediction,
                         "prediction_label": prediction_label})
        response = {"predict": prediction_label}

    except Exception as e:
        log_data.update(
            {"status": "error", "message": f"Prediction failed: {e}",
             "trace": traceback.format_exc()})
        response = {"error": "Prediction failed"}
        status_code = 500  # Internal Server Error
        log_event(log_data)
        return jsonify(response), status_code

    endpoint_duration = time.time() - endpoint_start_time
    log_data["duration_seconds"] = round(endpoint_duration, 4)
    log_event(log_data)

    return jsonify(response), 200


@app.route('/create_dataset', methods=['POST'])
def create_dataset_endpoint():
    """
    Endpoint to trigger dataset creation using RawDataHandler.
    :return: Creating training and test datasets path and success status or
    Error message.
    """
    """Endpoint to trigger dataset creation using RawDataHandler."""
    endpoint_start_time = time.time()
    log_data = {"event": "dataset_creation_request",
                "endpoint": "/create_dataset"}

    try:
        # Get parameters from request or use defaults
        req_data = request.get_json() if request.is_json else {}
        test_size = req_data.get('test_size_customers', 0.2)
        time_split = req_data.get('time_split_quantile', 0.8)
        ratio = req_data.get('nonfraud_ratio', 5)
        random_state = req_data.get('random_state', 42)

        log_data["parameters"] = {"test_size": test_size,
                                  "time_split": time_split, "ratio": ratio,
                                  "random_state": random_state}

        # Instantiate RawDataHandler
        handler = RawDataHandler(storage_path=STORAGE_PATH,
                                 save_path=STORAGE_PATH)

        # Extract data
        customer_info, transaction_info, fraud_info = handler.extract(
            "customer_release.csv",
            "transactions_release.parquet",
            "fraud_release.json")

        # Transform data
        merged_data = handler.transform(customer_info, transaction_info,
                                        fraud_info)

        # Convert dates and time
        merged_data = handler.convert_dates(merged_data)

        # Feature engineer
        engineered_data = handler.feature_engineer(merged_data)

        print("Creating train/test datasets...")
        train_path, test_path = handler.create_train_test_datasets(
            engineered_data,
            test_size_customers=test_size,
            time_split_quantile=time_split,
            nonfraud_ratio=ratio,
            random_state=random_state
        )

        log_data.update({
            "status": "success",
            "train_dataset_path": train_path,
            "test_dataset_path": test_path
        })
        response = {
            "status": "success",
            "message": "Datasets created successfully.",
            "train_dataset": train_path,
            "test_dataset": test_path
        }
        status_code = 200

    except FileNotFoundError as e:
        log_data.update(
            {"status": "error",
             "message": f"Source data file not found: {e}"})
        response = {"error": f"Source data file not found: {e}"}
        status_code = 400
    except Exception as e:
        log_data.update({"status": "error",
                         "message": f"Dataset creation failed: {e}",
                         "trace": traceback.format_exc()})
        response = {"error": f"Dataset creation failed: {e}"}
        status_code = 500

    endpoint_duration = time.time() - endpoint_start_time
    log_data["duration_seconds"] = round(endpoint_duration, 4)
    log_event(log_data)

    return jsonify(response), status_code


@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """
    Endpoint to trigger model training.
    :return: Model training success status and model trained path
    or Error message.
    """
    """Endpoint to trigger model training."""
    endpoint_start_time = time.time()
    log_data = {"event": "model_training_request",
                "endpoint": "/train_model"}

    try:
        # Configuration parameters
        req_data = request.get_json() if request.is_json else {}

        # Specify dataset path in request
        train_dataset_path = req_data.get('train_dataset_path')

        # If no dataset was past, automatically find latest dataset
        if not train_dataset_path:
            # Find the latest training dataset
            try:
                train_files = [f for f in os.listdir(DATASET_PATH) if
                               f.startswith("dataset_train_") and f.endswith(
                                   ".csv")]
                train_files.sort(reverse=True)
                if not train_files:
                    raise FileNotFoundError("No training dataset files found.")
                train_dataset_path = os.path.join(DATASET_PATH, train_files[0])
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not automatically find training dataset: {e}")

        # Default parameters for model
        rf_params = req_data.get('model_params', {
            'n_estimators': 100,
            'max_depth': 20,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        })
        log_data["parameters"] = {"train_dataset": train_dataset_path,
                                  "model_params": rf_params}

        # Load training dataset
        train_df = pd.read_csv(train_dataset_path)
        # Check if 'is_fraud' in training data
        if 'is_fraud' not in train_df.columns:
            raise ValueError(
                "'is_fraud' column not found in training dataset.")

        # Get feature and target
        y_train = train_df['is_fraud']
        X_train = train_df.drop(columns=['is_fraud'])

        # Drop cc_num if present
        if 'cc_num' in X_train.columns:
            X_train = X_train.drop(columns=['cc_num'])

        # Train model
        model_to_train = RandomForestModel(**rf_params)

        training_start_time = time.time()
        model_to_train.train(X_train, y_train)  # Training happens here
        training_duration = time.time() - training_start_time
        log_data["training_duration_seconds"] = round(training_duration, 2)

        # Save model
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"fraud_model_rf_{timestamp}.joblib"
        save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
        model_to_train.save_model(save_path)

        log_data.update({"status": "success", "model_save_path": save_path})
        response = {
            "status": "success",
            "message": "Model trained successfully.",
            "model_path": save_path,
            "training_duration_seconds": round(training_duration, 2)
        }
        status_code = 200

    except FileNotFoundError as e:
        log_data.update({"status": "error",
                         "message": f"Training data file not found: {e}"})
        response = {"error": f"Training data file not found: {e}"}
        status_code = 400
    # Catch specific errors like missing columns
    except ValueError as e:
        log_data.update({"status": "error",
                         "message": f"Data loading/validation error: {e}"})
        response = {"error": f"Data loading/validation error: {e}"}
        status_code = 400
    except Exception as e:
        log_data.update(
            {"status": "error", "message": f"Model training failed: {e}",
             "trace": traceback.format_exc()})
        response = {"error": f"Model training failed: {e}"}
        status_code = 500

    endpoint_duration = time.time() - endpoint_start_time
    log_data["duration_seconds"] = round(endpoint_duration, 4)
    log_event(log_data)

    return jsonify(response), status_code


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)  # Runs on http://localhost:5000
