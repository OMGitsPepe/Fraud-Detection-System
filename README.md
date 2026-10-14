# SecureBank Fraud Detection System v2 Prototype

## Overview

This project presents a prototype machine learning system designed to detect fraudulent credit card transactions for 
SecureBank. It aims to significantly improve upon the performance of a previous system by incorporating advanced data 
handling techniques, insightful feature engineering, and a robust evaluation strategy. The system provides an API for 
real-time predictions, on-demand dataset generation, and model retraining.

## Features

* **Fraud Prediction:** `/predict` endpoint takes transaction details (JSON) and returns a classification "legitimate" or "fraudulent".
* **Dataset Creation:** `/create_dataset` endpoint triggers the generation of timestamped, preprocessed, and partitioned 
training and testing datasets based on source data.
* **Model Training:** `/train_model` endpoint trains a new Random Forest model on a specified dataset and saves the versioned model artifact.
* **Advanced Data Handling:** Implements time-aware and customer-aware data splitting to prevent data leakage and provide realistic performance evaluation.
* **Feature Engineering:** Creates features based on data analysis, including time-based patterns (hour, day, month), 
transaction velocity (`log_time_diff`), and category-specific amount normalization (`amt_zscore`).
* **Dockerized Deployment:** Runs as a containerized Flask application for portability and consistent environments.
* **JSON Logging:** Logs API requests, responses, errors, and key events to timestamped JSON files for monitoring and auditing.

## Project Structure

## Prerequisites

* **Docker:** Docker engine must be installed and running. [Install Docker](https://docs.docker.com/engine/install/)
* **Python 3.9+:** Required if running outside Docker.
* **Git (Optional):** For cloning the repository.

## Setup

1. **Clone Repository (if applicable):**
```bash
  git clone <your-repo-url>
  cd securebank
```

2. **Place Raw Data:** Copy the source data files (`customer_release.csv`, `transactions_release.parquet`, `fraud_release.json`) 
into the `securebank/storage/` directory.<br><br>

3. **Place Clean Database CSV:** Place your comprehensive `clean_database.csv` into the `securebank/storage/database/` directory. 
Create the `database` subdirectory if it doesn't exist.<br><br>

4. **Place Default Model:** Train a Random Forest model using your training script and save it using the `RandomForestModel` 
class's `.save_model()` method. Place the resulting `.joblib` file into `securebank/storage/models/` and rename it to `default_model.joblib`. 
The `/predict` endpoint requires this file to be present on startup.<br><br>

5. **Create `test.json`:** Create a file named `test.json` in the main `securebank/` directory. This file will be used 
by `predict.sh`. Add valid JSON representing a single transaction. Example content:
```json
{
    "trans_date_trans_time": "2024-09-15 12:34:56",
    "cc_num": 38859492057661,
    "unix_time": 1694775296,
    "merchant": "fraud_Lind-Buckridge",
    "category": "entertainment",
    "amt": 50.75,
    "merch_lat": 43.150704,
    "merch_long": -112.154481
}
```

## Running the Application (Docker)

The easiest way to run the application is using Docker via the provided script.

1. **Navigate to Executables:**
```bash
  cd securebank/executables/
```

2. **Make Scripts Executable:**
```bash
  chmod +x run_server.sh
  chmod +x predict.sh
  chmod +x create_dataset.sh
  chmod +x train_model.sh
```

3. **Run the Server:**
```bash
  ./run_server.sh
```
This will build the Docker image `securebank` if it doesn't exist and start the container in the background. 
The API will be accessible at `http://localhost:5000`.

4. **Check if Running:**
```bash
  docker ps
```
Look for a container using the `securebank` image.

5. **View Server Logs:**
```bash
  docker logs <container_id_or_name_from_docker_ps>
```
If the container is not running check for docker logs. Maybe the app run into an error.

## Using the API / Testing Scripts

While the server is running use a separate terminal, you can use the scripts in `securebank/executables/` to interact with the API.

1. **Navigate to Executables:**
```bash
  cd securebank/executables/
```

2. **Test Prediction (`/predict`):**
`../test.json` exists and is populated
```bash
  ./predict.sh
```
*Output:* JSON response:
```bash
  "predict": "legitimate"`
```
```bash
  "predict": "fraudulent"`
```

3. **Test Dataset Creation (`/create_dataset`):**
* Using default parameters:
```bash
  ./create_dataset.sh
```
* Using custom parameters (provide JSON string):
```bash
  ./create_dataset.sh '{"test_size_customers": 0.3, "nonfraud_ratio": 4}'
```
*Output:* JSON response indicating success/failure and paths to created datasets in `securebank/storage/datasets/`.

4. **Test Model Training (`/train_model`):**
* Requires the path to an existing training dataset (relative to project root).
* Using a specific dataset and default model params:
```bash
  ./train_model.sh storage/datasets/<trainig_dataset_path>
```

* Using a specific dataset and custom model params:
```bash
  ./train_model.sh ignored_path '{"train_dataset_path": "securebank/storage/datasets/<train_dataset_path>", "model_params": {"n_estimators": 50}}'
```
*Output:* JSON response indicating success/failure and path to the saved model in `securebank/storage/models/`.

5. **Check Application Logs:** Review the timestamped JSON files created in `securebank/logs/` and the output of 
`docker logs <container_id>` for detailed information and potential errors.

## Meeting Requirements

This system prototype addresses the key functional requirements:

* **Improved Performance:** The combination of the advanced data partitioning strategy, insightful feature engineering, 
and the use of a tuned Random Forest model demonstrably improves performance significantly over the baseline.
* **Dataset Generation:** The `/create_dataset` endpoint, tested via `create_dataset.sh`, allows administrators to 
generate fresh, consistently processed training/testing datasets on demand.
* **Model Catalog/Selection:** The `/train_model` endpoint, tested via `train_model.sh`, trains models and saves them 
with unique timestamp identifiers, creating a versioned model catalog in the filesystem. The `/predict` endpoint uses a 
default model but can be extended to load specific versions from this catalog.
* **Performance Auditing:** The comprehensive JSON logging for every API request captures inputs, outputs, timestamps, 
and errors, providing the necessary data trail for administrators to monitor system activity and audit performance over time.