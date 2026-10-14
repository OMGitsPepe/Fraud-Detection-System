#!/bin/bash
# securebank/executables/train_model.sh
# Tests the /train_model endpoint of the Flask API.

API_BASE_URL="${API_BASE_URL:-http://localhost:5000}"
ENDPOINT="/train_model"
URL="${API_BASE_URL}${ENDPOINT}"

# Check if the required training dataset path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <train_dataset_path> ['<json_payload_string>']"
    echo "  Example 1 (Minimal): $0 securebank/storage/datasets/dataset_train_....csv"
    echo "  Example 2 (Full JSON): $0 ignored_path '{\"train_dataset_path\": \"path/to/data.csv\", \"model_params\": {\"n_estimators\": 150}}'"
    echo "ERROR: Training dataset path is required as the first argument."
    exit 1
fi

JSON_PAYLOAD=""

# If a second argument (full JSON string) is provided, use it
if [ "$2" ]; then
    JSON_PAYLOAD="$2"
    echo "INFO: Using provided JSON payload: $JSON_PAYLOAD"
    # Simple check if path is likely included
    if ! echo "$JSON_PAYLOAD" | grep -q "\"train_dataset_path\""; then
       echo "WARN: Provided JSON payload does not seem to contain 'train_dataset_path'. The API requires it."
    fi
else
    # Construct minimal JSON payload using only the first argument
    TRAIN_PATH="$1"
    # Basic JSON construction - assumes path doesn't contain quotes needing complex escaping
    JSON_PAYLOAD="{\"train_dataset_path\": \"$TRAIN_PATH\"}"
    echo "INFO: Using minimal JSON payload constructed from path: $JSON_PAYLOAD"
fi


# Execute curl command
echo "INFO: Sending request to $URL..."
echo "----------------------------------------"
curl -X POST \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD" \
     "$URL"
echo "----------------------------------------"
echo "INFO: Request complete. Check output above, log files, and model directory."

exit 0