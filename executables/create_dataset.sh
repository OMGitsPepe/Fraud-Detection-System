#!/bin/bash
# securebank/executables/create_dataset.sh
# Tests the /create_dataset endpoint of the Flask API.

# Configuration
# Uses environment variable API_BASE_URL if set, otherwise defaults to localhost:5000
API_BASE_URL="${API_BASE_URL:-http://localhost:5000}"
ENDPOINT="/create_dataset"
URL="${API_BASE_URL}${ENDPOINT}"

# Argument handling
# Default JSON payload
JSON_PAYLOAD="{}"

if [ "$1" ]; then
    JSON_PAYLOAD="$1"
    echo "INFO: Using provided JSON payload: $JSON_PAYLOAD"
else
    echo "INFO: Using default JSON payload: {}"
fi

# Execute curl command
echo "INFO: Sending request to $URL..."
echo "----------------------------------------"
curl -X POST \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD" \
     "$URL"
echo "----------------------------------------"
echo "INFO: Request complete. Check output above and log files."

exit 0