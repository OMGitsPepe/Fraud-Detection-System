#!/bin/bash

# Define the JSON file path
JSON_FILE="../test.json"

# Make the POST request
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @"$JSON_FILE"
