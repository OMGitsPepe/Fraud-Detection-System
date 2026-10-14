#!/bin/bash

# Define image name
IMAGE_NAME="securebank"
DOCKERFILE_PATH="../Dockerfile"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" ..

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' built successfully."
else
    echo "Error: Failed to build Docker image."
    exit 1
fi

# Run the Docker container
PORT=$(grep -i "EXPOSE" "$DOCKERFILE_PATH" | awk '{print $2}')
if [ -z "$PORT" ]; then
    echo "Error: No port found in Dockerfile."
    exit 1
fi

echo "Running Docker container on port $PORT..."
docker run -d -p "$PORT:$PORT" "$IMAGE_NAME"
