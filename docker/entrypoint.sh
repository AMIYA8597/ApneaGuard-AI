#!/bin/bash
set -e

# Fetch model artifacts if a release URL is provided
if [ -n "$MODEL_RELEASE_URL" ]; then
    echo "Downloading model artifacts from GitHub Release..."
    # The models should be packed in a tar.gz file. We extract them directly into models/artifacts/
    curl -sL "$MODEL_RELEASE_URL" | tar -xz -C models/artifacts/
    echo "Model artifacts downloaded successfully."
else
    echo "Warning: MODEL_RELEASE_URL environment variable is not set."
    echo "Assuming model artifacts are already present in models/artifacts/ or mounted via volume."
fi

# Run database migrations
echo "Running Alembic migrations..."
alembic upgrade head

# Start the application
echo "Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
