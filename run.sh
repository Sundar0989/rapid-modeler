#!/bin/bash

# Stop and remove existing container if it exists
docker stop rapid_modeler 2>/dev/null || true
docker rm rapid_modeler 2>/dev/null || true

# Build the Docker image
echo "🔨 Building Docker image..."
docker build . -t rapid_modeler

# Run the container with volume mounts for development
echo "🚀 Starting Rapid Modeler container with volume mounts..."
docker run -p 8080:8080 \
  -v $(pwd)/automl_pyspark/automl_results:/app/automl_pyspark/automl_results \
  -v $(pwd)/automl_pyspark/automl_jobs:/app/automl_pyspark/automl_jobs \
  -v $(pwd)/automl_pyspark/background_job_manager.py:/app/automl_pyspark/background_job_manager.py \
  -v $(pwd)/automl_pyspark/unified_job_script_generator.py:/app/automl_pyspark/unified_job_script_generator.py \
  -v $(pwd)/automl_pyspark/streamlit_automl_app.py:/app/automl_pyspark/streamlit_automl_app.py \
  -e GOOGLE_CLOUD_PROJECT=atus-prism-dev \
  -e ENABLE_DATAPROC_SERVERLESS=false \
  -e DATAPROC_SERVERLESS_ENABLED=false \
  --name rapid_modeler rapid_modeler:latest

echo "✅ Rapid Modeler container started successfully!"
echo "🌐 Access the application at: http://localhost:8080"
echo "🔑 Service account key mounted and configured"