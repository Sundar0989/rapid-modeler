#!/bin/bash
# Deploy Complete Decoupled AutoML Architecture

set -e

echo "🚀 Deploying Complete Decoupled AutoML Architecture"
echo "=================================================="

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"atus-prism-dev"}
REGION=${GCP_REGION:-"us-east1"}
API_SERVICE_NAME="automl-api-backend"
UI_SERVICE_NAME="automl-streamlit-frontend"
API_IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-repo/automl-api:latest"
UI_IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-repo/automl-streamlit:latest"

echo "📋 Configuration:"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   API Service: ${API_SERVICE_NAME}"
echo "   UI Service: ${UI_SERVICE_NAME}"

# Step 1: Create API Backend Dockerfile
echo ""
echo "📝 Creating API Backend Dockerfile..."
cat > Dockerfile.api << 'EOF'
# Dockerfile for AutoML API Backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (subset needed for API)
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    google-cloud-storage==3.2.0 \
    google-auth==2.40.3 \
    google-cloud-dataproc==5.21.0 \
    requests==2.32.4 \
    pyspark==3.5.6 \
    PyYAML==6.0.2

# Copy API and related files
COPY automl_pyspark/api_endpoints.py .
COPY automl_pyspark/output_manager.py .
COPY automl_pyspark/dataproc_serverless_manager.py .
COPY automl_pyspark/config_manager.py .
COPY automl_pyspark/config.yaml .
COPY env.dataproc .

# Create non-root user
RUN groupadd -g 1099 apiuser && \
    useradd -m -u 1099 -g apiuser apiuser

RUN chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run API server
CMD ["python", "api_endpoints.py"]
EOF

# Step 2: Build and deploy API backend
echo ""
echo "🔨 Building API Backend Docker image..."
docker build -f Dockerfile.api -t ${API_IMAGE_NAME} .

echo ""
echo "📦 Pushing API Backend image to Container Registry..."
docker push ${API_IMAGE_NAME}

echo ""
echo "🚀 Deploying API Backend to Cloud Run..."
gcloud run deploy ${API_SERVICE_NAME} \
    --image=${API_IMAGE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=10 \
    --port=8080 \
    --timeout=3600 \
    --concurrency=10

# Get API service URL
API_SERVICE_URL=$(gcloud run services describe ${API_SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo "✅ API Backend deployed: ${API_SERVICE_URL}"

# Step 3: Build and deploy Streamlit frontend
echo ""
echo "🔨 Building Streamlit Frontend Docker image..."
docker build -f Dockerfile.streamlit -t ${UI_IMAGE_NAME} .

echo ""
echo "📦 Pushing Streamlit Frontend image to Container Registry..."
docker push ${UI_IMAGE_NAME}

echo ""
echo "🚀 Deploying Streamlit Frontend to Cloud Run..."
gcloud run deploy ${UI_SERVICE_NAME} \
    --image=${UI_IMAGE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --port=8501 \
    --set-env-vars="AUTOML_API_URL=${API_SERVICE_URL}" \
    --timeout=3600 \
    --concurrency=80

# Get UI service URL
UI_SERVICE_URL=$(gcloud run services describe ${UI_SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo "✅ Streamlit Frontend deployed: ${UI_SERVICE_URL}"

# Step 4: Test the complete architecture
echo ""
echo "🧪 Testing Complete Architecture..."

# Test API health
echo "Testing API health..."
if curl -f "${API_SERVICE_URL}/health" > /dev/null 2>&1; then
    echo "✅ API Backend is healthy"
else
    echo "❌ API Backend health check failed"
fi

# Test UI health
echo "Testing UI health..."
if curl -f "${UI_SERVICE_URL}/_stcore/health" > /dev/null 2>&1; then
    echo "✅ Streamlit Frontend is healthy"
else
    echo "❌ Streamlit Frontend health check failed"
fi

# Cleanup temporary files
rm -f Dockerfile.api

echo ""
echo "🎉 Complete Decoupled AutoML Architecture Deployed Successfully!"
echo "=============================================================="
echo ""
echo "🌐 Service URLs:"
echo "   API Backend:        ${API_SERVICE_URL}"
echo "   Streamlit Frontend: ${UI_SERVICE_URL}"
echo ""
echo "🧪 Test the Complete System:"
echo "   1. Open: ${UI_SERVICE_URL}"
echo "   2. Submit a test job via the UI"
echo "   3. Monitor real-time progress"
echo "   4. View results and visualizations"
echo ""
echo "🔗 API Endpoints Available:"
echo "   ${API_SERVICE_URL}/health"
echo "   ${API_SERVICE_URL}/api/v1/jobs (POST - submit job)"
echo "   ${API_SERVICE_URL}/api/v1/jobs (GET - list jobs)"
echo "   ${API_SERVICE_URL}/api/v1/jobs/{job_id}/status"
echo "   ${API_SERVICE_URL}/api/v1/jobs/{job_id}/logs"
echo "   ${API_SERVICE_URL}/api/v1/jobs/{job_id}/results"
echo ""
echo "📊 Architecture Overview:"
echo "   ┌─────────────────┐    API Calls    ┌─────────────────┐"
echo "   │   Streamlit     │◄───────────────►│   AutoML API    │"
echo "   │   Frontend      │                 │   Backend       │"
echo "   │   (Cloud Run)   │                 │   (Cloud Run)   │"
echo "   └─────────────────┘                 └─────────────────┘"
echo "            │                                    │"
echo "            │                                    │"
echo "            ▼                                    ▼"
echo "   ┌─────────────────────────────────────────────────────┐"
echo "   │              Google Cloud Storage                   │"
echo "   │        gs://rapid_modeler_app/automl_jobs/         │"
echo "   └─────────────────────────────────────────────────────┘"
echo "                              │"
echo "                              ▼"
echo "                    ┌─────────────────┐"
echo "                    │   Dataproc      │"
echo "                    │   Serverless    │"
echo "                    │   (Processing)  │"
echo "                    └─────────────────┘"
echo ""
echo "🎯 Key Benefits Achieved:"
echo "   ✅ Decoupled architecture - UI and backend scale independently"
echo "   ✅ Real-time monitoring - Live job status and log streaming"
echo "   ✅ Standardized outputs - All results in predictable GCS locations"
echo "   ✅ Cost optimized - Frontend runs on minimal resources"
echo "   ✅ Developer friendly - Separate deployment and development cycles"
