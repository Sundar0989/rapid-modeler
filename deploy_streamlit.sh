#!/bin/bash
# Deploy Streamlit Frontend to Cloud Run

set -e

echo "🚀 Deploying Streamlit Frontend to Cloud Run"
echo "=============================================="

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"atus-prism-dev"}
REGION=${GCP_REGION:-"us-east1"}
SERVICE_NAME="automl-streamlit-ui"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-repo/automl-streamlit:latest"
AUTOML_API_URL="https://rapid-modeler-meaphyqd6a-ue.a.run.app"

echo "📋 Configuration:"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service: ${SERVICE_NAME}"
echo "   Image: ${IMAGE_NAME}"
echo "   API URL: ${AUTOML_API_URL}"

# Build Docker image
echo ""
echo "🔨 Building Streamlit Docker image..."
docker build -f Dockerfile.streamlit -t ${IMAGE_NAME} .

# Push to Container Registry
echo ""
echo "📦 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --port=8501 \
    --set-env-vars="AUTOML_API_URL=${AUTOML_API_URL}" \
    --timeout=3600 \
    --concurrency=80

# Get service URL
echo ""
echo "🔗 Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo ""
echo "✅ Streamlit Frontend deployed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🧪 Test the deployment:"
echo "   1. Open: ${SERVICE_URL}"
echo "   2. Submit a test job"
echo "   3. Monitor job progress in real-time"
echo "   4. View results when complete"
echo ""
echo "📊 Architecture Overview:"
echo "   Frontend (Streamlit): ${SERVICE_URL}"
echo "   Backend API: ${AUTOML_API_URL}"
echo "   Processing: Dataproc Serverless"
echo "   Storage: gs://rapid_modeler_app/"
echo ""
echo "🎯 Key Features Available:"
echo "   ✅ Real-time job monitoring"
echo "   ✅ Live log streaming"
echo "   ✅ Interactive results visualization"
echo "   ✅ Model comparison and analysis"
echo "   ✅ Variable tracking reports"
echo "   ✅ Standardized output locations"
