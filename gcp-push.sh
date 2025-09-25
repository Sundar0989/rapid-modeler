#!/bin/bash

# Configuration
PROJECT_ID="atus-prism-dev"
REGION="us-east1"
SERVICE_NAME="rapid-modeler"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-repo/rapid_modeler:latest"

# Bucket configuration
TEMP_BUCKET="rapid_modeler_app"
RESULTS_BUCKET="rapid_modeler_app"

echo "🚀 Building and deploying AutoML with Dataproc Serverless to Cloud Run..."
echo "📋 Configuration:"
echo "   Project ID: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service Name: ${SERVICE_NAME}"
echo "   Temp Bucket: ${TEMP_BUCKET}"
echo "   Results Bucket: ${RESULTS_BUCKET}"

# Build the Cloud Run Docker image
echo "🔨 Building Cloud Run Docker image..."
docker build . --platform linux/amd64 -t ${IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "❌ Cloud Run Docker build failed!"
    exit 1
fi

# Push Cloud Run image to Container Registry
echo "📦 Pushing Cloud Run image to Container Registry..."
docker push ${IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "❌ Cloud Run Docker push failed!"
    exit 1
fi

# Build and push Dataproc custom image (using working Dockerfile structure)
DATAPROC_IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-repo/rapid_modeler_dataproc:latest"

echo "🔨 Building Dataproc custom runtime image (based on working Dockerfile)..."
echo "   Using Dockerfile.dataproc with Python 3.13 + Java 21 + all dependencies pre-installed"
docker build -f Dockerfile.dataproc . --platform linux/amd64 -t ${DATAPROC_IMAGE}

if [ $? -ne 0 ]; then
    echo "❌ Dataproc Docker build failed!"
    exit 1
fi

echo "📦 Pushing Dataproc image to Container Registry..."
docker push ${DATAPROC_IMAGE}

if [ $? -ne 0 ]; then
    echo "❌ Dataproc Docker push failed!"
    exit 1
fi

echo "✅ Dataproc custom runtime image ready!"
echo "   Image: ${DATAPROC_IMAGE}"
echo "   Features: Pre-installed ML libraries, AutoML PySpark package, JAR files"
echo "   Benefits: 5-15 minute faster job startup, no runtime package installation"

# Deploy to Cloud Run with Dataproc Serverless enabled
echo "🚀 Deploying to Cloud Run with Dataproc Serverless..."
 gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 80 \
  --max-instances 10 \
  --min-instances 0 \
  --port 8080 \
  --set-env-vars="ENABLE_DATAPROC_SERVERLESS=true" \
  --set-env-vars="USE_DATAPROC_SERVERLESS=true" \
  --set-env-vars="SPARK_MODE=serverless" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars="GCP_REGION=${REGION}" \
  --set-env-vars="GCP_TEMP_BUCKET=${TEMP_BUCKET}" \
  --set-env-vars="GCP_RESULTS_BUCKET=${RESULTS_BUCKET}" \
  --set-env-vars="DATAPROC_BATCH_ID_PREFIX=automl-spark" \
  --set-env-vars="SPARK_EXECUTOR_COUNT=auto" \
  --set-env-vars="SPARK_DRIVER_MEMORY=4g" \
  --set-env-vars="SPARK_EXECUTOR_MEMORY=4g" \
  --set-env-vars="PLATFORM_ARCH=cloud" \
  --set-env-vars="ENABLE_SYNAPSEML_LIGHTGBM=true" \
  --set-env-vars="ENABLE_SPARK_XGBOOST=true" \
  --set-env-vars="ENABLE_NATIVE_SPARK_ML=true" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  --set-env-vars="STREAMLIT_SERVER_PORT=8080" \
  --set-env-vars="STREAMLIT_SERVER_ADDRESS=0.0.0.0" \
  --set-env-vars="STREAMLIT_SERVER_HEADLESS=true" \
  --set-env-vars="STREAMLIT_SERVER_ENABLE_CORS=false" \
  --set-env-vars="STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false" \
  --set-env-vars="STREAMLIT_SERVER_RUN_ON_SAVE=false" \
  --set-env-vars="STREAMLIT_SERVER_FILE_WATCHER_TYPE=none" \
  --set-env-vars="DATAPROC_CONTAINER_IMAGE=${DATAPROC_IMAGE}"

if [ $? -ne 0 ]; then
    echo "❌ Cloud Run deployment failed!"
    echo "🔍 Check the logs for more details:"
    echo "   gcloud logs read --project ${PROJECT_ID} --filter resource.type=cloud_run_revision"
    exit 1
fi

# Ensure the latest revision gets 100% traffic automatically
echo "🚦 Ensuring latest revision receives 100% traffic..."
gcloud run services update-traffic ${SERVICE_NAME} \
  --to-latest \
  --region ${REGION} \
  --project ${PROJECT_ID}

if [ $? -eq 0 ]; then
    echo "✅ Traffic successfully routed to latest revision"
else
    echo "⚠️ Warning: Traffic routing may need manual adjustment"
fi

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
sleep 30

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format="value(status.url)")

if [ -z "$SERVICE_URL" ]; then
    echo "❌ Failed to get service URL!"
    exit 1
fi

echo ""
echo "✅ AutoML with Dataproc Serverless deployed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🔧 Dataproc Serverless Configuration:"
echo "   - Spark jobs will run on Dataproc Serverless with custom runtime"
echo "   - Custom Container Image: ${DATAPROC_IMAGE}"
echo "   - Pre-installed: Python 3.13, Java 21, PySpark 3.5.6, XGBoost, LightGBM, SynapseML"
echo "   - Automatic scaling (0 to thousands of executors)"
echo "   - Pay-per-job pricing model"
echo "   - No cluster management required"
echo "   - 5-15 minute faster job startup (no runtime package installation)"
echo "   - Temp Bucket: ${TEMP_BUCKET}"
echo "   - Results Bucket: ${RESULTS_BUCKET}"
echo ""
echo "📊 Monitor your service:"
echo "   gcloud run services describe ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "📝 View logs:"
echo "   gcloud logs read --project ${PROJECT_ID} --filter resource.type=cloud_run_revision"
echo ""
echo "🔍 Check service health:"
echo "   curl -f ${SERVICE_URL}/_stcore/health"
echo ""
echo "🧪 Test the service:"
echo "   open ${SERVICE_URL}"
