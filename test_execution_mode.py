#!/usr/bin/env python3
"""
Test script to verify execution mode detection
"""

import sys
import os

# Add the automl_pyspark directory to the path
sys.path.insert(0, '/Users/sundar/Downloads/rapid-modeler/automl_pyspark')

print("🧪 Testing execution mode detection...")

# Test the configure_execution_environment function
def configure_execution_environment():
    """Configure execution environment based on deployment context."""
    
    # Check if running in Cloud Run (GCP deployment)
    is_cloud_run = (
        os.environ.get('K_SERVICE') is not None or  # Cloud Run service name
        os.environ.get('GOOGLE_CLOUD_PROJECT') is not None or  # GCP project
        os.environ.get('GAE_APPLICATION') is not None or  # App Engine
        'run.app' in os.environ.get('SERVER_SOFTWARE', '')  # Cloud Run server
    )
    
    print(f"🔍 Environment detection:")
    print(f"   K_SERVICE: {os.environ.get('K_SERVICE', 'not set')}")
    print(f"   GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'not set')}")
    print(f"   GAE_APPLICATION: {os.environ.get('GAE_APPLICATION', 'not set')}")
    print(f"   SERVER_SOFTWARE: {os.environ.get('SERVER_SOFTWARE', 'not set')}")
    print(f"   is_cloud_run: {is_cloud_run}")
    
    if is_cloud_run:
        print("☁️ Detected Cloud Run deployment - enabling Dataproc Serverless mode")
        
        # Load Dataproc environment variables
        env_file = os.path.join(os.path.dirname(__file__), 'env.dataproc')
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                print(f"✅ Loaded Dataproc environment variables from {env_file}")
                
                # Enable Dataproc Serverless for Cloud Run deployment
                os.environ['ENABLE_DATAPROC_SERVERLESS'] = 'true'
                print(f"   GCP_PROJECT_ID: {os.environ.get('GCP_PROJECT_ID', 'not set')}")
                print(f"   GCP_REGION: {os.environ.get('GCP_REGION', 'not set')}")
                
            except Exception as e:
                print(f"⚠️ Failed to load Dataproc environment variables: {e}")
        else:
            print(f"⚠️ Dataproc environment file not found: {env_file}")
    else:
        print("🏠 Detected local deployment - using local execution mode")
        # Ensure Dataproc is disabled for local runs
        os.environ['ENABLE_DATAPROC_SERVERLESS'] = 'false'
        os.environ['USE_DATAPROC_SERVERLESS'] = 'false'

# Test local execution
print("\n1. Testing local execution mode:")
configure_execution_environment()

print(f"\n📋 Final environment variables:")
print(f"   ENABLE_DATAPROC_SERVERLESS: {os.environ.get('ENABLE_DATAPROC_SERVERLESS', 'not set')}")
print(f"   USE_DATAPROC_SERVERLESS: {os.environ.get('USE_DATAPROC_SERVERLESS', 'not set')}")

# Test job manager import
print(f"\n2. Testing job manager import:")
try:
    from background_job_manager import job_manager
    print("✅ Background job manager imported successfully")
    if job_manager:
        execution_mode = "Dataproc Serverless" if job_manager.use_dataproc_serverless else "Local Background Threads"
        print(f"   Execution mode: {execution_mode}")
        print(f"   Jobs directory: {job_manager.jobs_dir}")
except ImportError as e:
    print(f"❌ Failed to import background job manager: {e}")

# Test simulated Cloud Run environment
print(f"\n3. Testing simulated Cloud Run environment:")
os.environ['K_SERVICE'] = 'rapid-modeler'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'atus-prism-dev'
configure_execution_environment()

print(f"\n📋 Final environment variables (Cloud Run simulation):")
print(f"   ENABLE_DATAPROC_SERVERLESS: {os.environ.get('ENABLE_DATAPROC_SERVERLESS', 'not set')}")
print(f"   GCP_PROJECT_ID: {os.environ.get('GCP_PROJECT_ID', 'not set')}")
