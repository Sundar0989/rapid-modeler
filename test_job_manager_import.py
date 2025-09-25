#!/usr/bin/env python3
"""
Test script to debug job manager import issues
"""

import sys
import os

# Add the automl_pyspark directory to the path
sys.path.insert(0, '/Users/sundar/Downloads/rapid-modeler/automl_pyspark')

print("🧪 Testing job manager import...")

try:
    print("1. Testing BackgroundJobManager import...")
    from background_job_manager import BackgroundJobManager
    print("✅ BackgroundJobManager imported successfully")
    
    print("2. Testing job_manager instance import...")
    from background_job_manager import job_manager
    print("✅ job_manager instance imported successfully")
    print(f"   job_manager type: {type(job_manager)}")
    print(f"   job_manager is None: {job_manager is None}")
    
    if job_manager is not None:
        print("3. Testing job_manager methods...")
        print(f"   use_dataproc_serverless: {job_manager.use_dataproc_serverless}")
        print(f"   use_gcp_queue: {job_manager.use_gcp_queue}")
        print(f"   jobs_dir: {job_manager.jobs_dir}")
        
        # Test a simple method
        try:
            active_jobs = job_manager.get_active_jobs()
            print(f"   active_jobs: {len(active_jobs) if active_jobs else 0}")
        except Exception as e:
            print(f"   ❌ Error getting active jobs: {e}")
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Environment variables:")
print(f"   USE_GCP_QUEUE: {os.getenv('USE_GCP_QUEUE', 'not set')}")
print(f"   ENABLE_DATAPROC_SERVERLESS: {os.getenv('ENABLE_DATAPROC_SERVERLESS', 'not set')}")
print(f"   USE_DATAPROC_SERVERLESS: {os.getenv('USE_DATAPROC_SERVERLESS', 'not set')}")
print(f"   GCP_PROJECT_ID: {os.getenv('GCP_PROJECT_ID', 'not set')}")
print(f"   GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT', 'not set')}")
