#!/usr/bin/env python3
"""
Test script to verify Dataproc deployment and run a simple job
"""

import sys
import os
sys.path.append('/Users/sundar/Downloads/rapid-modeler/automl_pyspark')

from dataproc_background_job_manager import DataprocBackgroundJobManager
import time
import json

def test_dataproc_deployment():
    """Test the Dataproc deployment with a simple job"""
    
    print("🧪 Testing Dataproc Deployment...")
    
    # Create a simple test job configuration
    job_config = {
        'job_id': f'test_deployment_{int(time.time())}',
        'task_type': 'classification',
        'data_source': '/Users/sundar/Downloads/rapid-modeler/automl_pyspark/bank.csv',
        'target_column': 'deposit',
        'source_type': 'existing',
        'intelligent_sampling': True,
        'sample_size': 1000,  # Small sample for quick testing
        'feature_profiling': True,
        'models_to_run': ['logistic_regression'],  # Just one model for quick test
        'hyperparameter_tuning': False,  # Skip tuning for speed
        'cross_validation_folds': 2,  # Minimal CV
        'test_size': 0.2,
        'random_state': 42
    }
    
    print(f"📋 Job Configuration:")
    print(json.dumps(job_config, indent=2))
    
    try:
        # Initialize the job manager
        print("\n🔧 Initializing Dataproc Job Manager...")
        job_manager = DataprocBackgroundJobManager()
        
        # Submit the job
        print("\n🚀 Submitting test job to Dataproc...")
        job_id = job_manager.submit_job(job_config)
        
        if job_id:
            print(f"✅ Job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            
            # Monitor the job for a few minutes
            print("\n⏳ Monitoring job status...")
            max_wait_time = 600  # 10 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status = job_manager.get_job_status(job_id)
                print(f"🔍 Job Status: {status}")
                
                if status in ['COMPLETED', 'SUCCEEDED']:
                    print("✅ Job completed successfully!")
                    
                    # Get results
                    results = job_manager.get_job_results(job_id)
                    if results:
                        print("📊 Job Results:")
                        print(json.dumps(results, indent=2))
                    break
                    
                elif status in ['FAILED', 'CANCELLED', 'ERROR']:
                    print(f"❌ Job failed with status: {status}")
                    
                    # Get logs for debugging
                    logs = job_manager.get_job_logs(job_id)
                    if logs:
                        print("📝 Job Logs:")
                        print(logs[-2000:])  # Last 2000 characters
                    break
                    
                elif status in ['RUNNING', 'PENDING', 'QUEUED']:
                    print(f"⏳ Job is {status.lower()}... waiting 30 seconds")
                    time.sleep(30)
                    
                else:
                    print(f"🤔 Unknown status: {status}")
                    time.sleep(30)
            
            else:
                print("⏰ Job monitoring timed out after 10 minutes")
                
        else:
            print("❌ Failed to submit job")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Starting Dataproc Deployment Test...")
    success = test_dataproc_deployment()
    
    if success:
        print("\n✅ Dataproc deployment test completed!")
    else:
        print("\n❌ Dataproc deployment test failed!")
        sys.exit(1)
