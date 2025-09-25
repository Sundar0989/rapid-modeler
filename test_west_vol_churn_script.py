#!/usr/bin/env python3
"""
AutoML Job Script for job_0003_west_vol_churn_baseline_model_v0_1758765831
Generated at: 2025-09-25T02:03:55.128458
Execution Mode: Dataproc Serverless
Batch ID: automl-spark-job-0003-west-vol-churn-baseline-model-v0-17587658
"""

import sys
import os
import json
import pandas as pd
import signal
import atexit
from datetime import datetime

# Add the automl_pyspark directory to Python path
sys.path.insert(0, '/app')
# Also add the parent directory to ensure automl_pyspark package is importable
sys.path.insert(0, '/tmp')
sys.path.insert(0, os.path.dirname('/app'))

# Global variable to track Spark session
spark_session = None

# Job configuration
JOB_ID = "job_0003_west_vol_churn_baseline_model_v0_1758765831"
BATCH_ID = "automl-spark-job-0003-west-vol-churn-baseline-model-v0-17587658"

# GCS bucket configuration for Dataproc
GCS_DATA_BUCKET = "rapid_modeler_app"
GCS_RESULTS_BUCKET = "rapid_modeler_app"
EXECUTION_MODE = "dataproc"  # Flag to indicate Dataproc execution mode

# Set up logging and progress tracking for Dataproc
def log_message(job_id, message):
    """Log a message with timestamp for Dataproc execution."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    # Also write to stdout for Dataproc logging
    sys.stdout.flush()

def update_progress(job_id, step, total_steps, message):
    """Update job progress for Dataproc execution."""
    progress = round((step / total_steps) * 100, 1)
    log_message(job_id, f"ðŸ“Š Progress: {progress}% - {message}")
    
# ========================================================================
# INTEGRATED DATAPROC CLEANUP AND TIMEOUT PROTECTION
# ========================================================================

def setup_cleanup_handlers():
    """Setup cleanup handlers to prevent container hangs."""
    
    def cleanup_handler(signum=None, frame=None):
        """Cleanup handler for graceful shutdown."""
        log_message(JOB_ID, f"ðŸ§¹ Cleanup handler triggered (signal: {signum})")
        
        # Only perform aggressive cleanup on actual termination signals
        if signum in [signal.SIGTERM, signal.SIGINT]:
            try:
                # Force Spark session cleanup
                if spark_session:
                    log_message(JOB_ID, "ðŸ”„ Force closing Spark session...")
                    spark_session.stop()
                    log_message(JOB_ID, "âœ… Spark session closed")
            except Exception as e:
                log_message(JOB_ID, f"âš ï¸ Error during Spark cleanup: {e}")
            
            try:
                # Graceful exit instead of force exit
                log_message(JOB_ID, "ðŸšª Graceful shutdown...")
                sys.exit(0)
            except:
                pass
        else:
            # For other cases, just log
            log_message(JOB_ID, "ðŸ§¹ Normal cleanup - job continuing...")
    
    # Register cleanup handlers
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    atexit.register(cleanup_handler)
    
    log_message(JOB_ID, "ðŸ›¡ï¸ Cleanup handlers registered")

def add_smart_termination_protection(max_timeout_minutes=180):
    """Add smart termination that triggers immediately after job completion."""
    
    def timeout_handler(signum, frame):
        log_message(JOB_ID, f"â° TIMEOUT: Job exceeded {max_timeout_minutes} minutes")
        log_message(JOB_ID, "ðŸš¨ Force terminating to prevent container hang...")
        
        # Force cleanup and exit
        try:
            if spark_session:
                spark_session.stop()
        except:
            pass
        
        os._exit(1)
    
    # Set maximum timeout alarm as safety net
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_timeout_minutes * 60)  # Convert to seconds
    
    log_message(JOB_ID, f"â° Maximum timeout protection set: {max_timeout_minutes} minutes")

def add_post_completion_force_shutdown(timeout_seconds=10):
    """Add aggressive force shutdown after job completion."""
    
    def force_shutdown_after_completion():
        import time
        time.sleep(timeout_seconds)
        log_message(JOB_ID, f"ðŸš¨ FORCE SHUTDOWN: Container still running after {timeout_seconds} seconds post-completion")
        log_message(JOB_ID, "ðŸš¨ Executing emergency termination...")
        
        try:
            import os
            import subprocess
            # Try multiple kill methods
            pid = os.getpid()
            subprocess.call(['kill', '-9', str(pid)])
            os._exit(1)
        except:
            pass
    
    import threading
    shutdown_thread = threading.Thread(target=force_shutdown_after_completion, daemon=True)
    shutdown_thread.start()
    log_message(JOB_ID, f"ðŸš¨ Emergency shutdown timer set: {timeout_seconds} seconds post-completion")

def _make_background_threads_daemon():
    """Mark all background threads as daemon to prevent blocking exit."""
    import threading
    try:
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.is_alive():
                thread.daemon = True
        log_message(JOB_ID, "âœ… Background threads marked as daemon")
    except Exception as e:
        log_message(JOB_ID, f"âš ï¸ Warning: Could not mark threads as daemon: {e}")

def immediate_termination_after_job_completion(spark_session=None, exit_code=0):
    """Terminate container immediately after job completion with comprehensive cleanup."""
    
    log_message(JOB_ID, "ðŸŽ‰ Dataproc Serverless job completed successfully!")

    # FINAL: ensure any background threads won't block, flush, then exit
    _make_background_threads_daemon()

    # attempt graceful exit first
    try:
        # stop spark cleanly
        if spark_session is not None:
            log_message(JOB_ID, "ðŸ§¹ Closing Spark session (graceful)...")
            spark_session.stop()
            log_message(JOB_ID, "âœ… Spark session closed successfully")
    except Exception as e:
        log_message(JOB_ID, f"âš ï¸ Error closing Spark session: {e}")

    log_message(JOB_ID, "ðŸ“Œ Job completed successfully - initiating immediate termination to avoid hang")
    
    # Flush output streams
    sys.stdout.flush()
    sys.stderr.flush()

    # SIMPLIFIED SHUTDOWN: Use os._exit directly for immediate termination
    log_message(JOB_ID, "ðŸšª Implementing simplified force shutdown mechanism...")
    
    try:
        import os
        log_message(JOB_ID, "ðŸšª Attempting os._exit...")
        # Use os._exit for immediate termination without cleanup
        os._exit(exit_code)
    except Exception as e:
        log_message(JOB_ID, f"âš ï¸ os._exit failed: {e}")
        
    # Fallback to sys.exit if os._exit fails
    try:
        log_message(JOB_ID, "ðŸšª Fallback to sys.exit...")
        sys.exit(exit_code)
    except Exception as e:
        log_message(JOB_ID, f"âš ï¸ sys.exit failed: {e}")
    
    # Final fallback - this should never be reached
    log_message(JOB_ID, "âš ï¸ All termination methods failed - container may hang")
    return exit_code

# ========================================================================
# LEGACY SIGNAL HANDLERS (kept for compatibility)
# ========================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    log_message(JOB_ID, f"ðŸ›‘ Received signal {signum}, shutting down gracefully...")
    if spark_session:
        try:
            spark_session.stop()
            log_message(JOB_ID, "âœ… Spark session stopped due to signal")
        except:
            pass
    sys.exit(0)

def cleanup_handler():
    """Cleanup handler for atexit."""
    if spark_session:
        try:
            spark_session.stop()
        except:
            pass

# Set up enhanced cleanup protection
setup_cleanup_handlers()
add_smart_termination_protection(180)  # 3 hours absolute max

# Set up legacy signal handlers (for compatibility)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

# Job configuration (already defined above)
JOB_CONFIG = {
  "job_id": "job_0003_west_vol_churn_baseline_model_v0_1758765831",
  "user_id": "west_vol_churn",
  "model_name": "baseline_model_v0",
  "task_type": "classification",
  "environment": "development",
  "preset": "",
  "data_file": "atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism",
  "target_column": "target",
  "output_dir": "/tmp/automl_results/job_0003_west_vol_churn_baseline_model_v0_1758765831",
  "timestamp": "2025-09-25T02:03:51.649063",
  "config_path": "/tmp/automl_pyspark/config.yaml",
  "model_params": {
  "run_logistic": False,
  "run_random_forest": False,
  "run_gradient_boosting": False,
  "run_decision_tree": False,
  "run_neural_network": False,
  "run_xgboost": True,
  "run_lightgbm": True
},
  "oot1_file": None,
  "oot2_file": None,
  "oot1_bigquery_table": "`atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism_oot1`",
  "oot2_bigquery_table": "`atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism_oot2`",
  "oot1_config": {
  "source_type": "bigquery",
  "data_source": "`atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism_oot1`",
  "options": {
  "bigquery_options": {
  "useAvroLogicalTypes": "true",
  "viewsEnabled": "true"
},
  "project_id": "atus-prism-dev"
}
},
  "oot2_config": {
  "source_type": "bigquery",
  "data_source": "`atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism_oot2`",
  "options": {
  "bigquery_options": {
  "useAvroLogicalTypes": "true",
  "viewsEnabled": "true"
},
  "project_id": "atus-prism-dev"
}
},
  "data_params": {
  "test_size": 0.2,
  "validation_size": 0.2
},
  "advanced_params": {
  "cv_folds": 5,
  "enable_hyperparameter_tuning": False,
  "auto_balance": True,
  "missing_threshold": 0.7,
  "categorical_threshold": 10,
  "sample_fraction": 1.0,
  "parallel_jobs": -1,
  "timeout_minutes": 120,
  "hp_method": None,
  "optuna_trials": None,
  "optuna_timeout": None,
  "random_trials": None,
  "classification_hp_ranges": {
  "logistic": {
  "maxIter": [10, 20, 50, 100],
  "regParam": [0.01, 0.1, 0.5, 1.0],
  "elasticNetParam": [0.0, 0.25, 0.5, 0.75, 1.0]
},
  "random_forest": {
  "numTrees": [10, 20, 50, 100],
  "maxDepth": [3, 5, 10, 15, 20],
  "minInstancesPerNode": [1, 2, 5, 10]
},
  "gradient_boosting": {
  "maxIter": [10, 20, 50, 100],
  "maxDepth": [3, 5, 10, 15],
  "stepSize": [0.1, 0.2, 0.3]
},
  "decision_tree": {
  "maxDepth": [3, 5, 10, 15, 20],
  "minInstancesPerNode": [1, 2, 5, 10],
  "minInfoGain": [0.0, 0.01, 0.1]
},
  "neural_network": {
  "layers": [[10], [20], [10, 10]],
  "maxIter": [50, 100, 200],
  "blockSize": [32, 64, 128]
},
  "xgboost": {
  "maxDepth": [3, 5, 7, 10],
  "numRound": [50, 100, 150, 200],
  "eta": [0.05, 0.1, 0.15, 0.2],
  "subsample": [0.8, 0.9, 1.0],
  "colsampleBytree": [0.8, 0.9, 1.0],
  "minChildWeight": [1, 2, 3, 5]
},
  "lightgbm": {
  "numLeaves": [15, 25, 31, 50],
  "numIterations": [50, 100, 150, 200],
  "learningRate": [0.05, 0.1, 0.15, 0.2],
  "featureFraction": [0.8, 0.9, 1.0],
  "baggingFraction": [0.8, 0.9, 1.0]
}
},
  "include_vars": [],
  "exclude_vars": ["CHC_ID", "SNAPSHOT_DAY_ID"],
  "include_prefix": [],
  "exclude_prefix": [],
  "include_suffix": [],
  "exclude_suffix": []
},
  "enhanced_data_config": {
  "source_type": "bigquery",
  "data_source": "atus-prism-dev.ds_sandbox.sundar_WEST_b2c_fixed_churn_train_prism",
  "options": {
  "bigquery_options": {
  "useAvroLogicalTypes": "true",
  "viewsEnabled": "true"
},
  "project_id": "atus-prism-dev"
},
  "size_check_passed": True
},
  "data_size_mb": 2775.809956550598,
  "estimated_rows": 1455123,
  "gcs_results_bucket": "rapid_modeler_app",
  "gcs_temp_bucket": "rapid_modeler_app"
}

def load_oot_datasets(config, data_manager=None):
    """Helper function to load OOT datasets based on configuration."""
    oot1_data = None
    oot2_data = None
    oot1_file = None
    oot2_file = None
    
    # Detect execution mode - if EXECUTION_MODE is 'local', skip GCS path conversion
    is_local_execution = globals().get('EXECUTION_MODE') == 'local'
    
    # Handle OOT1 - check for BigQuery table first, then file
    if config.get('oot1_bigquery_table'):
        log_message(JOB_ID, f"ðŸ“… Loading OOT1 data from BigQuery: {config['oot1_bigquery_table']}")
        try:
            # Use data_manager.load_data for BigQuery tables (more robust)
            oot1_data, _ = data_manager.load_data(
                config['oot1_bigquery_table'],
                source_type='bigquery',
                feature_engineering_phase=False,  # No feature engineering for OOT data
                enable_intelligent_sampling=False  # Load full OOT data
            )
            log_message(JOB_ID, f"âœ… OOT1 BigQuery data loaded: {oot1_data.count()} rows")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ OOT1 BigQuery loading failed: {str(e)} - will skip OOT1")
            oot1_data = None
    
    elif config.get('oot1_file') or (config.get('enhanced_data_config') and config['enhanced_data_config'].get('oot1_file')):
        # Check both root level and enhanced_data_config
        oot1_file = config.get('oot1_file') or config['enhanced_data_config'].get('oot1_file')
        log_message(JOB_ID, f"ðŸ“… Loading OOT1 data from file: {oot1_file}")
        try:
            # Use the same source_type as the main data if available
            oot1_source_type = "existing"  # Default fallback
            if config.get('oot1_config') and config['oot1_config'].get('source_type'):
                oot1_source_type = config['oot1_config']['source_type']
            elif config.get('enhanced_data_config') and config['enhanced_data_config'].get('source_type'):
                oot1_source_type = config['enhanced_data_config']['source_type']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"ðŸ  Local execution - loading OOT1 via data_manager: {oot1_file}")
                try:
                    oot1_data, _ = data_manager.load_data(oot1_file, source_type=oot1_source_type)
                    log_message(JOB_ID, f"âœ… OOT1 file data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT1 file loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
            else:
                # Dataproc execution - convert to GCS paths and use Spark directly
                if not oot1_file.startswith('gs://'):
                    if oot1_source_type == 'upload':
                        oot1_file = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{oot1_file}"
                        log_message(JOB_ID, f"ðŸ“ Upload OOT1 - converted to GCS path: {oot1_file}")
                    elif oot1_source_type == 'existing':
                        oot1_file = f"gs://{GCS_DATA_BUCKET}/data/{oot1_file}"
                        log_message(JOB_ID, f"ðŸ“ Existing OOT1 - converted to GCS path: {oot1_file}")
                else:
                    log_message(JOB_ID, f"ðŸ“ Using existing GCS path for OOT1: {oot1_file}")
                
                # For Dataproc, use direct Spark loading for GCS paths
                log_message(JOB_ID, f"ðŸ”„ Using direct Spark loading for OOT1...")
                oot1_data = None
                try:
                    from pyspark.sql import SparkSession
                    spark = SparkSession.getActiveSession()
                    if spark and oot1_file.endswith('.csv'):
                        oot1_data = spark.read.csv(oot1_file, header=True, inferSchema=True)
                        log_message(JOB_ID, f"âœ… OOT1 loaded via direct Spark")
                    else:
                        log_message(JOB_ID, f"âš ï¸ Cannot load OOT1 data: unsupported format or no Spark session")
                        oot1_data = None
                except Exception as spark_error:
                    log_message(JOB_ID, f"âš ï¸ Direct Spark loading failed: {{str(spark_error)}}")
                    oot1_data = None
            
            if oot1_data:
                try:
                    row_count = oot1_data.count()
                    log_message(JOB_ID, f"âœ… OOT1 file data loaded: {row_count} rows")
                    log_message(JOB_ID, f"âœ… OOT1 columns: {oot1_data.columns}")
                except Exception as count_error:
                    log_message(JOB_ID, f"âš ï¸ OOT1 data loaded but count failed: {str(count_error)}")
            else:
                log_message(JOB_ID, "âš ï¸ OOT1 data is None after loading attempt")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ OOT1 file loading failed: {str(e)} - will skip OOT1")
            import traceback
            log_message(JOB_ID, f"âš ï¸ OOT1 traceback: {traceback.format_exc()}")
    
    elif config.get('oot1_config'):
        oot1_config = config['oot1_config']
        if oot1_config.get('data_source'):
            source_type = oot1_config.get('source_type', 'existing')
            data_source = oot1_config['data_source']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"ðŸ  Local execution - loading OOT1 config via data_manager: {data_source}")
                try:
                    oot1_data, _ = data_manager.load_data(
                        data_source, 
                        source_type=source_type,
                        **oot1_config.get('options', {})
                    )
                    log_message(JOB_ID, f"âœ… OOT1 data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT1 loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
            else:
                # Dataproc execution - convert to GCS paths
                if not data_source.startswith('gs://'):
                    if source_type == 'upload':
                        data_source = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{data_source}"
                        log_message(JOB_ID, f"ðŸ“ Upload OOT1 config - converted to GCS path: {data_source}")
                    elif source_type == 'existing':
                        data_source = f"gs://{GCS_DATA_BUCKET}/data/{data_source}"
                        log_message(JOB_ID, f"ðŸ“ Existing OOT1 config - converted to GCS path: {data_source}")
                else:
                    log_message(JOB_ID, f"ðŸ“ Using existing GCS path for OOT1 config: {data_source}")
                
                log_message(JOB_ID, f"ðŸ“… Loading OOT1 data from {source_type}: {data_source}")
                try:
                    # For Dataproc, load data directly using Spark if data_manager is not available
                    if data_manager:
                        oot1_data, _ = data_manager.load_data(
                            data_source, 
                            source_type=source_type,
                            **oot1_config.get('options', {})
                        )
                    else:
                        # Fallback to direct Spark loading for Dataproc
                        from pyspark.sql import SparkSession
                        spark = SparkSession.getActiveSession()
                        if spark and data_source.endswith('.csv'):
                            oot1_data = spark.read.csv(data_source, header=True, inferSchema=True)
                        else:
                            log_message(JOB_ID, f"âš ï¸ Cannot load OOT1 config data: unsupported format or no Spark session")
                            oot1_data = None
                    log_message(JOB_ID, f"âœ… OOT1 data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT1 loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
    
    # Handle OOT2 - similar logic
    if config.get('oot2_bigquery_table'):
        log_message(JOB_ID, f"ðŸ“… Loading OOT2 data from BigQuery: {config['oot2_bigquery_table']}")
        try:
            # Use data_manager.load_data for BigQuery tables (more robust)
            oot2_data, _ = data_manager.load_data(
                config['oot2_bigquery_table'],
                source_type='bigquery',
                feature_engineering_phase=False,  # No feature engineering for OOT data
                enable_intelligent_sampling=False  # Load full OOT data
            )
            log_message(JOB_ID, f"âœ… OOT2 BigQuery data loaded: {oot2_data.count()} rows")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ OOT2 BigQuery loading failed: {str(e)} - will skip OOT2")
            oot2_data = None
    
    elif config.get('oot2_file') or (config.get('enhanced_data_config') and config['enhanced_data_config'].get('oot2_file')):
        # Check both root level and enhanced_data_config
        oot2_file = config.get('oot2_file') or config['enhanced_data_config'].get('oot2_file')
        log_message(JOB_ID, f"ðŸ“… Loading OOT2 data from file: {oot2_file}")
        try:
            # Use the same source_type as the main data if available
            oot2_source_type = "existing"  # Default fallback
            if config.get('oot2_config') and config['oot2_config'].get('source_type'):
                oot2_source_type = config['oot2_config']['source_type']
            elif config.get('enhanced_data_config') and config['enhanced_data_config'].get('source_type'):
                oot2_source_type = config['enhanced_data_config']['source_type']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"ðŸ  Local execution - loading OOT2 via data_manager: {oot2_file}")
                try:
                    oot2_data, _ = data_manager.load_data(oot2_file, source_type=oot2_source_type)
                    log_message(JOB_ID, f"âœ… OOT2 file data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT2 file loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
            else:
                # Dataproc execution - convert to GCS paths and use Spark directly
                if not oot2_file.startswith('gs://'):
                    if oot2_source_type == 'upload':
                        oot2_file = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{oot2_file}"
                        log_message(JOB_ID, f"ðŸ“ Upload OOT2 - converted to GCS path: {oot2_file}")
                    elif oot2_source_type == 'existing':
                        oot2_file = f"gs://{GCS_DATA_BUCKET}/data/{oot2_file}"
                        log_message(JOB_ID, f"ðŸ“ Existing OOT2 - converted to GCS path: {oot2_file}")
                else:
                    log_message(JOB_ID, f"ðŸ“ Using existing GCS path for OOT2: {oot2_file}")
                
                # For Dataproc, use direct Spark loading for GCS paths
                log_message(JOB_ID, f"ðŸ”„ Using direct Spark loading for OOT2...")
                oot2_data = None
                try:
                    from pyspark.sql import SparkSession
                    spark = SparkSession.getActiveSession()
                    if spark and oot2_file.endswith('.csv'):
                        oot2_data = spark.read.csv(oot2_file, header=True, inferSchema=True)
                        log_message(JOB_ID, f"âœ… OOT2 loaded via direct Spark")
                    else:
                        log_message(JOB_ID, f"âš ï¸ Cannot load OOT2 data: unsupported format or no Spark session")
                        oot2_data = None
                except Exception as spark_error:
                    log_message(JOB_ID, f"âš ï¸ Direct Spark loading failed: {{str(spark_error)}}")
                    oot2_data = None
            
            if oot2_data:
                try:
                    row_count = oot2_data.count()
                    log_message(JOB_ID, f"âœ… OOT2 file data loaded: {row_count} rows")
                    log_message(JOB_ID, f"âœ… OOT2 columns: {oot2_data.columns}")
                except Exception as count_error:
                    log_message(JOB_ID, f"âš ï¸ OOT2 data loaded but count failed: {str(count_error)}")
            else:
                log_message(JOB_ID, "âš ï¸ OOT2 data is None after loading attempt")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ OOT2 file loading failed: {str(e)} - will skip OOT2")
            import traceback
            log_message(JOB_ID, f"âš ï¸ OOT2 traceback: {traceback.format_exc()}")
    
    elif config.get('oot2_config'):
        oot2_config = config['oot2_config']
        if oot2_config.get('data_source'):
            source_type = oot2_config.get('source_type', 'existing')
            data_source = oot2_config['data_source']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"ðŸ  Local execution - loading OOT2 config via data_manager: {data_source}")
                try:
                    oot2_data, _ = data_manager.load_data(
                        data_source, 
                        source_type=source_type,
                        **oot2_config.get('options', {})
                    )
                    log_message(JOB_ID, f"âœ… OOT2 data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT2 loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
            else:
                # Dataproc execution - convert to GCS paths
                if not data_source.startswith('gs://'):
                    if source_type == 'upload':
                        data_source = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{data_source}"
                        log_message(JOB_ID, f"ðŸ“ Upload OOT2 config - converted to GCS path: {data_source}")
                    elif source_type == 'existing':
                        data_source = f"gs://{GCS_DATA_BUCKET}/data/{data_source}"
                        log_message(JOB_ID, f"ðŸ“ Existing OOT2 config - converted to GCS path: {data_source}")
                else:
                    log_message(JOB_ID, f"ðŸ“ Using existing GCS path for OOT2 config: {data_source}")
                
                log_message(JOB_ID, f"ðŸ“… Loading OOT2 data from {source_type}: {data_source}")
                try:
                    # For Dataproc, load data directly using Spark if data_manager is not available
                    if data_manager:
                        oot2_data, _ = data_manager.load_data(
                            data_source, 
                            source_type=source_type,
                            **oot2_config.get('options', {})
                        )
                    else:
                        # Fallback to direct Spark loading for Dataproc
                        from pyspark.sql import SparkSession
                        spark = SparkSession.getActiveSession()
                        if spark and data_source.endswith('.csv'):
                            oot2_data = spark.read.csv(data_source, header=True, inferSchema=True)
                        else:
                            log_message(JOB_ID, f"âš ï¸ Cannot load OOT2 config data: unsupported format or no Spark session")
                            oot2_data = None
                    log_message(JOB_ID, f"âœ… OOT2 data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ OOT2 loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
    
    return oot1_data, oot2_data

def main():
    """Main job execution for Dataproc Serverless."""
    global spark_session
    
    try:
        log_message(JOB_ID, f"ðŸš€ Starting Dataproc Serverless job: job_0003_west_vol_churn_baseline_model_v0_1758765831")
        log_message(JOB_ID, f"ðŸ“Š Batch ID: automl-spark-job-0003-west-vol-churn-baseline-model-v0-17587658")
        log_message(JOB_ID, f"ðŸ“ Working directory: " + os.getcwd())
        
        update_progress(JOB_ID, 1, 8, "Initializing Spark session...")
        
        # Initialize Spark for Dataproc with optimized settings
        from pyspark.sql import SparkSession
        
        # Create Spark session with Dataproc-optimized settings
        # Resource allocation is handled by Dataproc Serverless configuration
        spark_session = SparkSession.builder \
            .appName(f"AutoML_job_0003_west_vol_churn_baseline_model_v0_1758765831") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .getOrCreate()
        
        # Log the actual Spark configuration used
        log_message(JOB_ID, f"ðŸ§  Spark resource allocation:")
        log_message(JOB_ID, f"   Executors: {spark_session.conf.get('spark.executor.instances', 'auto')}")
        log_message(JOB_ID, f"   Executor Memory: {spark_session.conf.get('spark.executor.memory', 'auto')}")
        log_message(JOB_ID, f"   Driver Memory: {spark_session.conf.get('spark.driver.memory', 'auto')}")
        
        log_message(JOB_ID, "âœ… Spark session initialized for Dataproc Serverless")
        log_message(JOB_ID, f"ðŸ“Š Spark UI available at: {spark_session.sparkContext.uiWebUrl}")
        log_message(JOB_ID, f"ðŸ”§ Spark configuration: driver.memory={spark_session.conf.get('spark.driver.memory', 'default')}, executor.memory={spark_session.conf.get('spark.executor.memory', 'default')}")
        
        # Skip package installation - using custom container image with pre-installed packages
        log_message(JOB_ID, "ðŸ“¦ Using custom container image with pre-installed packages...")
        log_message(JOB_ID, "â„¹ï¸ Skipping runtime package installation - all dependencies are pre-installed")
        
        # Ensure config file is available - use automl_pyspark package config
        config_source_path = "/app/automl_pyspark/config.yaml"
        
        # Check if config file exists in the expected location
        if os.path.exists(config_source_path):
            log_message(JOB_ID, f"âœ… Config file found at: {config_source_path}")
        else:
            log_message(JOB_ID, f"âš ï¸ Config file not found at {config_source_path}, will use defaults")
        
        # Set up results directory - ensure it's in the mounted volume
        results_dir = "/tmp/automl_results/job_0003_west_vol_churn_baseline_model_v0_1758765831"
        
        # For Docker environments, ensure we use the mounted volume path
        if os.path.exists('/app/automl_pyspark/automl_results'):
            # We're in Docker with mounted volumes
            mounted_results_dir = f"/app/automl_pyspark/automl_results/{JOB_ID}"
            os.makedirs(mounted_results_dir, exist_ok=True)
            # Create symlink from results_dir to mounted_results_dir if they're different
            if results_dir != mounted_results_dir:
                try:
                    if os.path.exists(results_dir):
                        os.rmdir(results_dir)
                    os.symlink(mounted_results_dir, results_dir)
                    log_message(JOB_ID, f"ðŸ”— Created symlink: {results_dir} -> {mounted_results_dir}")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ Could not create symlink, using mounted path directly: {e}")
                    results_dir = mounted_results_dir
        else:
            os.makedirs(results_dir, exist_ok=True)
        
        log_message(JOB_ID, f"ðŸ“ Results will be saved to: {results_dir}")
        
        # Debug: List available files in mounted directories
        try:
            log_message(JOB_ID, "ðŸ” Debugging mounted directories:")
            
            # Check results directory
            if os.path.exists("/app/automl_pyspark/automl_results"):
                results_contents = os.listdir("/app/automl_pyspark/automl_results")
                log_message(JOB_ID, f"ðŸ“‚ /app/automl_pyspark/automl_results contains: {results_contents}")
            
            # Check jobs directory  
            if os.path.exists("/app/automl_pyspark/automl_jobs"):
                jobs_contents = os.listdir("/app/automl_pyspark/automl_jobs")
                log_message(JOB_ID, f"ðŸ“‚ /app/automl_pyspark/automl_jobs contains: {jobs_contents}")
            
            # Check specific job directory if it exists
            job_specific_dir = f"/app/automl_pyspark/automl_results/{JOB_ID}"
            if os.path.exists(job_specific_dir):
                job_contents = os.listdir(job_specific_dir)
                log_message(JOB_ID, f"ðŸ“‚ {job_specific_dir} contains: {job_contents}")
            else:
                log_message(JOB_ID, f"ðŸ“‚ {job_specific_dir} does not exist yet")
            
            # Check the actual results_dir being used
            if os.path.exists(results_dir):
                actual_contents = os.listdir(results_dir)
                log_message(JOB_ID, f"ðŸ“‚ Active results_dir {results_dir} contains: {actual_contents}")
            else:
                log_message(JOB_ID, f"ðŸ“‚ Active results_dir {results_dir} does not exist yet")
                
        except Exception as debug_e:
            log_message(JOB_ID, f"âš ï¸ Debug listing failed: {debug_e}")
        
        update_progress(JOB_ID, 2, 8, "Importing AutoML classes...")
        
        # Import AutoML classes with error handling
        log_message(JOB_ID, "ðŸ“¦ Importing AutoML classes...")
        try:
            from automl_pyspark.classification.automl_classifier import AutoMLClassifier
            from automl_pyspark.regression.automl_regressor import AutoMLRegressor  
            from automl_pyspark.clustering.automl_clusterer import AutoMLClusterer
            log_message(JOB_ID, "âœ… AutoML classes imported successfully")
        except Exception as e:
            log_message(JOB_ID, f"âŒ Failed to import AutoML classes: {e}")
            raise
        
        # Apply Dataproc compatibility fixes directly
        log_message(JOB_ID, "ðŸ”§ Applying Dataproc compatibility fixes...")
        
        # Override pandas to_excel globally to handle openpyxl gracefully
        try:
            import pandas as pd
            original_to_excel = pd.DataFrame.to_excel
            
            def safe_to_excel(self, excel_writer, sheet_name='Sheet1', **kwargs):
                try:
                    return original_to_excel(self, excel_writer, sheet_name, **kwargs)
                except ImportError as e:
                    if 'openpyxl' in str(e):
                        # Extract filename from excel_writer if it's a string
                        if isinstance(excel_writer, str):
                            csv_path = excel_writer.replace('.xlsx', '.csv').replace('.xls', '.csv')
                            log_message(JOB_ID, f"âš ï¸ Excel export failed: {e}")
                            log_message(JOB_ID, f"ðŸ”„ Falling back to CSV: {csv_path}")
                            result = self.to_csv(csv_path, index=kwargs.get('index', True))
                            log_message(JOB_ID, f"âœ… Data saved to CSV instead: {csv_path}")
                            return result
                        else:
                            raise
                    else:
                        raise
            
            pd.DataFrame.to_excel = safe_to_excel
            log_message(JOB_ID, "âœ… Applied pandas Excel fallback patch")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ Could not patch pandas to_excel: {e}")
        
        # Override SHAP computation globally
        try:
            class MockShapModule:
                def __getattr__(self, name):
                    def mock_function(*args, **kwargs):
                        log_message(JOB_ID, f"âš ï¸ SHAP function '{name}' called but SHAP library not available")
                        log_message(JOB_ID, f"   ðŸ’¡ Install with: pip install shap>=0.40.0")
                        if name == 'KernelExplainer':
                            # Return a mock explainer that will fail gracefully
                            class MockExplainer:
                                def __init__(self, *args, **kwargs):
                                    pass
                                def shap_values(self, *args, **kwargs):
                                    raise ImportError("SHAP library not available")
                            return MockExplainer
                        return None
                    return mock_function
            
            # Only mock if shap is not available
            try:
                import shap
                log_message(JOB_ID, "âœ… SHAP library is available")
            except ImportError:
                import sys
                sys.modules['shap'] = MockShapModule()
                log_message(JOB_ID, "âœ… Applied SHAP mock for graceful handling")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ Could not setup SHAP handling: {e}")
        
        log_message(JOB_ID, "âœ… Dataproc compatibility fixes applied")
        
        update_progress(JOB_ID, 3, 8, "Initializing AutoML...")
        
        # Extract job parameters
        data_file = JOB_CONFIG.get('data_file', 'data.csv')
        target_column = JOB_CONFIG.get('target_column')
        task_type = JOB_CONFIG.get('task_type', 'classification')
        
        # For Dataproc, use GCS paths for all data files
        enhanced_data_config = JOB_CONFIG.get('enhanced_data_config', {})
        source_type = enhanced_data_config.get('source_type', 'existing')
        
        # Convert data file to GCS path based on source type (only if not already a GCS path)
        if not data_file.startswith('gs://'):
            if source_type == 'upload':
                # For uploaded files, they're stored in the results bucket
                data_file = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{data_file}"
                log_message(JOB_ID, f"ðŸ“ Upload job - converted to GCS path: {data_file}")
            elif source_type == 'existing':
                # For existing files, they're in the data directory
                data_file = f"gs://{GCS_DATA_BUCKET}/data/{data_file}"
                log_message(JOB_ID, f"ðŸ“ Existing file job - converted to GCS path: {data_file}")
            else:
                log_message(JOB_ID, f"ðŸ“ Using original data file path: {data_file}")
        else:
            log_message(JOB_ID, f"ðŸ“ Using existing GCS path: {data_file}")
        
        log_message(JOB_ID, f"ðŸ“ Data file: {data_file}")
        log_message(JOB_ID, f"ðŸŽ¯ Target column: {target_column}")
        log_message(JOB_ID, f"ðŸ“‹ Task type: {task_type}")
        log_message(JOB_ID, f"ðŸ“‹ Source type: {source_type}")
        
        # Initialize the appropriate AutoML class
        log_message(JOB_ID, "ðŸ—ï¸ Initializing AutoML Class...")
        
        if task_type == 'classification':
            automl_class = AutoMLClassifier
        elif task_type == 'regression':
            automl_class = AutoMLRegressor
        elif task_type == 'clustering':
            automl_class = AutoMLClusterer
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Use the automl_pyspark package config file for Dataproc
        config_path = config_source_path if os.path.exists(config_source_path) else JOB_CONFIG.get('config_path', 'config.yaml')
        log_message(JOB_ID, f"ðŸ”§ Using config file: {config_path}")
        
        automl = automl_class(
            output_dir=results_dir,
            config_path=config_path,
            environment=JOB_CONFIG.get('environment', 'production'),
            preset=JOB_CONFIG.get('preset', ''),
            spark_session=spark_session
        )
        
        log_message(JOB_ID, "âœ… AutoML initialized successfully")
        update_progress(JOB_ID, 4, 8, "Initializing data manager...")
        
        # Initialize data manager for OOT data loading (not used for GCS paths - using direct Spark loading)
        try:
            from data_input_manager import DataInputManager
            data_manager = DataInputManager(spark=spark_session, output_dir=results_dir, user_id=JOB_CONFIG.get('user_id', 'automl_user'))
            log_message(JOB_ID, "âœ… Data manager initialized successfully")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ Data manager initialization failed: {e} - OOT data loading will use direct Spark loading")
            data_manager = None
        
        # Prepare fit parameters
        fit_params = JOB_CONFIG.get('model_params', {})
        fit_params.update(JOB_CONFIG.get('data_params', {}))
        fit_params.update(JOB_CONFIG.get('advanced_params', {}))
        
        update_progress(JOB_ID, 5, 8, "Running AutoML pipeline...")
        
        # Run the AutoML pipeline with robust error handling
        log_message(JOB_ID, f"ðŸŽ¯ Starting {task_type} AutoML pipeline...")
        
        try:
            if task_type in ['classification', 'regression']:
                # Load OOT datasets if provided
                oot1_data, oot2_data = load_oot_datasets(JOB_CONFIG, data_manager)
                
                # Debug: Log what OOT data we're passing
                if oot1_data is not None:
                    try:
                        log_message(JOB_ID, f"ðŸ”„ OOT1 data will be passed to AutoML.fit(): {oot1_data.count()} rows")
                    except:
                        log_message(JOB_ID, "ðŸ”„ OOT1 data exists but count failed")
                else:
                    log_message(JOB_ID, "ðŸ”„ OOT1 data is None - will not be passed to AutoML.fit()")
                    
                if oot2_data is not None:
                    try:
                        log_message(JOB_ID, f"ðŸ”„ OOT2 data will be passed to AutoML.fit(): {oot2_data.count()} rows")
                    except:
                        log_message(JOB_ID, "ðŸ”„ OOT2 data exists but count failed")
                else:
                    log_message(JOB_ID, "ðŸ”„ OOT2 data is None - will not be passed to AutoML.fit()")
                
                log_message(JOB_ID, f"ðŸ”„ Calling AutoML.fit() with {len(fit_params)} parameters...")
                automl.fit(
                    train_data=data_file,
                    target_column=target_column,
                    oot1_data=oot1_data,
                    oot2_data=oot2_data,
                    **fit_params
                )
            elif task_type == 'clustering':
                log_message(JOB_ID, f"ðŸ”„ Calling AutoML.fit() for clustering with {len(fit_params)} parameters...")
                automl.fit(
                    train_data=data_file,
                    **fit_params
                )
            
            update_progress(JOB_ID, 6, 8, "AutoML training completed")
            log_message(JOB_ID, "âœ… AutoML.fit() completed successfully")
            
        except Exception as e:
            log_message(JOB_ID, f"âŒ AutoML pipeline failed: {str(e)}")
            import traceback
            log_message(JOB_ID, f"ðŸ” Full traceback: {traceback.format_exc()}")
            raise
        
        update_progress(JOB_ID, 7, 8, "Saving results to GCS...")
        log_message(JOB_ID, "ðŸŽ‰ AutoML pipeline completed successfully!")
        
        # Copy results from local temp directory to GCS and mounted volume
        try:
            import subprocess
            log_message(JOB_ID, f"ðŸ“¤ Copying results from {results_dir} to GCS...")
            
            # First, try to copy to mounted volume if it exists
            mounted_results_dir = f"/app/automl_pyspark/automl_results/{JOB_ID}"
            if os.path.exists("/app/automl_pyspark/automl_results"):
                try:
                    import shutil
                    if os.path.exists(results_dir):
                        os.makedirs(mounted_results_dir, exist_ok=True)
                        # Copy all files from temp results to mounted volume
                        for item in os.listdir(results_dir):
                            src = os.path.join(results_dir, item)
                            dst = os.path.join(mounted_results_dir, item)
                            if os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src, dst)
                        log_message(JOB_ID, f"âœ… Results copied to mounted volume: {mounted_results_dir}")
                except Exception as e:
                    log_message(JOB_ID, f"âš ï¸ Could not copy to mounted volume: {e}")
            
            # Then copy to GCS bucket
            copy_cmd = [
                "gsutil", "-m", "cp", "-r", 
                f"{results_dir}/*", 
                f"gs://{GCS_RESULTS_BUCKET}/automl_results/{JOB_ID}/"
            ]
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                log_message(JOB_ID, f"âœ… Results successfully copied to gs://{GCS_RESULTS_BUCKET}/automl_results/{JOB_ID}/")
            else:
                log_message(JOB_ID, f"âš ï¸ Results copy warning: {result.stderr}")
                
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ Warning: Failed to copy results to GCS: {e}")
        
        # Create job status and log files in GCS
        try:
            import subprocess
            
            # Create local temp status files
            temp_jobs_dir = "/tmp/automl_jobs"
            os.makedirs(temp_jobs_dir, exist_ok=True)
            temp_job_dir = os.path.join(temp_jobs_dir, JOB_ID)
            os.makedirs(temp_job_dir, exist_ok=True)
            
            # Write job status
            status_file = os.path.join(temp_job_dir, f"{JOB_ID}_status.txt")
            with open(status_file, 'w') as f:
                f.write("Completed")
            
            # Create logs directory for comprehensive logging
            temp_log_dir = os.path.join(temp_job_dir, "logs")
            os.makedirs(temp_log_dir, exist_ok=True)
            
            # CRITICAL FIX: Create comprehensive job execution logs
            log_message(JOB_ID, "ðŸ“‹ Creating comprehensive execution logs for Streamlit...")
            
            # Write comprehensive job execution log
            comprehensive_log_file = os.path.join(temp_log_dir, "job_execution.log")
            compat_log_file = os.path.join(temp_job_dir, f"{JOB_ID}_log.txt")
            
            # Capture comprehensive log content
            log_content_parts = []
            log_content_parts.append(f"=== DATAPROC SERVERLESS JOB EXECUTION LOG ===")
            log_content_parts.append(f"Job ID: {JOB_ID}")
            log_content_parts.append(f"Batch ID: {BATCH_ID}")
            log_content_parts.append(f"Completed at: {datetime.now().isoformat()}")
            log_content_parts.append(f"Results saved to: gs://{GCS_RESULTS_BUCKET}/automl_results/{JOB_ID}/")
            log_content_parts.append(f"Status: COMPLETED")
            log_content_parts.append("")
            
            # Add execution environment info
            log_content_parts.append(f"=== EXECUTION ENVIRONMENT ===")
            log_content_parts.append(f"Python version: " + str(sys.version))
            log_content_parts.append(f"Working directory: " + str(os.getcwd()))
            log_content_parts.append(f"Container hostname: " + str(os.uname().nodename if hasattr(os, 'uname') else 'unknown'))
            log_content_parts.append("")
            
            # Add key environment variables
            log_content_parts.append(f"=== KEY ENVIRONMENT VARIABLES ===")
            env_keys_to_capture = ['SPARK_HOME', 'JAVA_HOME', 'HADOOP_HOME', 'DATAPROC_VERSION', 
                                 'GCS_RESULTS_BUCKET', 'PYSPARK_PYTHON', 'SPARK_CONF_DIR']
            for key in env_keys_to_capture:
                if key in os.environ:
                    log_content_parts.append(f"{key}={os.environ[key]}")
            log_content_parts.append("")
            
            # Add job configuration summary
            log_content_parts.append(f"=== JOB CONFIGURATION ===")
            log_content_parts.append(f"Task Type: {JOB_CONFIG.get('task_type', 'unknown')}")
            log_content_parts.append(f"Target Column: {JOB_CONFIG.get('target_column', 'unknown')}")
            log_content_parts.append(f"Data Source: {JOB_CONFIG.get('data_source', 'unknown')}")
            log_content_parts.append("")
            
            # Try to capture recent application logs
            log_content_parts.append(f"=== RECENT APPLICATION OUTPUT ===")
            log_content_parts.append(f"Note: Full execution logs are captured in real-time during job execution.")
            log_content_parts.append(f"This summary shows job completion status and environment details.")
            log_content_parts.append(f"For detailed execution logs, check Dataproc console:")
            log_content_parts.append(f"https://console.cloud.google.com/dataproc/batches")
            log_content_parts.append("")
            
            # Combine all content
            comprehensive_log_content = "\n".join(log_content_parts)
            
            # Write comprehensive log file
            with open(comprehensive_log_file, 'w', encoding='utf-8') as f:
                f.write(comprehensive_log_content)
            
            # Write compatibility log file (for backward compatibility)
            with open(compat_log_file, 'w', encoding='utf-8') as f:
                f.write(comprehensive_log_content)
            
            # Write job completion marker (separate file)
            completion_file = os.path.join(temp_job_dir, "job_completed.txt")
            with open(completion_file, 'w') as f:
                f.write(f"Job completed at: {datetime.now().isoformat()}\n")
                f.write(f"Results saved to: gs://{GCS_RESULTS_BUCKET}/automl_results/{JOB_ID}/\n")
                f.write(f"Batch ID: {BATCH_ID}\n")
            
            # Copy all files to GCS with proper structure for Streamlit
            status_copy_cmd = ["gsutil", "cp", status_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/{JOB_ID}_status.txt"]
            comprehensive_log_copy_cmd = ["gsutil", "cp", comprehensive_log_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/logs/job_execution.log"]
            compat_log_copy_cmd = ["gsutil", "cp", compat_log_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/{JOB_ID}_log.txt"]
            completion_copy_cmd = ["gsutil", "cp", completion_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/job_completed.txt"]
            
            subprocess.run(status_copy_cmd, check=True)
            subprocess.run(comprehensive_log_copy_cmd, check=True)
            subprocess.run(compat_log_copy_cmd, check=True)
            subprocess.run(completion_copy_cmd, check=True)
            
            log_message(JOB_ID, f"âœ… Comprehensive logs created and uploaded to GCS")
            log_message(JOB_ID, f"ðŸ“‹ Streamlit logs available at: gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/logs/job_execution.log")
            
            log_message(JOB_ID, f"âœ… Job status files uploaded to GCS")
        except Exception as e:
            log_message(JOB_ID, f"âš ï¸ Warning: Failed to create job status files: {e}")
        
        update_progress(JOB_ID, 8, 8, "Job completed successfully")
        log_message(JOB_ID, "ðŸŽ‰ Dataproc Serverless job completed successfully!")
        
        # ðŸš¨ ACTIVATE EMERGENCY SHUTDOWN TIMER (10 seconds)
        add_post_completion_force_shutdown(timeout_seconds=10)
        
        # âœ… IMMEDIATE TERMINATION AFTER SUCCESSFUL COMPLETION
        immediate_termination_after_job_completion(spark_session=spark_session, exit_code=0)
        
    except Exception as e:
        log_message(JOB_ID, f"âŒ Job failed with error: {str(e)}")
        import traceback
        log_message(JOB_ID, f"ðŸ” Full traceback: {traceback.format_exc()}")
        
        # Create failure status files in GCS
        try:
            import subprocess
            
            # Create local temp error files
            temp_jobs_dir = "/tmp/automl_jobs"
            os.makedirs(temp_jobs_dir, exist_ok=True)
            temp_job_dir = os.path.join(temp_jobs_dir, JOB_ID)
            os.makedirs(temp_job_dir, exist_ok=True)
            
            # Create logs directory
            temp_log_dir = os.path.join(temp_job_dir, "logs")
            os.makedirs(temp_log_dir, exist_ok=True)
            
            # Write job status
            status_file = os.path.join(temp_job_dir, f"{JOB_ID}_status.txt")
            with open(status_file, 'w') as f:
                f.write("FAILED")
            
            # Write error details
            error_file = os.path.join(temp_job_dir, f"{JOB_ID}_error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Job failed at: {datetime.now().isoformat()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
            # Create comprehensive log files
            log_file = os.path.join(temp_log_dir, "job_execution.log")
            compat_log_file = os.path.join(temp_job_dir, f"{JOB_ID}_log.txt")
            
            log_content = f"""[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] âŒ Job failed with error: {str(e)}
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ðŸ” Full traceback: {traceback.format_exc()}
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ðŸ“Š Job ID: {JOB_ID}
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ðŸ—ï¸ Batch ID: {BATCH_ID}
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] âš ï¸ This job failed during execution - check error details above
"""
            
            with open(log_file, 'w') as f:
                f.write(log_content)
            with open(compat_log_file, 'w') as f:
                f.write(log_content)
            
            # Copy all files to GCS
            status_copy_cmd = ["gsutil", "cp", status_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/{JOB_ID}_status.txt"]
            error_copy_cmd = ["gsutil", "cp", error_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/{JOB_ID}_error.txt"]
            log_copy_cmd = ["gsutil", "cp", log_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/logs/job_execution.log"]
            compat_log_copy_cmd = ["gsutil", "cp", compat_log_file, f"gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/{JOB_ID}_log.txt"]
            
            subprocess.run(status_copy_cmd, check=False)
            subprocess.run(error_copy_cmd, check=False)
            subprocess.run(log_copy_cmd, check=False)
            subprocess.run(compat_log_copy_cmd, check=False)
            
            log_message(JOB_ID, f"âœ… Error files and logs uploaded to GCS: gs://{GCS_RESULTS_BUCKET}/automl_jobs/{JOB_ID}/")
        except Exception as status_error:
            log_message(JOB_ID, f"âš ï¸ Warning: Failed to create error status files: {status_error}")
        
        # ensure background threads won't block and force exit with non-zero code
        _make_background_threads_daemon()
        sys.stdout.flush()
        sys.stderr.flush()
        immediate_termination_after_job_completion(spark_session=spark_session, exit_code=1)
    
    finally:
        # final best-effort cleanup (this won't run if os._exit fired earlier)
        try:
            if spark_session is not None:
                log_message(JOB_ID, "ðŸ§¹ Final cleanup: closing spark_session if still open")
                spark_session.stop()
                log_message(JOB_ID, "âœ… Spark session closed in finally")
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())