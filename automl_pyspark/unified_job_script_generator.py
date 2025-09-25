#!/usr/bin/env python3
"""
Unified Job Script Generator for both BackgroundJobManager and DataprocServerlessManager
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class UnifiedJobScriptGenerator:
    """Generates clean Python job scripts for both local and GCP execution."""
    
    def __init__(self):
        self.template_cache = {}
    
    def generate_job_script(
        self,
        job_id: str,
        config: Dict[str, Any],
        execution_mode: str = "local",  # "local" or "dataproc"
        job_files: Optional[Dict[str, str]] = None,
        batch_id: Optional[str] = None,
        jobs_dir: Optional[str] = None
    ) -> str:
        """
        Generate a clean Python job script.
        
        Args:
            job_id: Unique job identifier
            config: Job configuration dictionary
            execution_mode: "local" for BackgroundJobManager, "dataproc" for DataprocServerlessManager
            job_files: Additional files for Dataproc execution
            batch_id: Batch ID for Dataproc execution
        """
        
        # Prepare configuration - convert to Python-compatible format
        config_python = self._convert_to_python_format(config)
        generation_time = datetime.now().isoformat()
        
        # Generate script based on execution mode
        if execution_mode == "local":
            return self._generate_local_script(job_id, config, config_python, generation_time, jobs_dir)
        elif execution_mode == "dataproc":
            return self._generate_dataproc_script(job_id, config, config_python, generation_time, job_files, batch_id)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
    
    def _generate_local_script(self, job_id: str, config: Dict, config_python: str, generation_time: str, jobs_dir: Optional[str] = None) -> str:
        """Generate script for local execution (BackgroundJobManager)."""
        
        # Detect execution environment and set appropriate paths
        is_docker = os.path.exists('/app') and os.path.isdir('/app')
        is_dataproc = os.environ.get('DATAPROC_BATCH_ID') is not None or os.environ.get('GOOGLE_CLOUD_PROJECT') is not None
        
        if is_docker:
            python_path = '/app/automl_pyspark'
            if jobs_dir is None:
                jobs_dir = "/app/automl_pyspark/automl_jobs"
        elif is_dataproc:
            python_path = os.getcwd()
            if jobs_dir is None:
                jobs_dir = os.path.join(os.getcwd(), 'automl_jobs')
        else:
            python_path = os.path.join(os.path.dirname(__file__))
            if jobs_dir is None:
                jobs_dir = os.path.join(os.path.dirname(__file__), 'automl_jobs')
        
        script = f'''#!/usr/bin/env python3
"""
AutoML Job Script for {job_id}
Generated at: {generation_time}
Execution Mode: Local Background Threads
"""

import sys
import os
import json
import pandas as pd
import signal
import atexit
from datetime import datetime

# Add the automl_pyspark directory to Python path
sys.path.insert(0, '{python_path}')
# Also add the parent directory to ensure automl_pyspark package is importable
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname('{python_path}'))

# Global variable to track Spark session
spark_session = None

# --- Clean Termination Utilities (based on template) -------------------

def log_message(job_id, message):
    """Log a message with timestamp for local execution."""
    timestamp = datetime.now().isoformat()
    print(f"[{{timestamp}}] {{message}}", flush=True)

def is_dataproc():
    """Detect if running inside Dataproc Serverless."""
    return False  # Local execution mode

def _make_background_threads_daemon():
    """Prevent non-daemon background threads from blocking shutdown."""
    import threading
    for t in threading.enumerate():
        if t is not threading.current_thread() and not t.daemon:
            t.daemon = True

def terminate_job(spark_session=None, exit_code=0):
    """Terminate job differently for Dataproc vs local runs."""
    try:
        if spark_session is not None:
            log_message(JOB_ID, "🧹 Stopping Spark session...")
            spark_session.stop()
    except Exception as e:
        log_message(JOB_ID, f"⚠️ Error stopping Spark: {{e}}")

    _make_background_threads_daemon()

    if is_dataproc():
        # 🚨 On Dataproc: kill process immediately (skip cleanup)
        log_message(JOB_ID, "📕 Dataproc mode: forcing container shutdown with os._exit")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    else:
        # 🏡 Local mode: exit gracefully
        log_message(JOB_ID, "📕 Local mode: exiting gracefully with sys.exit")
        sys.exit(exit_code)

# Set up signal handler for clean termination (using template approach)
signal.signal(signal.SIGTERM, lambda s, f: terminate_job(spark_session, exit_code=1))

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import col, when, isnan, count, isnull

# Job configuration
JOB_ID = "{job_id}"
JOB_CONFIG = {config_python}
JOBS_DIR = "{jobs_dir}"
EXECUTION_MODE = "local"  # Local execution mode
JOB_CONFIG_FILE = os.path.join(JOBS_DIR, JOB_ID, f"{{JOB_ID}}.json")
JOB_STATUS_FILE = os.path.join(JOBS_DIR, JOB_ID, f"{{JOB_ID}}_status.txt")
JOB_ERROR_FILE = os.path.join(JOBS_DIR, JOB_ID, f"{{JOB_ID}}_error.log")

def log_message(job_id, message):
    """Log a message for the job."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{{timestamp}}] {{message}}\\n"
    
    # Write to structured log file
    log_dir = os.path.join(JOBS_DIR, job_id, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'job_execution.log')
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    # Write to compatibility log file
    compat_log_file = os.path.join(JOBS_DIR, job_id, f"{{job_id}}_log.txt")
    with open(compat_log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(f"[{{timestamp}}] {{message}}")

def update_progress(job_id, current_step, total_steps, current_task):
    """Update progress for the job."""
    progress_data = {{
        'current_step': current_step,
        'total_steps': total_steps,
        'current_task': current_task,
        'progress_percentage': round((current_step / total_steps) * 100, 1),
        'timestamp': datetime.now().isoformat()
    }}
    
    progress_file = os.path.join(JOBS_DIR, job_id, f"{{job_id}}_progress.json")
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

{self._get_load_oot_datasets_function()}

def main():
    """Main job execution."""
    try:
        # Set up proper signal handling for subprocess
        import signal
        import sys
        
        def signal_handler(signum, frame):
            log_message(JOB_ID, f"🛑 Received signal {{signum}}, shutting down gracefully...")
            if spark_session:
                try:
                    spark_session.stop()
                except:
                    pass
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        log_message(JOB_ID, f"🚀 Starting AutoML job: {{JOB_ID}}")
        update_progress(JOB_ID, 1, 8, "Initializing...")
        
        # Initialize Spark with robust configuration
        global spark_session
        spark_session = SparkSession.builder \\
            .appName(f"AutoML_{{JOB_ID}}") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \\
            .config("spark.driver.memory", "2g") \\
            .config("spark.driver.maxResultSize", "1g") \\
            .config("spark.executor.memory", "2g") \\
            .config("spark.executor.cores", "2") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \\
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .config("spark.sql.execution.pythonUDF.arrow.enabled", "false") \\
            .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5") \\
            .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \\
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \\
            .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB") \\
            .config("spark.sql.adaptive.coalescePartitions.initialPartitionNum", "200") \\
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \\
            .config("spark.eventLog.enabled", "false") \\
            .config("spark.eventLog.compress", "false") \\
            .config("spark.ui.showConsoleProgress", "false") \\
            .getOrCreate()
        
        spark = spark_session
        
        # Configure logging to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")
        log_message(JOB_ID, "✅ Spark session initialized")
        log_message(JOB_ID, f"📊 Spark UI available at: {{spark.sparkContext.uiWebUrl}}")
        log_message(JOB_ID, f"🔧 Spark configuration: driver.memory={{spark.conf.get('spark.driver.memory')}}, executor.memory={{spark.conf.get('spark.executor.memory')}}")
        
        update_progress(JOB_ID, 2, 8, "Loading data...")
        
        # Get job configuration
        data_file = JOB_CONFIG['data_file']
        target_column = JOB_CONFIG['target_column']
        task_type = JOB_CONFIG['task_type']
        
        log_message(JOB_ID, f"📁 Data file: {{data_file}}")
        log_message(JOB_ID, f"🎯 Target column: {{target_column}}")
        log_message(JOB_ID, f"📋 Task type: {{task_type}}")
        
        update_progress(JOB_ID, 3, 8, "Preparing AutoML pipeline...")
        
        # Import appropriate AutoML class based on task type
        log_message(JOB_ID, "📦 Importing AutoML classes...")
        if task_type == 'classification':
            from classification.automl_classifier import AutoMLClassifier
            automl_class = AutoMLClassifier
        elif task_type == 'regression':
            from regression.automl_regressor import AutoMLRegressor
            automl_class = AutoMLRegressor
        elif task_type == 'clustering':
            from clustering.automl_clusterer import AutoMLClusterer
            automl_class = AutoMLClusterer
        else:
            raise ValueError(f'Unsupported task type: {{task_type}}')
        
        # Initialize AutoML with proper configuration
        update_progress(JOB_ID, 4, 8, "Initializing AutoML Class...")
        log_message(JOB_ID, "🏗️ Initializing AutoML Class...")
        
        # Set results directory based on execution environment
        is_docker_env = os.path.exists('/app') and os.path.isdir('/app')
        is_dataproc_env = os.environ.get('DATAPROC_BATCH_ID') is not None or os.environ.get('GOOGLE_CLOUD_PROJECT') is not None
        
        if is_docker_env:
            results_dir = f"/app/automl_pyspark/automl_results/{{JOB_ID}}"
        elif is_dataproc_env:
            results_dir = f"/tmp/automl_results/{{JOB_ID}}"
        else:
            results_dir = os.getcwd() + f"/automl_results/{{JOB_ID}}"
        
        os.makedirs(results_dir, exist_ok=True)
        log_message(JOB_ID, f"📁 Results will be saved to: {{results_dir}}")
        
        automl = automl_class(
            output_dir=results_dir,
            config_path=JOB_CONFIG.get('config_path', 'config.yaml'),
            environment=JOB_CONFIG.get('environment', 'production'),
            preset=JOB_CONFIG.get('preset', ''),
            spark_session=spark
        )
        
        log_message(JOB_ID, "✅ AutoML initialized successfully")
        update_progress(JOB_ID, 5, 8, "Running AutoML pipeline...")
        
        # Initialize data manager for OOT data loading
        try:
            from data_input_manager import DataInputManager
            data_manager = DataInputManager(spark=spark_session, output_dir=results_dir, user_id=JOB_CONFIG.get('user_id', 'automl_user'))
            log_message(JOB_ID, "✅ Data manager initialized successfully")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Data manager initialization failed: {{e}} - OOT data loading will be skipped")
            data_manager = None
        
        # Prepare fit parameters
        fit_params = JOB_CONFIG.get('model_params', {{}})
        fit_params.update(JOB_CONFIG.get('data_params', {{}}))
        fit_params.update(JOB_CONFIG.get('advanced_params', {{}}))
        
        # Run the AutoML pipeline with robust error handling
        log_message(JOB_ID, f"🎯 Starting {{task_type}} AutoML pipeline...")
        
        try:
            if task_type in ['classification', 'regression']:
                # Load OOT datasets if provided
                oot1_data, oot2_data = load_oot_datasets(JOB_CONFIG, data_manager)
                
                log_message(JOB_ID, f"🔄 Calling AutoML.fit() with {{len(fit_params)}} parameters...")
                automl.fit(
                    train_data=data_file,
                    target_column=target_column,
                    oot1_data=oot1_data,
                    oot2_data=oot2_data,
                    **fit_params
                )
            else:  # clustering
                log_message(JOB_ID, f"🔄 Calling AutoML.fit() for clustering...")
                automl.fit(
                    train_data=data_file,
                    **fit_params
                )
            
            log_message(JOB_ID, "✅ AutoML.fit() completed successfully")
            
        except Exception as fit_error:
            log_message(JOB_ID, f"❌ AutoML.fit() failed: {{str(fit_error)}}")
            log_message(JOB_ID, f"📋 Error type: {{type(fit_error).__name__}}")
            raise fit_error
        
        log_message(JOB_ID, "🎉 AutoML pipeline completed successfully!")
        update_progress(JOB_ID, 6, 8, "Saving results...")
        
        # Update job status
        with open(JOB_STATUS_FILE, 'w') as f:
            f.write("Completed")
        
        log_message(JOB_ID, "🎉 Job completed successfully!")
        update_progress(JOB_ID, 8, 8, "Completed")
        
        # ✅ LOCAL EXECUTION - JOB COMPLETED (no immediate termination needed)
        log_message(JOB_ID, "✅ Local execution completed - container will remain active")
        
    except Exception as e:
        error_msg = str(e)
        log_message(JOB_ID, f"❌ Job failed: {{error_msg}}")
        
        # Write error to error file
        with open(JOB_ERROR_FILE, 'w') as f:
            f.write(f"Job failed: {{error_msg}}\\n")
            f.write(f"Timestamp: {{datetime.now().isoformat()}}\\n")
        
        # Update job status
        with open(JOB_STATUS_FILE, 'w') as f:
            f.write("Failed")
        
        raise
    
    finally:
        # Ensure Spark session is properly closed
        try:
            if spark_session:
                log_message(JOB_ID, "🔄 Closing Spark session...")
                spark_session.stop()
                log_message(JOB_ID, "✅ Spark session closed successfully")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Warning: Error closing Spark session: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _generate_dataproc_script(self, job_id: str, config: Dict, config_python: str, generation_time: str, job_files: Dict[str, str], batch_id: str) -> str:
        """Generate script for Dataproc Serverless execution."""
        
        # For Dataproc, use current working directory as the base path
        python_path = os.getcwd()
        # For Dataproc, use GCS paths directly (no volume mounts available)
        results_bucket = config.get('gcs_results_bucket', 'rapid_modeler_app')
        results_dir = f"/tmp/automl_results/{job_id}"  # Local temp for processing
        gcs_results_dir = f"gs://{results_bucket}/automl_results/{job_id}"  # GCS path for final results
        # For data files, use GCS paths directly
        gcs_data_bucket = config.get('gcs_temp_bucket', 'rapid_modeler_app')
        
        # Generate the same comprehensive script as local but with Dataproc-specific configurations
        script = f'''#!/usr/bin/env python3
"""
AutoML Job Script for {job_id}
Generated at: {generation_time}
Execution Mode: Dataproc Serverless
Batch ID: {batch_id}
"""

import sys
import os
import json
import pandas as pd
import signal
import atexit
from datetime import datetime

# Add the automl_pyspark directory to Python path
sys.path.insert(0, '{python_path}')
# Also add the parent directory to ensure automl_pyspark package is importable
sys.path.insert(0, '/tmp')
sys.path.insert(0, os.path.dirname('{python_path}'))

# Global variable to track Spark session
spark_session = None

# Job configuration
JOB_ID = "{job_id}"
BATCH_ID = "{batch_id}"

# GCS bucket configuration for Dataproc
GCS_DATA_BUCKET = "{config.get('gcs_temp_bucket', 'rapid_modeler_app')}"
GCS_RESULTS_BUCKET = "{results_bucket}"
EXECUTION_MODE = "dataproc"  # Flag to indicate Dataproc execution mode

# Set up logging and progress tracking for Dataproc
def log_message(job_id, message):
    """Log a message with timestamp for Dataproc execution."""
    timestamp = datetime.now().isoformat()
    print(f"[{{timestamp}}] {{message}}", flush=True)

def update_progress(job_id, step, total_steps, message):
    """Update job progress for Dataproc execution."""
    progress = round((step / total_steps) * 100, 1)
    log_message(job_id, f"📊 Progress: {{progress}}% - {{message}}")

# --- Clean Termination Utilities (based on template) -------------------

def is_dataproc():
    """Detect if running inside Dataproc Serverless."""
    return EXECUTION_MODE == "dataproc"

def _make_background_threads_daemon():
    """Prevent non-daemon background threads from blocking shutdown."""
    import threading
    for t in threading.enumerate():
        if t is not threading.current_thread() and not t.daemon:
            t.daemon = True

def terminate_job(spark_session=None, exit_code=0):
    """Terminate job differently for Dataproc vs local runs."""
    try:
        if spark_session is not None:
            log_message(JOB_ID, "🧹 Stopping Spark session...")
            spark_session.stop()
    except Exception as e:
        log_message(JOB_ID, f"⚠️ Error stopping Spark: {{e}}")

    _make_background_threads_daemon()

    if is_dataproc():
        # 🚨 On Dataproc: kill process immediately (skip cleanup)
        log_message(JOB_ID, "📕 Dataproc mode: forcing container shutdown with os._exit")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    else:
        # 🏡 Local mode: exit gracefully
        log_message(JOB_ID, "📕 Local mode: exiting gracefully with sys.exit")
        sys.exit(exit_code)

# ========================================================================
# LEGACY CLEANUP FUNCTIONS (kept for compatibility)
# ========================================================================


# Set up signal handler for clean termination (using template approach)
signal.signal(signal.SIGTERM, lambda s, f: terminate_job(spark_session, exit_code=1))

# Job configuration (already defined above)
JOB_CONFIG = {config_python}

{self._get_load_oot_datasets_function()}

def main():
    """Main job execution for Dataproc Serverless."""
    global spark_session
    
    try:
        log_message(JOB_ID, f"🚀 Starting Dataproc Serverless job: {job_id}")
        log_message(JOB_ID, f"📊 Batch ID: {batch_id}")
        log_message(JOB_ID, f"📁 Working directory: " + os.getcwd())
        
        update_progress(JOB_ID, 1, 8, "Initializing Spark session...")
        
        # Initialize Spark for Dataproc with optimized settings
        from pyspark.sql import SparkSession
        
        # Create Spark session with Dataproc-optimized settings
        # Resource allocation is handled by Dataproc Serverless configuration
        spark_session = SparkSession.builder \\
            .appName(f"AutoML_{job_id}") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
            .config("spark.dynamicAllocation.enabled", "false") \\
            .getOrCreate()
        
        # Log the actual Spark configuration used
        log_message(JOB_ID, f"🧠 Spark resource allocation:")
        log_message(JOB_ID, f"   Executors: {{spark_session.conf.get('spark.executor.instances', 'auto')}}")
        log_message(JOB_ID, f"   Executor Memory: {{spark_session.conf.get('spark.executor.memory', 'auto')}}")
        log_message(JOB_ID, f"   Driver Memory: {{spark_session.conf.get('spark.driver.memory', 'auto')}}")
        
        log_message(JOB_ID, "✅ Spark session initialized for Dataproc Serverless")
        log_message(JOB_ID, f"📊 Spark UI available at: {{spark_session.sparkContext.uiWebUrl}}")
        log_message(JOB_ID, f"🔧 Spark configuration: driver.memory={{spark_session.conf.get('spark.driver.memory', 'default')}}, executor.memory={{spark_session.conf.get('spark.executor.memory', 'default')}}")
        
        # Skip package installation - using custom container image with pre-installed packages
        log_message(JOB_ID, "📦 Using custom container image with pre-installed packages...")
        log_message(JOB_ID, "ℹ️ Skipping runtime package installation - all dependencies are pre-installed")
        
        # Ensure config file is available - use automl_pyspark package config
        config_source_path = "/app/automl_pyspark/config.yaml"
        
        # Check if config file exists in the expected location
        if os.path.exists(config_source_path):
            log_message(JOB_ID, f"✅ Config file found at: {{config_source_path}}")
        else:
            log_message(JOB_ID, f"⚠️ Config file not found at {{config_source_path}}, will use defaults")
        
        # Set up results directory - ensure it's in the mounted volume
        results_dir = "{results_dir}"
        
        # For Docker environments, ensure we use the mounted volume path
        if os.path.exists('/app/automl_pyspark/automl_results'):
            # We're in Docker with mounted volumes
            mounted_results_dir = f"/app/automl_pyspark/automl_results/{{JOB_ID}}"
            os.makedirs(mounted_results_dir, exist_ok=True)
            # Create symlink from results_dir to mounted_results_dir if they're different
            if results_dir != mounted_results_dir:
                try:
                    if os.path.exists(results_dir):
                        os.rmdir(results_dir)
                    os.symlink(mounted_results_dir, results_dir)
                    log_message(JOB_ID, f"🔗 Created symlink: {{results_dir}} -> {{mounted_results_dir}}")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ Could not create symlink, using mounted path directly: {{e}}")
                    results_dir = mounted_results_dir
        else:
            os.makedirs(results_dir, exist_ok=True)
        
        log_message(JOB_ID, f"📁 Results will be saved to: {{results_dir}}")
        
        # Debug: List available files in mounted directories
        try:
            log_message(JOB_ID, "🔍 Debugging mounted directories:")
            
            # Check results directory
            if os.path.exists("/app/automl_pyspark/automl_results"):
                results_contents = os.listdir("/app/automl_pyspark/automl_results")
                log_message(JOB_ID, f"📂 /app/automl_pyspark/automl_results contains: {{results_contents}}")
            
            # Check jobs directory  
            if os.path.exists("/app/automl_pyspark/automl_jobs"):
                jobs_contents = os.listdir("/app/automl_pyspark/automl_jobs")
                log_message(JOB_ID, f"📂 /app/automl_pyspark/automl_jobs contains: {{jobs_contents}}")
            
            # Check specific job directory if it exists
            job_specific_dir = f"/app/automl_pyspark/automl_results/{{JOB_ID}}"
            if os.path.exists(job_specific_dir):
                job_contents = os.listdir(job_specific_dir)
                log_message(JOB_ID, f"📂 {{job_specific_dir}} contains: {{job_contents}}")
            else:
                log_message(JOB_ID, f"📂 {{job_specific_dir}} does not exist yet")
            
            # Check the actual results_dir being used
            if os.path.exists(results_dir):
                actual_contents = os.listdir(results_dir)
                log_message(JOB_ID, f"📂 Active results_dir {{results_dir}} contains: {{actual_contents}}")
            else:
                log_message(JOB_ID, f"📂 Active results_dir {{results_dir}} does not exist yet")
                
        except Exception as debug_e:
            log_message(JOB_ID, f"⚠️ Debug listing failed: {{debug_e}}")
        
        update_progress(JOB_ID, 2, 8, "Importing AutoML classes...")
        
        # Import AutoML classes with error handling
        log_message(JOB_ID, "📦 Importing AutoML classes...")
        try:
            from automl_pyspark.classification.automl_classifier import AutoMLClassifier
            from automl_pyspark.regression.automl_regressor import AutoMLRegressor  
            from automl_pyspark.clustering.automl_clusterer import AutoMLClusterer
            log_message(JOB_ID, "✅ AutoML classes imported successfully")
        except Exception as e:
            log_message(JOB_ID, f"❌ Failed to import AutoML classes: {{e}}")
            raise
        
        # Apply Dataproc compatibility fixes directly
        log_message(JOB_ID, "🔧 Applying Dataproc compatibility fixes...")
        
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
                            log_message(JOB_ID, f"⚠️ Excel export failed: {{e}}")
                            log_message(JOB_ID, f"🔄 Falling back to CSV: {{csv_path}}")
                            result = self.to_csv(csv_path, index=kwargs.get('index', True))
                            log_message(JOB_ID, f"✅ Data saved to CSV instead: {{csv_path}}")
                            return result
                        else:
                            raise
                    else:
                        raise
            
            pd.DataFrame.to_excel = safe_to_excel
            log_message(JOB_ID, "✅ Applied pandas Excel fallback patch")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Could not patch pandas to_excel: {{e}}")
        
        # Override SHAP computation globally
        try:
            class MockShapModule:
                def __getattr__(self, name):
                    def mock_function(*args, **kwargs):
                        log_message(JOB_ID, f"⚠️ SHAP function '{{name}}' called but SHAP library not available")
                        log_message(JOB_ID, f"   💡 Install with: pip install shap>=0.40.0")
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
                log_message(JOB_ID, "✅ SHAP library is available")
            except ImportError:
                import sys
                sys.modules['shap'] = MockShapModule()
                log_message(JOB_ID, "✅ Applied SHAP mock for graceful handling")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Could not setup SHAP handling: {{e}}")
        
        log_message(JOB_ID, "✅ Dataproc compatibility fixes applied")
        
        update_progress(JOB_ID, 3, 8, "Initializing AutoML...")
        
        # Extract job parameters
        data_file = JOB_CONFIG.get('data_file', 'data.csv')
        target_column = JOB_CONFIG.get('target_column')
        task_type = JOB_CONFIG.get('task_type', 'classification')
        
        # For Dataproc, use GCS paths for all data files
        enhanced_data_config = JOB_CONFIG.get('enhanced_data_config', {{}})
        source_type = enhanced_data_config.get('source_type', 'existing')
        
        # Convert data file to GCS path based on source type (only if not already a GCS path)
        if not data_file.startswith('gs://'):
            if source_type == 'upload':
                # For uploaded files, they're stored in the results bucket
                data_file = f"gs://{{GCS_DATA_BUCKET}}/automl_results/{{JOB_ID}}/{{data_file}}"
                log_message(JOB_ID, f"📁 Upload job - converted to GCS path: {{data_file}}")
            elif source_type == 'existing':
                # For existing files, they're in the data directory
                data_file = f"gs://{{GCS_DATA_BUCKET}}/data/{{data_file}}"
                log_message(JOB_ID, f"📁 Existing file job - converted to GCS path: {{data_file}}")
            else:
                log_message(JOB_ID, f"📁 Using original data file path: {{data_file}}")
        else:
            log_message(JOB_ID, f"📁 Using existing GCS path: {{data_file}}")
        
        log_message(JOB_ID, f"📁 Data file: {{data_file}}")
        log_message(JOB_ID, f"🎯 Target column: {{target_column}}")
        log_message(JOB_ID, f"📋 Task type: {{task_type}}")
        log_message(JOB_ID, f"📋 Source type: {{source_type}}")
        
        # Initialize the appropriate AutoML class
        log_message(JOB_ID, "🏗️ Initializing AutoML Class...")
        
        if task_type == 'classification':
            automl_class = AutoMLClassifier
        elif task_type == 'regression':
            automl_class = AutoMLRegressor
        elif task_type == 'clustering':
            automl_class = AutoMLClusterer
        else:
            raise ValueError(f"Unsupported task type: {{task_type}}")
        
        # Use the automl_pyspark package config file for Dataproc
        config_path = config_source_path if os.path.exists(config_source_path) else JOB_CONFIG.get('config_path', 'config.yaml')
        log_message(JOB_ID, f"🔧 Using config file: {{config_path}}")
        
        automl = automl_class(
            output_dir=results_dir,
            config_path=config_path,
            environment=JOB_CONFIG.get('environment', 'production'),
            preset=JOB_CONFIG.get('preset', ''),
            spark_session=spark_session
        )
        
        log_message(JOB_ID, "✅ AutoML initialized successfully")
        update_progress(JOB_ID, 4, 8, "Initializing data manager...")
        
        # Initialize data manager for OOT data loading (not used for GCS paths - using direct Spark loading)
        try:
            from data_input_manager import DataInputManager
            data_manager = DataInputManager(spark=spark_session, output_dir=results_dir, user_id=JOB_CONFIG.get('user_id', 'automl_user'))
            log_message(JOB_ID, "✅ Data manager initialized successfully")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Data manager initialization failed: {{e}} - OOT data loading will use direct Spark loading")
            data_manager = None
        
        # Prepare fit parameters
        fit_params = JOB_CONFIG.get('model_params', {{}})
        fit_params.update(JOB_CONFIG.get('data_params', {{}}))
        fit_params.update(JOB_CONFIG.get('advanced_params', {{}}))
        
        # CRITICAL: Create early status files BEFORE starting AutoML pipeline
        # This ensures logs are available even if the job fails during execution
        try:
            import subprocess
            
            # Create local temp status files early
            temp_jobs_dir = "/tmp/automl_jobs"
            os.makedirs(temp_jobs_dir, exist_ok=True)
            temp_job_dir = os.path.join(temp_jobs_dir, JOB_ID)
            os.makedirs(temp_job_dir, exist_ok=True)
            
            # Create logs directory
            temp_log_dir = os.path.join(temp_job_dir, "logs")
            os.makedirs(temp_log_dir, exist_ok=True)
            
            # Write initial status
            status_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_status.txt")
            with open(status_file, 'w') as f:
                f.write("RUNNING")
            
            # Create initial log file
            log_file = os.path.join(temp_log_dir, "job_execution.log")
            with open(log_file, 'w') as f:
                f.write(f"[{{datetime.now().isoformat()}}] 🚀 Job started: {{JOB_ID}}\\n")
                f.write(f"[{{datetime.now().isoformat()}}] 📊 Batch ID: {{BATCH_ID}}\\n")
                f.write(f"[{{datetime.now().isoformat()}}] 🎯 Starting {{task_type}} AutoML pipeline...\\n")
            
            # Upload initial status to GCS immediately
            status_copy_cmd = ["gsutil", "cp", status_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_status.txt"]
            log_copy_cmd = ["gsutil", "cp", log_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/logs/job_execution.log"]
            
            subprocess.run(status_copy_cmd, check=False, timeout=30)
            subprocess.run(log_copy_cmd, check=False, timeout=30)
            
            log_message(JOB_ID, "✅ Early status files created in GCS")
        except Exception as early_status_error:
            log_message(JOB_ID, f"⚠️ Warning: Could not create early status files: {{early_status_error}}")
        
        update_progress(JOB_ID, 5, 8, "Running AutoML pipeline...")
        
        # Run the AutoML pipeline with robust error handling
        log_message(JOB_ID, f"🎯 Starting {{task_type}} AutoML pipeline...")
        
        try:
            if task_type in ['classification', 'regression']:
                # Load OOT datasets if provided
                oot1_data, oot2_data = load_oot_datasets(JOB_CONFIG, data_manager)
                
                # Debug: Log what OOT data we're passing
                if oot1_data is not None:
                    try:
                        log_message(JOB_ID, f"🔄 OOT1 data will be passed to AutoML.fit(): {{oot1_data.count()}} rows")
                    except:
                        log_message(JOB_ID, "🔄 OOT1 data exists but count failed")
                else:
                    log_message(JOB_ID, "🔄 OOT1 data is None - will not be passed to AutoML.fit()")
                    
                if oot2_data is not None:
                    try:
                        log_message(JOB_ID, f"🔄 OOT2 data will be passed to AutoML.fit(): {{oot2_data.count()}} rows")
                    except:
                        log_message(JOB_ID, "🔄 OOT2 data exists but count failed")
                else:
                    log_message(JOB_ID, "🔄 OOT2 data is None - will not be passed to AutoML.fit()")
                
                log_message(JOB_ID, f"🔄 Calling AutoML.fit() with {{len(fit_params)}} parameters...")
                automl.fit(
                    train_data=data_file,
                    target_column=target_column,
                    oot1_data=oot1_data,
                    oot2_data=oot2_data,
                    **fit_params
                )
            elif task_type == 'clustering':
                log_message(JOB_ID, f"🔄 Calling AutoML.fit() for clustering with {{len(fit_params)}} parameters...")
                automl.fit(
                    train_data=data_file,
                    **fit_params
                )
            
            update_progress(JOB_ID, 6, 8, "AutoML training completed")
            log_message(JOB_ID, "✅ AutoML.fit() completed successfully")
            
        except Exception as e:
            log_message(JOB_ID, f"❌ AutoML pipeline failed: {{str(e)}}")
            import traceback
            log_message(JOB_ID, f"🔍 Full traceback: {{traceback.format_exc()}}")
            raise
        
        update_progress(JOB_ID, 7, 8, "Saving results to GCS...")
        log_message(JOB_ID, "🎉 AutoML pipeline completed successfully!")
        
        # Copy results from local temp directory to GCS and mounted volume
        try:
            import subprocess
            log_message(JOB_ID, f"📤 Copying results from {{results_dir}} to GCS...")
            
            # First, try to copy to mounted volume if it exists
            mounted_results_dir = f"/app/automl_pyspark/automl_results/{{JOB_ID}}"
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
                        log_message(JOB_ID, f"✅ Results copied to mounted volume: {{mounted_results_dir}}")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ Could not copy to mounted volume: {{e}}")
            
            # Then copy to GCS bucket
            copy_cmd = [
                "gsutil", "-m", "cp", "-r", 
                f"{{results_dir}}/*", 
                f"gs://{{GCS_RESULTS_BUCKET}}/automl_results/{{JOB_ID}}/"
            ]
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                log_message(JOB_ID, f"✅ Results successfully copied to gs://{{GCS_RESULTS_BUCKET}}/automl_results/{{JOB_ID}}/")
            else:
                log_message(JOB_ID, f"⚠️ Results copy warning: {{result.stderr}}")
                
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Warning: Failed to copy results to GCS: {{e}}")
        
        # Create job status and log files in GCS
        try:
            import subprocess
            
            # Create local temp status files
            temp_jobs_dir = "/tmp/automl_jobs"
            os.makedirs(temp_jobs_dir, exist_ok=True)
            temp_job_dir = os.path.join(temp_jobs_dir, JOB_ID)
            os.makedirs(temp_job_dir, exist_ok=True)
            
            # Write job status
            status_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_status.txt")
            with open(status_file, 'w') as f:
                f.write("Completed")
            
            # Create logs directory for comprehensive logging
            temp_log_dir = os.path.join(temp_job_dir, "logs")
            os.makedirs(temp_log_dir, exist_ok=True)
            
            # CRITICAL FIX: Create comprehensive job execution logs
            log_message(JOB_ID, "📋 Creating comprehensive execution logs for Streamlit...")
            
            # Write comprehensive job execution log
            comprehensive_log_file = os.path.join(temp_log_dir, "job_execution.log")
            compat_log_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_log.txt")
            
            # Capture comprehensive log content
            log_content_parts = []
            log_content_parts.append(f"=== DATAPROC SERVERLESS JOB EXECUTION LOG ===")
            log_content_parts.append(f"Job ID: {{JOB_ID}}")
            log_content_parts.append(f"Batch ID: {{BATCH_ID}}")
            log_content_parts.append(f"Completed at: {{datetime.now().isoformat()}}")
            log_content_parts.append(f"Results saved to: gs://{{GCS_RESULTS_BUCKET}}/automl_results/{{JOB_ID}}/")
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
                    log_content_parts.append(f"{{key}}={{os.environ[key]}}")
            log_content_parts.append("")
            
            # Add job configuration summary
            log_content_parts.append(f"=== JOB CONFIGURATION ===")
            log_content_parts.append(f"Task Type: {{JOB_CONFIG.get('task_type', 'unknown')}}")
            log_content_parts.append(f"Target Column: {{JOB_CONFIG.get('target_column', 'unknown')}}")
            log_content_parts.append(f"Data Source: {{JOB_CONFIG.get('data_source', 'unknown')}}")
            log_content_parts.append("")
            
            # Try to capture recent application logs
            log_content_parts.append(f"=== RECENT APPLICATION OUTPUT ===")
            log_content_parts.append(f"Note: Full execution logs are captured in real-time during job execution.")
            log_content_parts.append(f"This summary shows job completion status and environment details.")
            log_content_parts.append(f"For detailed execution logs, check Dataproc console:")
            log_content_parts.append(f"https://console.cloud.google.com/dataproc/batches")
            log_content_parts.append("")
            
            # Combine all content
            comprehensive_log_content = "\\n".join(log_content_parts)
            
            # Write comprehensive log file
            with open(comprehensive_log_file, 'w', encoding='utf-8') as f:
                f.write(comprehensive_log_content)
            
            # Write compatibility log file (for backward compatibility)
            with open(compat_log_file, 'w', encoding='utf-8') as f:
                f.write(comprehensive_log_content)
            
            # Write job completion marker (separate file)
            completion_file = os.path.join(temp_job_dir, "job_completed.txt")
            with open(completion_file, 'w') as f:
                f.write(f"Job completed at: {{datetime.now().isoformat()}}\\n")
                f.write(f"Results saved to: gs://{{GCS_RESULTS_BUCKET}}/automl_results/{{JOB_ID}}/\\n")
                f.write(f"Batch ID: {{BATCH_ID}}\\n")
            
            # Copy all files to GCS with proper structure for Streamlit
            status_copy_cmd = ["gsutil", "cp", status_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_status.txt"]
            comprehensive_log_copy_cmd = ["gsutil", "cp", comprehensive_log_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/logs/job_execution.log"]
            compat_log_copy_cmd = ["gsutil", "cp", compat_log_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_log.txt"]
            completion_copy_cmd = ["gsutil", "cp", completion_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/job_completed.txt"]
            
            subprocess.run(status_copy_cmd, check=True)
            subprocess.run(comprehensive_log_copy_cmd, check=True)
            subprocess.run(compat_log_copy_cmd, check=True)
            subprocess.run(completion_copy_cmd, check=True)
            
            log_message(JOB_ID, f"✅ Comprehensive logs created and uploaded to GCS")
            log_message(JOB_ID, f"📋 Streamlit logs available at: gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/logs/job_execution.log")
            
            log_message(JOB_ID, f"✅ Job status files uploaded to GCS")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ Warning: Failed to create job status files: {{e}}")
        
        update_progress(JOB_ID, 8, 8, "Job completed successfully")
        log_message(JOB_ID, "🎉 Dataproc Serverless job completed successfully!")
        
        # 🎯 IMMEDIATE COMPLETION MARKER - Create this ASAP to survive potential cancellation
        try:
            import subprocess
            import tempfile
            
            # Create early completion marker in GCS immediately
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("COMPLETED")
                temp_file = f.name
            
            early_completion_cmd = ["gsutil", "cp", temp_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/early_completion_marker.txt"]
            subprocess.run(early_completion_cmd, check=False, timeout=30)
            os.unlink(temp_file)
            
            # Also create a timestamped completion record
            completion_timestamp = datetime.now().isoformat()
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(f"Job completed at: {{completion_timestamp}}")
                temp_file2 = f.name
            
            timestamp_cmd = ["gsutil", "cp", temp_file2, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/completion_timestamp.txt"]
            subprocess.run(timestamp_cmd, check=False, timeout=30)
            os.unlink(temp_file2)
            
            log_message(JOB_ID, "✅ Early completion markers created in GCS")
        except Exception as marker_error:
            log_message(JOB_ID, f"⚠️ Warning: Could not create early completion markers: {{marker_error}}")
        
        # ✅ CLEAN TERMINATION USING TEMPLATE APPROACH
        terminate_job(spark_session=spark_session, exit_code=0)
        
    except Exception as e:
        log_message(JOB_ID, f"❌ Job failed with error: {{str(e)}}")
        import traceback
        log_message(JOB_ID, f"🔍 Full traceback: {{traceback.format_exc()}}")
        
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
            status_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_status.txt")
            with open(status_file, 'w') as f:
                f.write("FAILED")
            
            # Write error details
            error_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Job failed at: {{datetime.now().isoformat()}}\\n")
                f.write(f"Error: {{str(e)}}\\n")
                f.write(f"Traceback:\\n{{traceback.format_exc()}}\\n")
            
            # Create comprehensive log files
            log_file = os.path.join(temp_log_dir, "job_execution.log")
            compat_log_file = os.path.join(temp_job_dir, f"{{JOB_ID}}_log.txt")
            
            log_content = f"""[{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}] ❌ Job failed with error: {{str(e)}}
[{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}] 🔍 Full traceback: {{traceback.format_exc()}}
[{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}] 📊 Job ID: {{JOB_ID}}
[{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}] 🏗️ Batch ID: {{BATCH_ID}}
[{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}] ⚠️ This job failed during execution - check error details above
"""
            
            with open(log_file, 'w') as f:
                f.write(log_content)
            with open(compat_log_file, 'w') as f:
                f.write(log_content)
            
            # Copy all files to GCS
            status_copy_cmd = ["gsutil", "cp", status_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_status.txt"]
            error_copy_cmd = ["gsutil", "cp", error_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_error.txt"]
            log_copy_cmd = ["gsutil", "cp", log_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/logs/job_execution.log"]
            compat_log_copy_cmd = ["gsutil", "cp", compat_log_file, f"gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/{{JOB_ID}}_log.txt"]
            
            subprocess.run(status_copy_cmd, check=False)
            subprocess.run(error_copy_cmd, check=False)
            subprocess.run(log_copy_cmd, check=False)
            subprocess.run(compat_log_copy_cmd, check=False)
            
            log_message(JOB_ID, f"✅ Error files and logs uploaded to GCS: gs://{{GCS_RESULTS_BUCKET}}/automl_jobs/{{JOB_ID}}/")
        except Exception as status_error:
            log_message(JOB_ID, f"⚠️ Warning: Failed to create error status files: {{status_error}}")
        
        # ❌ CLEAN TERMINATION ON ERROR USING TEMPLATE APPROACH
        terminate_job(spark_session=spark_session, exit_code=1)
    
    finally:
        # final best-effort cleanup (this won't run if os._exit fired earlier)
        try:
            if spark_session is not None:
                log_message(JOB_ID, "🧹 Final cleanup: closing spark_session if still open")
                spark_session.stop()
                log_message(JOB_ID, "✅ Spark session closed in finally")
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
'''
        
        return script
    
    def _get_load_oot_datasets_function(self):
        """Get the load_oot_datasets function as a string for inclusion in generated scripts."""
        return '''def load_oot_datasets(config, data_manager=None):
    """Helper function to load OOT datasets based on configuration."""
    oot1_data = None
    oot2_data = None
    oot1_file = None
    oot2_file = None
    
    # Detect execution mode - if EXECUTION_MODE is 'local', skip GCS path conversion
    is_local_execution = globals().get('EXECUTION_MODE') == 'local'
    
    # Handle OOT1 - check for BigQuery table first, then file
    if config.get('oot1_bigquery_table'):
        log_message(JOB_ID, f"📅 Loading OOT1 data from BigQuery: {config['oot1_bigquery_table']}")
        try:
            # Use data_manager.load_data for BigQuery tables (more robust)
            oot1_data, _ = data_manager.load_data(
                config['oot1_bigquery_table'],
                source_type='bigquery',
                feature_engineering_phase=False,  # No feature engineering for OOT data
                enable_intelligent_sampling=False  # Load full OOT data
            )
            log_message(JOB_ID, f"✅ OOT1 BigQuery data loaded: {oot1_data.count()} rows")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ OOT1 BigQuery loading failed: {str(e)} - will skip OOT1")
            oot1_data = None
    
    elif config.get('oot1_file') or (config.get('enhanced_data_config') and config['enhanced_data_config'].get('oot1_file')):
        # Check both root level and enhanced_data_config
        oot1_file = config.get('oot1_file') or config['enhanced_data_config'].get('oot1_file')
        log_message(JOB_ID, f"📅 Loading OOT1 data from file: {oot1_file}")
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
                log_message(JOB_ID, f"🏠 Local execution - loading OOT1 via data_manager: {oot1_file}")
                try:
                    oot1_data, _ = data_manager.load_data(oot1_file, source_type=oot1_source_type)
                    log_message(JOB_ID, f"✅ OOT1 file data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT1 file loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
            else:
                # Dataproc execution - convert to GCS paths and use Spark directly
                if not oot1_file.startswith('gs://'):
                    if oot1_source_type == 'upload':
                        oot1_file = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{oot1_file}"
                        log_message(JOB_ID, f"📁 Upload OOT1 - converted to GCS path: {oot1_file}")
                    elif oot1_source_type == 'existing':
                        oot1_file = f"gs://{GCS_DATA_BUCKET}/data/{oot1_file}"
                        log_message(JOB_ID, f"📁 Existing OOT1 - converted to GCS path: {oot1_file}")
                else:
                    log_message(JOB_ID, f"📁 Using existing GCS path for OOT1: {oot1_file}")
                
                # For Dataproc, use direct Spark loading for GCS paths
                log_message(JOB_ID, f"🔄 Using direct Spark loading for OOT1...")
                oot1_data = None
                try:
                    from pyspark.sql import SparkSession
                    spark = SparkSession.getActiveSession()
                    if spark and oot1_file.endswith('.csv'):
                        oot1_data = spark.read.csv(oot1_file, header=True, inferSchema=True)
                        log_message(JOB_ID, f"✅ OOT1 loaded via direct Spark")
                    else:
                        log_message(JOB_ID, f"⚠️ Cannot load OOT1 data: unsupported format or no Spark session")
                        oot1_data = None
                except Exception as spark_error:
                    log_message(JOB_ID, f"⚠️ Direct Spark loading failed: {{str(spark_error)}}")
                    oot1_data = None
            
            if oot1_data:
                try:
                    row_count = oot1_data.count()
                    log_message(JOB_ID, f"✅ OOT1 file data loaded: {row_count} rows")
                    log_message(JOB_ID, f"✅ OOT1 columns: {oot1_data.columns}")
                except Exception as count_error:
                    log_message(JOB_ID, f"⚠️ OOT1 data loaded but count failed: {str(count_error)}")
            else:
                log_message(JOB_ID, "⚠️ OOT1 data is None after loading attempt")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ OOT1 file loading failed: {str(e)} - will skip OOT1")
            import traceback
            log_message(JOB_ID, f"⚠️ OOT1 traceback: {traceback.format_exc()}")
    
    elif config.get('oot1_config'):
        oot1_config = config['oot1_config']
        if oot1_config.get('data_source'):
            source_type = oot1_config.get('source_type', 'existing')
            data_source = oot1_config['data_source']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"🏠 Local execution - loading OOT1 config via data_manager: {data_source}")
                try:
                    oot1_data, _ = data_manager.load_data(
                        data_source, 
                        source_type=source_type,
                        **oot1_config.get('options', {})
                    )
                    log_message(JOB_ID, f"✅ OOT1 data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT1 loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
            else:
                # Dataproc execution - convert to GCS paths
                if not data_source.startswith('gs://'):
                    if source_type == 'upload':
                        data_source = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{data_source}"
                        log_message(JOB_ID, f"📁 Upload OOT1 config - converted to GCS path: {data_source}")
                    elif source_type == 'existing':
                        data_source = f"gs://{GCS_DATA_BUCKET}/data/{data_source}"
                        log_message(JOB_ID, f"📁 Existing OOT1 config - converted to GCS path: {data_source}")
                else:
                    log_message(JOB_ID, f"📁 Using existing GCS path for OOT1 config: {data_source}")
                
                log_message(JOB_ID, f"📅 Loading OOT1 data from {source_type}: {data_source}")
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
                            log_message(JOB_ID, f"⚠️ Cannot load OOT1 config data: unsupported format or no Spark session")
                            oot1_data = None
                    log_message(JOB_ID, f"✅ OOT1 data loaded: {oot1_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT1 loading failed: {str(e)} - will skip OOT1")
                    oot1_data = None
    
    # Handle OOT2 - similar logic
    if config.get('oot2_bigquery_table'):
        log_message(JOB_ID, f"📅 Loading OOT2 data from BigQuery: {config['oot2_bigquery_table']}")
        try:
            # Use data_manager.load_data for BigQuery tables (more robust)
            oot2_data, _ = data_manager.load_data(
                config['oot2_bigquery_table'],
                source_type='bigquery',
                feature_engineering_phase=False,  # No feature engineering for OOT data
                enable_intelligent_sampling=False  # Load full OOT data
            )
            log_message(JOB_ID, f"✅ OOT2 BigQuery data loaded: {oot2_data.count()} rows")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ OOT2 BigQuery loading failed: {str(e)} - will skip OOT2")
            oot2_data = None
    
    elif config.get('oot2_file') or (config.get('enhanced_data_config') and config['enhanced_data_config'].get('oot2_file')):
        # Check both root level and enhanced_data_config
        oot2_file = config.get('oot2_file') or config['enhanced_data_config'].get('oot2_file')
        log_message(JOB_ID, f"📅 Loading OOT2 data from file: {oot2_file}")
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
                log_message(JOB_ID, f"🏠 Local execution - loading OOT2 via data_manager: {oot2_file}")
                try:
                    oot2_data, _ = data_manager.load_data(oot2_file, source_type=oot2_source_type)
                    log_message(JOB_ID, f"✅ OOT2 file data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT2 file loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
            else:
                # Dataproc execution - convert to GCS paths and use Spark directly
                if not oot2_file.startswith('gs://'):
                    if oot2_source_type == 'upload':
                        oot2_file = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{oot2_file}"
                        log_message(JOB_ID, f"📁 Upload OOT2 - converted to GCS path: {oot2_file}")
                    elif oot2_source_type == 'existing':
                        oot2_file = f"gs://{GCS_DATA_BUCKET}/data/{oot2_file}"
                        log_message(JOB_ID, f"📁 Existing OOT2 - converted to GCS path: {oot2_file}")
                else:
                    log_message(JOB_ID, f"📁 Using existing GCS path for OOT2: {oot2_file}")
                
                # For Dataproc, use direct Spark loading for GCS paths
                log_message(JOB_ID, f"🔄 Using direct Spark loading for OOT2...")
                oot2_data = None
                try:
                    from pyspark.sql import SparkSession
                    spark = SparkSession.getActiveSession()
                    if spark and oot2_file.endswith('.csv'):
                        oot2_data = spark.read.csv(oot2_file, header=True, inferSchema=True)
                        log_message(JOB_ID, f"✅ OOT2 loaded via direct Spark")
                    else:
                        log_message(JOB_ID, f"⚠️ Cannot load OOT2 data: unsupported format or no Spark session")
                        oot2_data = None
                except Exception as spark_error:
                    log_message(JOB_ID, f"⚠️ Direct Spark loading failed: {{str(spark_error)}}")
                    oot2_data = None
            
            if oot2_data:
                try:
                    row_count = oot2_data.count()
                    log_message(JOB_ID, f"✅ OOT2 file data loaded: {row_count} rows")
                    log_message(JOB_ID, f"✅ OOT2 columns: {oot2_data.columns}")
                except Exception as count_error:
                    log_message(JOB_ID, f"⚠️ OOT2 data loaded but count failed: {str(count_error)}")
            else:
                log_message(JOB_ID, "⚠️ OOT2 data is None after loading attempt")
        except Exception as e:
            log_message(JOB_ID, f"⚠️ OOT2 file loading failed: {str(e)} - will skip OOT2")
            import traceback
            log_message(JOB_ID, f"⚠️ OOT2 traceback: {traceback.format_exc()}")
    
    elif config.get('oot2_config'):
        oot2_config = config['oot2_config']
        if oot2_config.get('data_source'):
            source_type = oot2_config.get('source_type', 'existing')
            data_source = oot2_config['data_source']
            
            # For local execution, use data_manager directly. For Dataproc, convert to GCS paths
            if is_local_execution:
                # Local execution - use data_manager directly with original file path
                log_message(JOB_ID, f"🏠 Local execution - loading OOT2 config via data_manager: {data_source}")
                try:
                    oot2_data, _ = data_manager.load_data(
                        data_source, 
                        source_type=source_type,
                        **oot2_config.get('options', {})
                    )
                    log_message(JOB_ID, f"✅ OOT2 data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT2 loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
            else:
                # Dataproc execution - convert to GCS paths
                if not data_source.startswith('gs://'):
                    if source_type == 'upload':
                        data_source = f"gs://{GCS_DATA_BUCKET}/automl_results/{JOB_ID}/{data_source}"
                        log_message(JOB_ID, f"📁 Upload OOT2 config - converted to GCS path: {data_source}")
                    elif source_type == 'existing':
                        data_source = f"gs://{GCS_DATA_BUCKET}/data/{data_source}"
                        log_message(JOB_ID, f"📁 Existing OOT2 config - converted to GCS path: {data_source}")
                else:
                    log_message(JOB_ID, f"📁 Using existing GCS path for OOT2 config: {data_source}")
                
                log_message(JOB_ID, f"📅 Loading OOT2 data from {source_type}: {data_source}")
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
                            log_message(JOB_ID, f"⚠️ Cannot load OOT2 config data: unsupported format or no Spark session")
                            oot2_data = None
                    log_message(JOB_ID, f"✅ OOT2 data loaded: {oot2_data.count()} rows")
                except Exception as e:
                    log_message(JOB_ID, f"⚠️ OOT2 loading failed: {str(e)} - will skip OOT2")
                    oot2_data = None
    
    return oot1_data, oot2_data'''

    def _convert_to_python_format(self, obj):
        """Convert a dictionary to Python-compatible string representation."""
        if isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                key_str = f'"{key}"' if isinstance(key, str) else str(key)
                value_str = self._convert_to_python_format(value)
                items.append(f'{key_str}: {value_str}')
            return '{\n  ' + ',\n  '.join(items) + '\n}'
        elif isinstance(obj, list):
            items = [self._convert_to_python_format(item) for item in obj]
            return '[' + ', '.join(items) + ']'
        elif isinstance(obj, str):
            # Properly escape quotes in strings
            escaped_str = obj.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped_str}"'
        elif isinstance(obj, bool):
            return 'True' if obj else 'False'
        elif obj is None:
            return 'None'
        else:
            return str(obj)

# Example usage
if __name__ == "__main__":
    generator = UnifiedJobScriptGenerator()
    
    # Test local script generation
    config = {
        'job_id': 'test_job',
        'data_file': 'IRIS.csv',
        'target_column': 'species',
        'task_type': 'classification',
        'model_params': {'run_logistic': True},
        'timestamp': datetime.now().isoformat()
    }
    
    local_script = generator.generate_job_script('test_job', config, 'local')
    print("✅ Local script generated successfully")
    print(f"📏 Script length: {{len(local_script)}} characters")
    
    # Test Dataproc script generation
    dataproc_script = generator.generate_job_script('test_job', config, 'dataproc', {}, 'batch_123')
    print("✅ Dataproc script generated successfully")
    print(f"📏 Script length: {{len(dataproc_script)}} characters")
