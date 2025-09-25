"""
Dataproc Serverless Manager for AutoML PySpark

This module provides integration with Google Cloud Dataproc Serverless
to run Spark jobs in a fully managed, autoscaling environment.

Features:
- Submit Spark jobs to Dataproc Serverless
- Automatic executor scaling (0 to thousands)
- No cluster management required
- Cost optimization (pay per job)
- Integration with existing AutoML pipeline
"""

import os
import json
import time
import logging
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from google.cloud import dataproc_v1
from google.cloud import storage
from google.oauth2 import service_account
from google.protobuf import duration_pb2
import tempfile
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataprocServerlessManager:
    """
    Manages Spark job submission to Google Cloud Dataproc Serverless.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dataproc Serverless manager.
        
        Args:
            config: Configuration dictionary for Dataproc Serverless
        """
        self.config = config or self._get_default_config()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize clients
        self._init_clients()
        
        # Performance monitoring
        self.job_history = []
        self.performance_metrics = {
            'job_durations': [],
            'executor_counts': [],
            'cost_estimates': []
        }
        
        logger.info("🚀 Dataproc Serverless Manager initialized")
    
    def _validate_config(self):
        """Validate the configuration and set defaults for missing values."""
        required_fields = ['project_id', 'region', 'temp_bucket']
        
        for field in required_fields:
            if not self.config.get(field):
                logger.warning(f"⚠️ Missing required field: {field}")
                if field == 'project_id':
                    self.config[field] = os.environ.get('GCP_PROJECT_ID') or os.environ.get('GOOGLE_CLOUD_PROJECT') or ''
                elif field == 'region':
                    self.config[field] = os.environ.get('GCP_REGION', 'us-east1')
                elif field == 'temp_bucket':
                    # Try to construct a default bucket name
                    project_id = self.config.get('project_id', '')
                    if project_id:
                        self.config[field] = f"{project_id}-automl-temp"
                    else:
                        # Use a fallback bucket name that won't cause indexing errors
                        self.config[field] = 'automl-temp-bucket-default'
        
        # Validate project ID
        if not self.config['project_id']:
            raise ValueError("GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable must be set")
        
        # Validate temp bucket - ensure it's not empty and has valid characters
        if not self.config['temp_bucket'] or not self.config['temp_bucket'].strip():
            raise ValueError("GCP_TEMP_BUCKET environment variable must be set or project_id must be available")
        
        # Ensure bucket name starts and ends with alphanumeric characters
        bucket_name = self.config['temp_bucket'].strip()
        if not bucket_name[0].isalnum() or not bucket_name[-1].isalnum():
            # Fix invalid bucket name
            if not bucket_name[0].isalnum():
                bucket_name = 'a' + bucket_name
            if not bucket_name[-1].isalnum():
                bucket_name = bucket_name + 'z'
            self.config['temp_bucket'] = bucket_name
            logger.warning(f"⚠️ Fixed invalid bucket name to: {bucket_name}")
        
        logger.info(f"✅ Configuration validated - Project: {self.config['project_id']}, Region: {self.config['region']}, Bucket: {self.config['temp_bucket']}")
    
    def _init_clients(self):
        """Initialize Google Cloud clients."""
        try:
            # Check if service account credentials are available
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and credentials_path.strip():
                try:
                    with open(credentials_path, 'r') as f:
                        service_account_info = json.load(f)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info
                    )
                    
                    # Configure Dataproc client with regional endpoint
                    client_options = {"api_endpoint": f"{self.config.get('region', 'us-east1')}-dataproc.googleapis.com:443"}
                    self.dataproc_client = dataproc_v1.BatchControllerClient(
                        credentials=credentials,
                        client_options=client_options
                    )
                    self.storage_client = storage.Client(
                        credentials=credentials
                    )
                except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
                    logger.warning(f"⚠️ Could not load service account credentials from {credentials_path}: {e}")
                    logger.info("🔄 Falling back to default credentials (Cloud Run Workload Identity)")
                    # Use default credentials (Cloud Run Workload Identity)
                    # Configure Dataproc client with regional endpoint
                    client_options = {"api_endpoint": f"{self.config.get('region', 'us-east1')}-dataproc.googleapis.com:443"}
                    self.dataproc_client = dataproc_v1.BatchControllerClient(
                        client_options=client_options
                    )
                    self.storage_client = storage.Client()
            else:
                # Use default credentials (Cloud Run Workload Identity)
                # Configure Dataproc client with regional endpoint
                client_options = {"api_endpoint": f"{self.config.get('region', 'us-east1')}-dataproc.googleapis.com:443"}
                self.dataproc_client = dataproc_v1.BatchControllerClient(
                    client_options=client_options
                )
                self.storage_client = storage.Client()
                
            logger.info("✅ Google Cloud clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Dataproc Serverless configuration."""
        return {
            # Dataproc Serverless settings
            'project_id': os.environ.get('GCP_PROJECT_ID', ''),
            'region': os.environ.get('GCP_REGION', 'us-east1'),
            'batch_id_prefix': 'automl-spark',
            
            # Spark configuration
            'spark_version': '3.5',
            'runtime_config_version': '1.0',
            
            # Container image configuration
            'container_image': os.environ.get('DATAPROC_CONTAINER_IMAGE', 'us-central1-docker.pkg.dev/atus-prism-dev/ml-repo/rapid_modeler_dataproc:latest'),  # Use custom runtime image
            
            # Resource configuration - OPTIMIZED FOR LARGE DATASETS
            'executor_count': {
                'min': 4,
                'max': 100
            },
            'executor_memory': '31g',  # Increased for 1.4M+ row datasets
            'executor_cpu': '4',
            'driver_memory': '14g',    # Increased for large dataset coordination
            'driver_cpu': '4',
            
            # Storage configuration
            'temp_bucket': os.environ.get('GCP_TEMP_BUCKET', 'rapid_modeler_app'),
            'results_bucket': os.environ.get('GCP_RESULTS_BUCKET', 'rapid_modeler_app'),
            
            # Job configuration
            'timeout_minutes': 60,
            'idle_timeout_minutes': 10,
            
            # Cost optimization
            'enable_autoscaling': True,
            'max_executor_count': 100,
            'min_executor_count': 2
        }
    
    def _estimate_data_size(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate data size for scaling calculations."""
        try:
            # Check if data size is manually specified
            if 'data_size_mb' in job_config:
                manual_size = job_config['data_size_mb']
                manual_rows = job_config.get('estimated_rows', int(manual_size * 1000))
                logger.info(f"📊 Using manually specified data size: {manual_size:.1f} MB, {manual_rows:,} rows")
                return {'data_size_mb': manual_size, 'estimated_rows': manual_rows}
            
            data_file = job_config.get('data_file')
            if not data_file:
                logger.warning("⚠️ No data_file provided in job_config - using default size estimate")
                return {'data_size_mb': 100, 'estimated_rows': 10000}
            
            logger.info(f"🔍 Estimating data size for: {data_file}")
            
            # Handle different data sources
            if data_file.startswith('gs://'):
                # Cloud Storage file
                try:
                    if not hasattr(self, 'storage_client') or self.storage_client is None:
                        logger.warning("⚠️ Google Cloud Storage client not initialized - cannot estimate GCS file size")
                        raise Exception("Storage client not available")
                    
                    bucket_name = data_file.split('/')[2]
                    blob_name = '/'.join(data_file.split('/')[3:])
                    bucket = self.storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    if blob.exists():
                        size_bytes = blob.size
                        size_mb = size_bytes / (1024 * 1024)
                        
                        # Estimate rows based on file type and size
                        if data_file.endswith('.csv'):
                            estimated_rows = int(size_mb * 1000)  # Rough estimate: 1MB ≈ 1000 rows
                        elif data_file.endswith('.parquet'):
                            estimated_rows = int(size_mb * 5000)  # Parquet is more compressed
                        else:
                            estimated_rows = int(size_mb * 1000)
                        
                        logger.info(f"📊 Detected Cloud Storage file: {size_mb:.1f} MB, estimated {estimated_rows:,} rows")
                        return {'data_size_mb': size_mb, 'estimated_rows': estimated_rows}
                except Exception as e:
                    logger.warning(f"⚠️ Could not estimate Cloud Storage file size: {e}")
            
            elif 'bigquery' in data_file.lower() or ('.' in data_file and len(data_file.split('.')) >= 2):
                # BigQuery table
                logger.info(f"🔍 Attempting BigQuery table size estimation for: {data_file}")
                try:
                    from google.cloud import bigquery
                    logger.info("📦 BigQuery client library imported successfully")
                    
                    client = bigquery.Client()
                    logger.info(f"🔗 BigQuery client initialized for project: {client.project}")
                    
                    # Parse table reference - handle different formats
                    if '.' in data_file:
                        parts = data_file.split('.')
                        if len(parts) >= 3:
                            project_id = parts[0]
                            dataset_id = parts[1]
                            table_id = '.'.join(parts[2:])  # Handle table names with dots
                            logger.info(f"📋 Parsed 3-part reference: {project_id}.{dataset_id}.{table_id}")
                        elif len(parts) == 2:
                            project_id = client.project
                            dataset_id = parts[0]
                            table_id = parts[1]
                            logger.info(f"📋 Parsed 2-part reference: {project_id}.{dataset_id}.{table_id}")
                        else:
                            raise ValueError(f"Invalid BigQuery table reference format: {data_file}")
                    else:
                        raise ValueError(f"BigQuery table reference must contain dots: {data_file}")
                    
                    # Get table reference and metadata
                    table_ref = client.dataset(dataset_id, project=project_id).table(table_id)
                    logger.info(f"🎯 Getting table metadata for: {project_id}.{dataset_id}.{table_id}")
                    
                    table = client.get_table(table_ref)
                    
                    # Get table size and row count
                    size_bytes = table.num_bytes or 0
                    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                    num_rows = table.num_rows or 0
                    
                    logger.info(f"✅ BigQuery table analysis complete:")
                    logger.info(f"   📊 Size: {size_bytes:,} bytes ({size_mb:.1f} MB)")
                    logger.info(f"   📈 Rows: {num_rows:,}")
                    logger.info(f"   📅 Created: {table.created}")
                    logger.info(f"   🔄 Modified: {table.modified}")
                    
                    # Ensure we have reasonable values
                    if size_mb == 0 and num_rows > 0:
                        # Estimate size based on rows if size is not available
                        estimated_size_mb = max(1, num_rows / 10000)  # Rough estimate: 10k rows ≈ 1MB
                        logger.info(f"📊 Size not available, estimating {estimated_size_mb:.1f} MB based on {num_rows:,} rows")
                        size_mb = estimated_size_mb
                    elif size_mb == 0 and num_rows == 0:
                        logger.warning("⚠️ Both size and row count are 0 - table might be empty or metadata unavailable")
                        size_mb = 1  # Minimum size to avoid division by zero
                        num_rows = 1000
                    
                    return {'data_size_mb': size_mb, 'estimated_rows': num_rows}
                    
                except ImportError as e:
                    logger.error(f"❌ BigQuery client library not available: {e}")
                    logger.error("💡 Install with: pip install google-cloud-bigquery")
                except Exception as e:
                    logger.error(f"❌ BigQuery table size estimation failed: {e}")
                    logger.error(f"💡 Ensure table reference is correct: {data_file}")
                    logger.error("💡 Verify service account has BigQuery Data Viewer permissions")
            
            # Default fallback
            logger.warning(f"📊 Could not determine data size for '{data_file}' - using default estimate (100 MB)")
            logger.warning("💡 This may cause suboptimal resource allocation. Ensure:")
            logger.warning("   1. File path is correct (gs://bucket/path for GCS)")
            logger.warning("   2. BigQuery table format is project.dataset.table")
            logger.warning("   3. Service account has proper permissions")
            return {'data_size_mb': 100, 'estimated_rows': 10000}
            
        except Exception as e:
            logger.error(f"❌ Error estimating data size: {e}")
            logger.warning("📊 Using default data size estimate (100 MB) - this may cause suboptimal scaling")
            return {'data_size_mb': 100, 'estimated_rows': 10000}

    def submit_spark_job(
        self,
        job_config: Dict[str, Any],
        job_id: str,
        data_files: List[str] = None,
        dependencies: List[str] = None
    ) -> str:
        """
        Submit a Spark job to Dataproc Serverless.
        
        Args:
            job_config: AutoML job configuration
            job_id: Unique job identifier
            data_files: List of data file paths to upload
            dependencies: List of dependency files to upload
            
        Returns:
            Batch ID from Dataproc Serverless
        """
        try:
            logger.info(f"🚀 Submitting Spark job {job_id} to Dataproc Serverless")
            
            # Validate configuration before proceeding
            if not self.config.get('project_id'):
                raise ValueError("project_id is not configured. Please set GCP_PROJECT_ID environment variable.")
            
            if not self.config.get('temp_bucket'):
                raise ValueError("temp_bucket is not configured. Please set GCP_TEMP_BUCKET environment variable.")
            
            # Create unique batch ID - Dataproc requires lowercase letters, digits, and hyphens only, 4-63 chars
            batch_id = self._sanitize_batch_id(job_id, self.config['batch_id_prefix'])
            
            # Estimate data size for intelligent scaling
            data_info = self._estimate_data_size(job_config)
            job_config.update(data_info)
            logger.info(f"📊 Data analysis complete: {data_info['data_size_mb']:.1f} MB, {data_info['estimated_rows']:,} rows")
            
            # Upload job files to Cloud Storage
            job_files = self._upload_job_files(job_id, job_config, data_files, dependencies, batch_id)
            
            # Create batch request
            batch_request = self._create_batch_request(batch_id, job_files, job_config)
            
            # Submit batch
            operation = self.dataproc_client.create_batch(
                parent=f"projects/{self.config['project_id']}/locations/{self.config['region']}",
                batch=batch_request,
                batch_id=batch_id  # Important: pass the batch_id
            )
            
            logger.info(f"✅ Job {job_id} submitted successfully. Batch ID: {batch_id}")
            logger.info(f"🔄 Batch creation in progress, not waiting for completion to speed up submission")
            
            # Store job metadata without waiting for batch creation to complete
            # This significantly speeds up job submission
            self._store_job_metadata_async(job_id, batch_id, operation, job_config)
            
            return batch_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit job {job_id}: {e}")
            logger.error(f"🔍 Job config keys: {list(job_config.keys()) if job_config else 'None'}")
            logger.error(f"🔍 Data files: {data_files}")
            logger.error(f"🔍 Dependencies: {dependencies}")
            raise
    
    def _upload_job_files(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        data_files: List[str] = None,
        dependencies: List[str] = None,
        batch_id: str = None
    ) -> Dict[str, str]:
        """
        Upload job files to Cloud Storage.
        
        Returns:
            Dictionary mapping file types to Cloud Storage URIs
        """
        # Validate temp_bucket before using it
        if not self.config.get('temp_bucket') or not self.config['temp_bucket'].strip():
            raise ValueError("temp_bucket is not configured or is empty. Please set GCP_TEMP_BUCKET environment variable.")
        
        # Ensure bucket name is valid for Google Cloud Storage
        bucket_name = self.config['temp_bucket'].strip()
        if not bucket_name[0].isalnum() or not bucket_name[-1].isalnum():
            # Fix invalid bucket name
            if not bucket_name[0].isalnum():
                bucket_name = 'a' + bucket_name
            if not bucket_name[-1].isalnum():
                bucket_name = bucket_name + 'z'
            self.config['temp_bucket'] = bucket_name
            logger.warning(f"⚠️ Fixed invalid bucket name to: {bucket_name}")
        
        bucket = self.storage_client.bucket(self.config['temp_bucket'])
        job_files = {}
        
        try:
            # Upload job configuration to organized structure
            config_blob = bucket.blob(f"automl_jobs/{job_id}/{job_id}.json")
            config_blob.upload_from_string(
                json.dumps(job_config, indent=2),
                content_type='application/json'
            )
            job_files['config'] = f"gs://{self.config['temp_bucket']}/automl_jobs/{job_id}/{job_id}.json"
            
            # Upload data files if provided
            if data_files:
                data_uris = []
                for data_file in data_files:
                    if os.path.exists(data_file):
                        blob_name = f"jobs/{job_id}/data/{os.path.basename(data_file)}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(data_file)
                        data_uris.append(f"gs://{self.config['temp_bucket']}/{blob_name}")
                job_files['data'] = data_uris
            
            # Upload dependencies if provided
            if dependencies:
                deps_uris = []
                for dep_file in dependencies:
                    if os.path.exists(dep_file):
                        blob_name = f"jobs/{job_id}/dependencies/{os.path.basename(dep_file)}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(dep_file)
                        deps_uris.append(f"gs://{self.config['temp_bucket']}/{blob_name}")
                job_files['dependencies'] = deps_uris
            
            # Create requirements.txt for package installation using dataproc_requirements.txt
            dataproc_requirements_path = os.path.join(os.path.dirname(__file__), 'dataproc_requirements.txt')
            logger.info(f"🔍 Looking for dataproc_requirements.txt at: {dataproc_requirements_path}")
            
            if os.path.exists(dataproc_requirements_path):
                with open(dataproc_requirements_path, 'r') as f:
                    requirements_content = f.read()
                logger.info(f"📦 Using dataproc_requirements.txt with {len(requirements_content.splitlines())} lines")
                logger.info(f"📦 First 5 lines: {requirements_content.splitlines()[:5]}")
            else:
                logger.error(f"❌ dataproc_requirements.txt not found at {dataproc_requirements_path}")
                logger.info(f"📁 Directory contents: {os.listdir(os.path.dirname(__file__))}")
                
                # Enhanced fallback requirements with all critical packages - Dataproc pinned versions
                requirements_content = """# Enhanced AutoML dependencies (automl_pyspark is packaged separately)
# Machine Learning and Explainability - pinned versions for Dataproc
shap==0.41.0
scipy==1.10.1
seaborn==0.12.2
matplotlib==3.7.5
lightgbm==3.3.5
xgboost==1.7.6

# Data processing - pinned versions for Dataproc
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
optuna==3.4.0
plotly==5.17.0
xlsxwriter==3.0.9
openpyxl==3.1.2
joblib==1.3.2
pyarrow==12.0.1

# Additional dependencies can be added here
# For example:
# custom-package==1.0.0
# another-package>=2.0.0
"""
                logger.warning("⚠️ dataproc_requirements.txt not found, using enhanced fallback requirements with all critical packages")
            
            requirements_blob = bucket.blob(f"automl_jobs/{job_id}/{job_id}_requirements.txt")
            requirements_blob.upload_from_string(requirements_content, content_type='text/plain')
            job_files['requirements'] = f"gs://{self.config['temp_bucket']}/automl_jobs/{job_id}/{job_id}_requirements.txt"
            
            logger.info("📦 Created requirements.txt for package installation")
            
            # Create main job script
            job_script = self._create_job_script(job_id, job_config, job_files, batch_id)
            script_blob = bucket.blob(f"automl_jobs/{job_id}/{job_id}_script.py")
            script_blob.upload_from_string(job_script, content_type='text/plain')
            job_files['script'] = f"gs://{self.config['temp_bucket']}/automl_jobs/{job_id}/{job_id}_script.py"

            # Package the automl_pyspark package as a zip archive. This ensures the package is
            # available to the Dataproc runtime via python_file_uris. We exclude __pycache__
            # directories and large data assets to keep the archive lean.
            # 
            # OPTIMIZATION: Use cached package if available to speed up submission
            try:
                cached_package_uri = self._get_or_create_cached_package(bucket, job_id)
                if cached_package_uri:
                    job_files.setdefault('py_deps', [])
                    job_files['py_deps'].append(cached_package_uri)
                    logger.info(f"📦 Using cached automl_pyspark package: {cached_package_uri}")
                else:
                    logger.warning("⚠️ Failed to create cached package, continuing without archive")
            except Exception as e:
                logger.warning(f"⚠️ Failed to package automl_pyspark: {e}. Continuing without archive.")

            # Upload JAR dependencies from the local libs directory. These include connectors like
            # LightGBM and SynapseML which are required by the AutoML pipelines. We copy them
            # into GCS so that they can be specified in jar_file_uris when the batch is created.
            try:
                libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
                jar_uris = []
                if os.path.isdir(libs_dir):
                    # Check if we're using BigQuery to decide whether to include BigQuery connector
                    data_source = job_config.get('data_file', '')
                    enhanced_config = job_config.get('enhanced_data_config', {})
                    source_type = enhanced_config.get('source_type', 'file')
                    
                    # Only include BigQuery connector if we're actually using BigQuery
                    include_bigquery = (source_type == 'bigquery' or 
                                      (isinstance(data_source, str) and 
                                       '.' in data_source and 
                                       not data_source.endswith('.csv')))
                    
                    # For BigQuery, use the built-in connector instead of uploading our own
                    if include_bigquery:
                        logger.info("🔗 Using built-in BigQuery connector (not uploading JAR)")
                    
                    # OPTIMIZATION: Use cached JARs to speed up submission
                    cached_jar_uris = self._get_or_upload_cached_jars(bucket, libs_dir, include_bigquery)
                    if cached_jar_uris:
                        jar_uris.extend(cached_jar_uris)
                    
                    if jar_uris:
                        # Append to existing jar URIs or create new list
                        existing_jars = job_files.get('jar_uris', [])
                        job_files['jar_uris'] = existing_jars + jar_uris
                        logger.info(f"📦 Using JAR dependencies: {len(jar_uris)} JARs")
            except Exception as e:
                logger.warning(f"⚠️ Failed to upload JAR dependencies: {e}")

            logger.info(f"✅ Job files uploaded successfully for job {job_id}")
            return job_files
            
        except Exception as e:
            logger.error(f"❌ Failed to upload job files for {job_id}: {e}")
            raise
    
    def _create_job_script(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        job_files: Dict[str, str],
        batch_id: str
    ) -> str:
        """Create the main Spark job script using unified script generator."""
        
        # Use the unified job script generator for Dataproc execution
        try:
            import sys
            import os
            # Add current directory to path for imports
            sys.path.insert(0, os.path.dirname(__file__))
            from unified_job_script_generator import UnifiedJobScriptGenerator
            
            # Convert job_config to Python format for script generation
            generator = UnifiedJobScriptGenerator()
            
            # Add GCS bucket info to config for proper path handling
            enhanced_config = job_config.copy()
            enhanced_config['gcs_results_bucket'] = self.config.get('results_bucket', self.config['temp_bucket'])
            enhanced_config['gcs_temp_bucket'] = self.config['temp_bucket']
            
            # Generate the comprehensive Dataproc script
            script = generator.generate_job_script(
                job_id=job_id,
                config=enhanced_config,
                execution_mode="dataproc",
                job_files=job_files,
                batch_id=batch_id
            )
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Failed to generate unified script: {e}")
            # Fallback to original script generation
            script = '''#!/usr/bin/env python3
"""
AutoML PySpark Job Script for Dataproc Serverless
Job ID: {{job_id}}
Generated: {{generation_time}}
"""

import sys
import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, expr

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main AutoML job execution."""
    # Get job_id from command line arguments or environment
    job_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('JOB_ID', 'unknown')
    
    try:
        logger.info("🚀 Starting AutoML job " + str(job_id))
        
        # Initialize Spark session - Dataproc Serverless compatible configuration
        # NO local networking configs - these break serverless connectivity
        spark = SparkSession.builder \\
            .appName("AutoML-" + str(job_id)) \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \\
            .config("spark.executor.heartbeatInterval", "60s") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \\
            .getOrCreate()
        
        # Install required packages from requirements.txt
        logger.info("📦 Installing required packages...")
        try:
            import subprocess
            
            # Install packages from requirements.txt, but skip automl_pyspark (it's already packaged)
            requirements_uri = "gs://{{temp_bucket}}/automl_jobs/{{job_id}}/{{job_id}}_requirements.txt"
            if requirements_uri.startswith('gs://'):
                # Download requirements.txt from GCS
                rdd = spark.sparkContext.wholeTextFiles(requirements_uri)
                _, requirements_content = rdd.collect()[0]
                
                # Filter out automl_pyspark from requirements (it's already packaged)
                filtered_requirements = []
                for line in requirements_content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and 'automl_pyspark' not in line:
                        filtered_requirements.append(line)
                
                if filtered_requirements:
                    # Write filtered requirements to local file
                    with open('/tmp/requirements.txt', 'w') as f:
                        f.write('\\n'.join(filtered_requirements))
                    
                    # Install packages
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', '/tmp/requirements.txt'])
                    logger.info("✅ Packages installed successfully")
                else:
                    logger.info("ℹ️ No additional packages to install (automl_pyspark is already packaged)")
            else:
                logger.warning("⚠️ No requirements.txt found, using pre-installed packages")
        except Exception as e:
            logger.warning("⚠️ Package installation failed: " + str(e))
            logger.info("ℹ️ Continuing with pre-installed packages")
        
        # CRITICAL: Let Dataproc Serverless manage ALL networking configurations
        # Don't set spark.driver.host, spark.driver.bindAddress, or spark.master
        # These break serverless connectivity and cause "Connection to master failed" errors
        
        # Only remove problematic local configurations if they exist
        if spark.conf.get("spark.master", "").startswith("local"):
            spark.conf.unset("spark.master")
            logger.info("✅ Removed local master configuration")
        
        # Additional serverless-specific configurations (only safe ones)
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
        
        # Arrow configurations for better performance
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        
        logger.info("✅ Spark session configured for Dataproc Serverless")
        
        # IMMEDIATE JAR VERIFICATION - Check what's available right after Spark init
        logger.info("🔍 IMMEDIATE JAR verification after Spark initialization...")
        try:
            # Check spark.jars configuration
            jars_conf = spark.sparkContext.getConf().get("spark.jars")
            if jars_conf:
                logger.info("📦 JAR files in spark.jars configuration:")
                for jar in jars_conf.split(","):
                    if jar.strip():
                        logger.info("   - " + jar.strip())
            else:
                logger.warning("⚠️ spark.jars configuration is empty")
            
            # Check spark.submit.pyFiles configuration
            pyfiles_conf = spark.sparkContext.getConf().get("spark.submit.pyFiles")
            if pyfiles_conf:
                logger.info("🐍 Python files in spark.submit.pyFiles configuration:")
                for py_file in pyfiles_conf.split(","):
                    if py_file.strip():
                        logger.info("   - " + py_file.strip())
            else:
                logger.info("ℹ️ spark.submit.pyFiles configuration is empty")
                
            # Check if we can access the BigQuery format
            try:
                # This will fail if the BigQuery connector isn't loaded
                test_reader = spark.read.format("bigquery")
                logger.info("✅ BigQuery format is available (connector JAR loaded)")
            except Exception as e:
                logger.error("❌ BigQuery format is NOT available: " + str(e))
                
        except Exception as e:
            logger.error("❌ Error during JAR verification: " + str(e))
        
        # NOTE: No need for external compatibility helper functions
        # All logic is now self-contained in this script
        
        logger.info("✅ Spark session initialized")
        
        # Load job configuration
        config_uri = "{config_uri}"
        if config_uri.startswith('gs://'):
            # Download config from Cloud Storage - read the WHOLE file, not just first line
            rdd = spark.sparkContext.wholeTextFiles(config_uri)
            _, config_content = rdd.collect()[0]
            job_config = json.loads(config_content)
        else:
            # Use local config
            with open(config_uri, 'r') as f:
                job_config = json.load(f)
        
        logger.info("✅ Job configuration loaded: " + str(job_config.get('model_name', 'Unknown')))
        
        # Execute AutoML pipeline based on task type
        task_type = job_config.get('task_type', 'classification')
        
        if task_type == 'classification':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running classification AutoML pipeline")
            from automl_pyspark.classification.automl_classifier import AutoMLClassifier
            automl = AutoMLClassifier(spark_session=spark, config_path=job_config, output_dir=f"/tmp/automl_results/{job_id}")
            results = automl.fit(
                train_data=job_config.get('data_file'),
                target_column=job_config.get('target_column')
            )
        elif task_type == 'regression':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running regression AutoML pipeline")
            from automl_pyspark.regression.automl_regressor import AutoMLRegressor
            automl = AutoMLRegressor(spark_session=spark, config_path=job_config, output_dir=f"/tmp/automl_results/{job_id}")
            results = automl.fit(
                train_data=job_config.get('data_file'),
                target_column=job_config.get('target_column')
            )
        elif task_type == 'clustering':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running clustering AutoML pipeline")
            from automl_pyspark.clustering.automl_clusterer import AutoMLClusterer
            automl = AutoMLClusterer(spark_session=spark, config_path=job_config, output_dir=f"/tmp/automl_results/{job_id}")
            results = automl.fit(
                train_data=job_config.get('data_file'),
                target_column=job_config.get('target_column')
            )
        else:
            raise ValueError("Unsupported task type: " + str(task_type))
        
        # Save results to GCS
        results_bucket = "{results_bucket}"
        results_prefix = "jobs/{job_id}/results"
        
        # Create results directory in GCS
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(results_bucket)
        
        # Save job summary (consistent with background job manager)
        job_summary = {{
            'job_id': job_id,
            'completion_time': '{generation_time}',
            'status': 'completed',
            'execution_platform': 'dataproc_serverless',
            'batch_id': '{batch_id}',
            'results_location': f"gs://{{results_bucket}}/automl_results/{{job_id}}/"
        }}
        
        # Save to consistent location: automl_results/{job_id}/
        summary_blob = bucket.blob(f"automl_results/{{job_id}}/job_summary.json")
        summary_blob.upload_from_string(
            json.dumps(job_summary, indent=2, default=str),
            content_type='application/json'
        )
        
        # Also save job metadata to automl_jobs for consistency
        job_metadata = {{
            'job_id': job_id,
            'batch_id': '{batch_id}',
            'submission_time': '{generation_time}',
            'job_config': job_config,
            'execution_platform': 'dataproc_serverless'
        }}
        
        # Upload job metadata to the correct automl_jobs structure
        metadata_blob = bucket.blob(f"automl_jobs/{{job_id}}/{{job_id}}_metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(job_metadata, indent=2, default=str),
            content_type='application/json'
        )
        logger.info(f"✅ Job metadata uploaded to GCS: gs://{{results_bucket}}/automl_jobs/{{job_id}}/{{job_id}}_metadata.json")
        
        # Also upload the original job configuration for consistency
        job_config_blob = bucket.blob(f"automl_jobs/{{job_id}}/{{job_id}}.json")
        job_config_blob.upload_from_string(
            json.dumps(job_config, indent=2, default=str),
            content_type='application/json'
        )
        logger.info(f"✅ Job configuration uploaded to GCS: gs://{{results_bucket}}/automl_jobs/{{job_id}}/{{job_id}}.json")
        
        # Save AutoML results if available (consistent location)
        if 'results' in locals():
            try:
                results_blob = bucket.blob(f"automl_results/{{job_id}}/automl_results.json")
                results_blob.upload_from_string(
                    json.dumps(results, indent=2, default=str),
                    content_type='application/json'
                )
                logger.info("💾 AutoML results saved to GCS")
            except Exception as e:
                logger.warning(f"⚠️ Could not save AutoML results: {{e}}")
        
        logger.info(f"💾 Job results saved to: gs://{{results_bucket}}/automl_results/{{job_id}}/")
        logger.info("🔍 Skipping JAR verification to avoid BigQuery dataset parsing errors...")
        logger.info("✅ BigQuery connector will be tested during actual data loading if needed")
        
        # List all loaded JARs for debugging
        try:
            jars_conf = spark.sparkContext.getConf().get("spark.jars")
            if jars_conf:
                logger.info("📦 Loaded JAR files:")
                for jar in jars_conf.split(","):
                    if jar.strip():
                        logger.info("   - " + jar.strip())
            else:
                logger.warning("⚠️ No JAR files found in spark.jars configuration")
        except Exception as e:
            logger.warning("⚠️ Could not retrieve JAR configuration: " + str(e))
        
        # Check Python files availability
        try:
            python_files_conf = spark.sparkContext.getConf().get("spark.submit.pyFiles")
            if python_files_conf:
                logger.info("🐍 Loaded Python files:")
                for py_file in python_files_conf.split(","):
                    if py_file.strip():
                        logger.info("   - " + py_file.strip())
            else:
                logger.info("ℹ️ No additional Python files loaded via spark.submit.pyFiles")
        except Exception as e:
            logger.info("ℹ️ Could not retrieve Python files configuration: " + str(e))
        
        # Execute AutoML pipeline
        logger.info("✅ AutoML pipeline completed successfully")
        
        # Save results to Cloud Storage (consistent with background job manager)
        results_bucket = "{results_bucket}"
        if results_bucket:
            logger.info("📊 Uploading results to Cloud Storage...")
            try:
                from google.cloud import storage
                storage_client = storage.Client()
                bucket = storage_client.bucket(results_bucket)
                
                # Upload files from automl_results directory
                local_results_dir = f"/tmp/automl_results/{job_id}"
                if os.path.exists(local_results_dir):
                    logger.info(f"📁 Found local results directory: {{local_results_dir}}")
                    
                    # Create the automl_results/{job_id} directory in GCS
                    gcs_results_prefix = f"automl_results/{{job_id}}/"
                    
                    uploaded_files = []
                    for root, dirs, files in os.walk(local_results_dir):
                        for file in files:
                            if file.endswith(('.xlsx', '.png', '.csv', '.txt', '.py', '.json', '.pkl', '.joblib')):
                                local_file_path = os.path.join(root, file)
                                # Create GCS path maintaining directory structure
                                relative_path = os.path.relpath(local_file_path, local_results_dir)
                                gcs_path = f"{{gcs_results_prefix}}{{relative_path}}"
                                
                                try:
                                    blob = bucket.blob(gcs_path)
                                    blob.upload_from_filename(local_file_path)
                                    uploaded_files.append(file)
                                    logger.info(f"✅ Uploaded {{file}} to gs://{{results_bucket}}/{{gcs_path}}")
                                except Exception as upload_error:
                                    logger.warning(f"⚠️ Failed to upload {{file}}: {{upload_error}}")
                    
                    if uploaded_files:
                        logger.info(f"✅ Successfully uploaded {{len(uploaded_files)}} files to gs://{{results_bucket}}/{{gcs_results_prefix}}")
                        
                        # Create a results summary file
                        results_summary = {{
                            'job_id': job_id,
                            'completion_time': '{generation_time}',
                            'status': 'completed',
                            'execution_platform': 'dataproc_serverless',
                            'batch_id': '{batch_id}',
                            'results_location': f"gs://{{results_bucket}}/{{gcs_results_prefix}}",
                            'uploaded_files': uploaded_files,
                            'total_files': len(uploaded_files)
                        }}
                        
                        summary_blob = bucket.blob(f"{{gcs_results_prefix}}results_summary.json")
                        summary_blob.upload_from_string(
                            json.dumps(results_summary, indent=2, default=str),
                            content_type='application/json'
                        )
                        logger.info(f"✅ Results summary uploaded to gs://{{results_bucket}}/{{gcs_results_prefix}}results_summary.json")
                    else:
                        logger.warning("⚠️ No files were uploaded to Cloud Storage")
                else:
                    logger.warning(f"⚠️ Local results directory not found: {{local_results_dir}}")
                
            except Exception as e:
                logger.error(f"❌ Failed to upload results to Cloud Storage: {{e}}")
                logger.info("ℹ️ Check local filesystem for results")
        
        logger.info("🎉 AutoML job " + str(job_id) + " completed successfully")
        
        # Create job status file and upload to GCS for consistency
        try:
            # Use /tmp for Dataproc environment (writable directory)
            jobs_dir = "/tmp/automl_jobs"
            os.makedirs(jobs_dir, exist_ok=True)
            job_dir = os.path.join(jobs_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            status_file = os.path.join(job_dir, f"{job_id}_status.txt")
            with open(status_file, 'w') as f:
                f.write("COMPLETED")
            
            # Upload status file to GCS
            status_blob = bucket.blob(f"automl_jobs/{{job_id}}/{{job_id}}_status.txt")
            status_blob.upload_from_filename(status_file)
            logger.info(f"✅ Status file uploaded to GCS: gs://{{results_bucket}}/automl_jobs/{{job_id}}/{{job_id}}_status.txt")
        except Exception as e:
            logger.warning(f"⚠️ Could not create/upload status file: {{e}}")
        
        # Create local log file for Streamlit compatibility (consistent with background job manager)
        logger.info("📝 Creating local log file for Streamlit compatibility...")
        try:
            # Create automl_jobs directory structure using /tmp for Dataproc environment
            jobs_dir = "/tmp/automl_jobs"
            job_dir = os.path.join(jobs_dir, job_id)
            log_dir = os.path.join(job_dir, 'logs')
            
            os.makedirs(jobs_dir, exist_ok=True)
            os.makedirs(job_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            # Create local log file in the proper structure
            local_log_file = os.path.join(log_dir, 'job_execution.log')
            compat_log_file = os.path.join(job_dir, f"{job_id}_log.txt")
            
            # Get the batch ID from the job
            batch_id = "{batch_id}"
            
            # Download logs from GCS and save locally
            if batch_id and batch_id != "unknown":
                try:
                    # Get logs from Dataproc using the batch resource name
                    batch_resource_name = f"projects/{project_id}/locations/{region}/batches/{batch_id}"
                    
                    # Download logs from GCS directly
                    from google.cloud import dataproc_v1
                    from google.cloud import storage
                    
                    # Use region-specific endpoint for Dataproc client
                    dataproc_client = dataproc_v1.BatchControllerClient(
                        client_options={{"api_endpoint": f"{region}-dataproc.googleapis.com:443"}}
                    )
                    storage_client = storage.Client()
                    
                    # Use the known output URI pattern for Dataproc Serverless
                    # The output URI follows the pattern: gs://bucket/google-cloud-dataproc-metainfo/.../driveroutput
                    output_uri_pattern = f"gs://{{results_bucket}}/google-cloud-dataproc-metainfo/*/jobs/srvls-batch-*/driveroutput*"
                    
                    # Find logs for the current job only by using the job_id
                    # We'll look for the most recent driveroutput files since we can't easily match batch_id
                    metainfo_prefix = f"google-cloud-dataproc-metainfo/"
                    bucket = storage_client.bucket(results_bucket)
                    blobs = bucket.list_blobs(prefix=metainfo_prefix)
                    
                    log_content = []
                    found_logs = False
                    current_job_id = "{{job_id}}"
                    
                    # Get all driveroutput files and sort by creation time (most recent first)
                    driveroutput_blobs = []
                    for blob in blobs:
                        if ('driveroutput' in blob.name and 
                            blob.name.endswith(('driveroutput', 'driveroutput.000000000', 'driveroutput.000000001', 'driveroutput.000000002', 'driveroutput.000000003'))):
                            driveroutput_blobs.append(blob)
                    
                    # Sort by creation time (most recent first)
                    driveroutput_blobs.sort(key=lambda x: x.time_created, reverse=True)
                    
                    # Take the most recent set of driveroutput files (should be our job)
                    if driveroutput_blobs:
                        # Group by job directory (everything before the last slash)
                        job_dirs = {}
                        for blob in driveroutput_blobs:
                            job_dir = '/'.join(blob.name.split('/')[:-1])
                            if job_dir not in job_dirs:
                                job_dirs[job_dir] = []
                            job_dirs[job_dir].append(blob)
                        
                        # Get the most recent job directory
                        if job_dirs:
                            most_recent_job_dir = max(job_dirs.keys(), key=lambda x: max(blob.time_created for blob in job_dirs[x]))
                            recent_blobs = job_dirs[most_recent_job_dir]
                            
                            for blob in recent_blobs:
                                try:
                                    content = blob.download_as_text()
                                    log_content.append(content)
                                    found_logs = True
                                    logger.info(f"✅ Downloaded log from: {{blob.name}}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Could not download log from {{blob.name}}: {{e}}")
                    
                    if found_logs and log_content:
                        # Write to structured log file
                        with open(local_log_file, 'w', encoding='utf-8') as f:
                            f.write('\\n'.join(log_content))
                        logger.info(f"✅ Structured log file created: {{local_log_file}}")
                        
                        # Write to compatibility log file
                        with open(compat_log_file, 'w', encoding='utf-8') as f:
                            f.write('\\n'.join(log_content))
                        logger.info(f"✅ Compatibility log file created: {{compat_log_file}}")
                        
                        # Upload log files to structured GCS directory within automl_jobs
                        try:
                            # Upload structured log file
                            log_blob = bucket.blob(f"automl_jobs/{{job_id}}/logs/job_execution.log")
                            log_blob.upload_from_filename(local_log_file)
                            logger.info(f"✅ Structured log uploaded to GCS: gs://{{results_bucket}}/automl_jobs/{{job_id}}/logs/job_execution.log")
                            
                            # Upload compatibility log file
                            compat_log_blob = bucket.blob(f"automl_jobs/{{job_id}}/{{job_id}}_log.txt")
                            compat_log_blob.upload_from_filename(compat_log_file)
                            logger.info(f"✅ Compatibility log uploaded to GCS: gs://{{results_bucket}}/automl_jobs/{{job_id}}/{{job_id}}_log.txt")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not upload log files to GCS: {{e}}")
                    else:
                        logger.warning("⚠️ No driver output logs found in metainfo directory")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not download logs: {{e}}")
            else:
                logger.warning("⚠️ No batch ID available for log download")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not create local log file: {{e}}")
        
        return 0
        
    except Exception as e:
        logger.error("❌ AutoML job " + str(job_id) + " failed: " + str(e))
        return 1
    
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    sys.exit(main())
'''
        
        # Format the script with actual config values using string replacement
        # First convert double braces to single braces
        script = script.replace('{{', '{').replace('}}', '}')
        
        # Then replace the actual placeholders
        replacements = {
            '{job_id}': job_id,
            '{batch_id}': batch_id,
            '{project_id}': self.config['project_id'],
            '{region}': self.config['region'],
            '{generation_time}': datetime.now().isoformat(),
            '{temp_bucket}': self.config['temp_bucket'],
            '{results_bucket}': self.config.get('results_bucket', self.config['temp_bucket']),
            '{config_uri}': job_files.get('config', '')
        }
        
        for placeholder, value in replacements.items():
            script = script.replace(placeholder, value)
        
        # Add job_id as an argument to the job config
        job_config['args'] = [job_id]
        
        return script
    
    def _create_batch_request(
        self,
        batch_id: str,
        job_files: Dict[str, str],
        job_config: Dict[str, Any]
    ) -> dataproc_v1.Batch:
        """Create the Dataproc Serverless batch request."""
        
        # Calculate optimal resources based on data size and complexity
        executor_count = self._calculate_executor_count(job_config)
        executor_cores = self._calculate_executor_cores(job_config)
        executor_memory = self._calculate_executor_memory(job_config)
        driver_memory = self._calculate_driver_memory(job_config)
        
        logger.info(f"🎯 Final resource allocation:")
        logger.info(f"   Executors: {executor_count}")
        logger.info(f"   Executor cores: {executor_cores}")
        logger.info(f"   Executor memory: {executor_memory}")
        logger.info(f"   Driver memory: {driver_memory}")
        
        # Container image configuration
        container_image = job_config.get('container_image') or self.config.get('container_image')
        if container_image:
            logger.info(f"📦 Using custom container image: {container_image}")
        else:
            logger.info("📦 Using default Dataproc runtime image")
        
        batch = dataproc_v1.Batch(
            pyspark_batch=dataproc_v1.PySparkBatch(
                main_python_file_uri=job_files['script'],
                args=job_config.get('args', []),
                # Use requirements.txt for package installation
                python_file_uris=job_files.get('py_deps', []),
                # Use any JAR URIs uploaded with the job. For Dataproc, use the built-in BigQuery connector
                # which is already available in the Dataproc runtime
                jar_file_uris=job_files.get('jar_uris', []),
                file_uris=job_files.get('file_uris', []),
                archive_uris=job_files.get('archive_uris', [])
            ),
            runtime_config=dataproc_v1.RuntimeConfig(
                version=job_config.get('runtime_version', '2.1'),
                container_image=container_image if container_image else None,
                properties={
                    'spark.executor.memory': executor_memory,
                    'spark.driver.memory': driver_memory,
                    'spark.executor.cores': str(executor_cores),  # Dynamic core scaling
                    'spark.driver.cores': '4',
                    'spark.sql.adaptive.enabled': 'true',
                    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                    'spark.sql.adaptive.skewJoin.enabled': 'true',
                    'spark.sql.adaptive.localShuffleReader.enabled': 'true',
                    'spark.sql.execution.arrow.pyspark.enabled': 'true',
                    'spark.sql.execution.arrow.pyspark.fallback.enabled': 'true',
                    # XGBoost barrier mode compatibility - disable dynamic allocation
                    'spark.dynamicAllocation.enabled': 'false',
                    'spark.dynamicAllocation.executorIdleTimeout': '60s',
                    # Note: minExecutors and maxExecutors cannot be set when dynamicAllocation is disabled
                    'spark.executor.instances': str(executor_count)  # Calculated executor count
                }
            ),
            environment_config=dataproc_v1.EnvironmentConfig(
                execution_config=dataproc_v1.ExecutionConfig(
                    # NOTE: bucket NAME, NOT 'gs://...'
                    staging_bucket=self.config['temp_bucket']
                ),
                peripherals_config=None
            ),
            # Sanitize label values to conform to Dataproc requirements. Label values must be
            #  lowercase letters, numbers, underscores and hyphens only and no longer than 63 chars.
            labels={
                "job-type": "automl",
                "user": self._sanitize_label_value(job_config.get('user_id', 'unknown')),
                "model": self._sanitize_label_value(job_config.get('model_name', 'unknown'))
            }
        )
        
        return batch
    
    def _sanitize_batch_id(self, raw: str, prefix: str) -> str:
        """Sanitize batch ID to meet Dataproc Serverless requirements.
        
        Dataproc Serverless batch IDs must:
        - Contain only lowercase letters, digits, and hyphens
        - Be 4-63 characters long
        - Start with a letter
        """
        import re
        import time
        
        # Convert to lowercase, replace invalid chars with '-', collapse repeats
        s = re.sub(r'[^a-z0-9-]+', '-', raw.lower())
        s = re.sub(r'-{2,}', '-', s).strip('-')
        
        # Must start with a letter
        if not s or not s[0].isalpha():
            s = 'b' + s
        
        # Compose with prefix and timestamp and trim to 63
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{prefix}-{s}-{ts}"
        
        # Ensure length 4-63
        base = base[:63].strip('-')
        if len(base) < 4:
            base = (base + "-bxxx")[:63]
        
        return base

    def _sanitize_label_value(self, value: str) -> str:
        """
        Sanitize a label value to meet Dataproc Serverless label requirements.

        Dataproc labels must adhere to the following rules:
        - Only lowercase letters, numbers, hyphens and underscores are allowed
        - Must start and end with an alphanumeric character
        - Cannot be empty and must be at most 63 characters long

        Args:
            value: The raw label value (can be None or any type convertible to string)

        Returns:
            A sanitized string that is safe to use as a label value.
        """
        import re
        # Ensure value is a string
        s = str(value or "unknown").lower()
        # Replace invalid characters with hyphens
        s = re.sub(r'[^a-z0-9_-]+', '-', s)
        # Collapse consecutive separators
        s = re.sub(r'[-_]{2,}', '-', s).strip('-_')
        # Ensure it starts with a letter or number
        if not s or not s[0].isalnum():
            s = f"a{s}"
        # Trim to 63 characters
        s = s[:63]
        # Remove trailing separators
        s = s.rstrip('-_')
        # Fallback to 'unknown' if empty after sanitization
        return s if s else "unknown"
    
    def verify_jar_availability(self, job_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify that required JAR files and dependencies are available.
        
        This method checks:
        1. BigQuery connector availability
        2. Custom JAR files if specified
        3. Python library files
        
        Args:
            job_config: Optional job configuration to check specific requirements
            
        Returns:
            Dictionary with verification results
        """
        verification_results = {
            "bigquery_connector": False,
            "custom_jars": [],
            "python_libs": [],
            "errors": []
        }
        
        try:
            # Check if BigQuery connector is accessible
            try:
                # Test BigQuery format availability
                from pyspark.sql import SparkSession
                
                # Create a minimal Spark session for testing
                test_spark = SparkSession.builder \
                    .appName("JAR-Verification") \
                    .config("spark.sql.adaptive.enabled", "false") \
                    .config("spark.driver.host", "localhost") \
                    .config("spark.driver.bindAddress", "127.0.0.1") \
                    .master("local[1]") \
                    .getOrCreate()
                
                # Test BigQuery format
                test_reader = test_spark.read.format("bigquery")
                verification_results["bigquery_connector"] = True
                logger.info("✅ BigQuery connector is available locally")
                
                test_spark.stop()
                
            except Exception as e:
                verification_results["bigquery_connector"] = False
                verification_results["errors"].append(f"BigQuery connector test failed: {e}")
                logger.warning(f"⚠️ BigQuery connector not available locally: {e}")
            
            # Check custom JAR files if specified
            if job_config and 'jar_uris' in job_config:
                for jar_uri in job_config['jar_uris']:
                    if jar_uri.startswith('gs://'):
                        # Check if GCS file exists
                        try:
                            bucket_name = jar_uri.split('/')[2]
                            blob_name = '/'.join(jar_uri.split('/')[3:])
                            bucket = self.storage_client.bucket(bucket_name)
                            blob = bucket.blob(blob_name)
                            
                            if blob.exists():
                                verification_results["custom_jars"].append({
                                    "uri": jar_uri,
                                    "status": "available",
                                    "size": blob.size
                                })
                                logger.info(f"✅ Custom JAR available: {jar_uri}")
                            else:
                                verification_results["custom_jars"].append({
                                    "uri": jar_uri,
                                    "status": "not_found"
                                })
                                verification_results["errors"].append(f"Custom JAR not found: {jar_uri}")
                                logger.warning(f"⚠️ Custom JAR not found: {jar_uri}")
                        except Exception as e:
                            verification_results["custom_jars"].append({
                                "uri": jar_uri,
                                "status": "error",
                                "error": str(e)
                            })
                            verification_results["errors"].append(f"Error checking custom JAR {jar_uri}: {e}")
                            logger.error(f"❌ Error checking custom JAR {jar_uri}: {e}")
            
            # Check Python library files
            automl_dir = os.path.join(os.path.dirname(__file__), '..', 'automl_pyspark')
            if os.path.exists(automl_dir):
                for root, dirs, files in os.walk(automl_dir):
                    for file in files:
                        if file.endswith('.py'):
                            rel_path = os.path.relpath(os.path.join(root, file), automl_dir)
                            verification_results["python_libs"].append({
                                "file": rel_path,
                                "status": "available",
                                "size": os.path.getsize(os.path.join(root, file))
                            })
            
            logger.info(f"✅ JAR verification completed. Found {len(verification_results['python_libs'])} Python library files")
            
        except Exception as e:
            verification_results["errors"].append(f"Verification failed: {e}")
            logger.error(f"❌ JAR verification failed: {e}")
        
        return verification_results
    
    def _calculate_executor_cores(self, job_config: Dict[str, Any]) -> int:
        """Calculate optimal executor cores based on data size and complexity.
        
        Dataproc Serverless only allows 4, 8, or 16 cores per executor.
        """
        data_size_mb = job_config.get('data_size_mb', 100)
        data_size_gb = data_size_mb / 1024
        
        num_models = len([k for k, v in job_config.get('model_params', {}).items() 
                         if k.startswith('run_') and v])
        hp_tuning = job_config.get('advanced_params', {}).get('enable_hyperparameter_tuning', False)
        
        # Calculate base cores requirement
        if data_size_gb < 1.0:  # < 1 GB - small datasets
            base_cores = 4
        elif data_size_gb < 50.0:  # 1-50 GB - medium datasets
            base_cores = 8
        else:  # > 50 GB - large datasets
            base_cores = 16
            
        # Adjust for model complexity (more models = more cores needed)
        if num_models > 5:
            base_cores = min(base_cores * 2, 16)  # Scale up but cap at 16
            
        # Adjust for hyperparameter tuning (more intensive processing)
        if hp_tuning:
            base_cores = min(base_cores * 2, 16)  # Scale up but cap at 16
        
        # Ensure we only use valid Dataproc Serverless core counts: 4, 8, or 16
        if base_cores <= 4:
            final_cores = 4
        elif base_cores <= 8:
            final_cores = 8
        else:
            final_cores = 16
            
        logger.info(f"🔧 Executor cores: {final_cores} (data: {data_size_gb:.1f}GB, models: {num_models}, HP: {hp_tuning})")
        logger.info(f"   Valid Dataproc Serverless core options: [4, 8, 16] - selected: {final_cores}")
        return final_cores

    def _calculate_executor_count(self, job_config: Dict[str, Any]) -> int:
        """Calculate optimal executor count based on comprehensive data analysis."""
        logger.info("🎯 Calculating optimal executor count based on data size and complexity...")
        
        # Base executor count
        base_count = self.config['executor_count']['min']
        max_count = self.config['executor_count']['max']
        
        # Get data size information
        data_size_mb = job_config.get('data_size_mb', 100)
        data_size_gb = data_size_mb / 1024
        
        # Get model complexity information
        model_params = job_config.get('model_params', {})
        enabled_models = [k for k, v in model_params.items() if v]
        num_models = len(enabled_models)
        
        # Get advanced parameters
        advanced_params = job_config.get('advanced_params', {})
        enable_hp_tuning = advanced_params.get('enable_hyperparameter_tuning', False)
        cv_folds = advanced_params.get('cv_folds', 5)
        
        # Calculate scaling factors
        logger.info(f"📊 Data size: {data_size_mb:.1f} MB ({data_size_gb:.2f} GB)")
        logger.info(f"🤖 Number of models: {num_models}")
        logger.info(f"🔧 Hyperparameter tuning: {'Enabled' if enable_hp_tuning else 'Disabled'}")
        
        # Data size scaling (updated per your requirements)
        if data_size_gb < 0.1:  # < 100 MB
            data_factor = 1.0
            logger.info("📊 Small dataset detected - using minimal scaling")
        elif data_size_gb < 1.0:  # < 1 GB
            data_factor = 2.0
            logger.info("📊 Medium dataset detected - moderate scaling")
        elif data_size_gb < 10.0:  # 1-10 GB - YOUR REQUIREMENT
            data_factor = 4.0  # This will give 8-12 executors (2 * 4.0 = 8 base)
            logger.info("📊 Large dataset detected - using 8-12 executors as requested")
        elif data_size_gb < 100.0:  # < 100 GB
            data_factor = 6.0
            logger.info("📊 Very large dataset detected - high scaling")
        else:  # >= 100 GB
            data_factor = 8.0
            logger.info("📊 Massive dataset detected - maximum scaling")
        
        # Model complexity scaling
        if num_models <= 2:
            model_factor = 1.0
        elif num_models <= 4:
            model_factor = 1.3
        elif num_models <= 6:
            model_factor = 1.6
        else:
            model_factor = 2.0
        
        # Hyperparameter tuning scaling
        hp_factor = 1.5 if enable_hp_tuning else 1.0
        
        # Cross-validation scaling
        cv_factor = 1.0 + (cv_folds - 5) * 0.1  # Each additional CV fold adds 10%
        
        # Calculate final executor count
        calculated_count = int(base_count * data_factor * model_factor * hp_factor * cv_factor)
        
        # Apply bounds and ensure reasonable scaling
        final_count = max(base_count, min(calculated_count, max_count))
        
        # Log the calculation details
        logger.info(f"🎯 Scaling calculation:")
        logger.info(f"   Base count: {base_count}")
        logger.info(f"   Data factor: {data_factor:.1f}")
        logger.info(f"   Model factor: {model_factor:.1f}")
        logger.info(f"   HP tuning factor: {hp_factor:.1f}")
        logger.info(f"   CV factor: {cv_factor:.1f}")
        logger.info(f"   Calculated: {calculated_count}")
        logger.info(f"   Final count: {final_count}")
        
        return final_count
    
    def _calculate_executor_memory(self, job_config: Dict[str, Any]) -> str:
        """Calculate optimal executor memory based on data size and complexity."""
        logger.info("💾 Calculating optimal executor memory based on data size and complexity...")
        
        # Get data size information
        data_size_mb = job_config.get('data_size_mb', 100)
        data_size_gb = data_size_mb / 1024
        estimated_rows = job_config.get('estimated_rows', 10000)
        
        # Get model complexity information
        model_params = job_config.get('model_params', {})
        enabled_models = [k for k, v in model_params.items() if v]
        num_models = len(enabled_models)
        
        # Get advanced parameters
        advanced_params = job_config.get('advanced_params', {})
        enable_hp_tuning = advanced_params.get('enable_hyperparameter_tuning', False)
        
        # Base memory calculation - prioritize row count for memory estimation
        if estimated_rows > 2000000:  # > 2M rows - very large dataset
            base_memory_gb = 32  # Need significant memory for large datasets
        elif estimated_rows > 1000000:  # > 1M rows - large dataset
            base_memory_gb = 24
        elif estimated_rows > 500000:  # > 500K rows - medium-large dataset
            base_memory_gb = 20
        elif data_size_gb < 0.1:  # < 100 MB
            base_memory_gb = 4  # Small datasets
        elif data_size_gb < 1.0:  # < 1 GB  
            base_memory_gb = 8  # Medium datasets
        elif data_size_gb < 10.0:  # 1-10 GB - YOUR REQUIREMENT
            base_memory_gb = 20  # Large datasets - minimum 20GB as requested
        elif data_size_gb < 100.0:  # 10-100 GB
            base_memory_gb = 32  # Very large datasets
        else:  # >= 100 GB
            base_memory_gb = 48  # Massive datasets
        
        # Adjust for model complexity
        if num_models > 4:
            base_memory_gb = int(base_memory_gb * 1.2)  # 20% more for multiple models
        
        # Adjust for hyperparameter tuning
        if enable_hp_tuning:
            base_memory_gb = int(base_memory_gb * 1.3)  # 30% more for HP tuning
        
        # Ensure minimum requirements for your use case
        if data_size_gb >= 1.0 or estimated_rows > 100000:  # For 1GB+ datasets or 100K+ rows
            base_memory_gb = max(base_memory_gb, 20)  # Minimum 20GB as requested
        
        # Log the calculation details
        logger.info(f"💾 Memory calculation:")
        logger.info(f"   Data size: {data_size_gb:.2f} GB ({estimated_rows:,} rows)")
        logger.info(f"   Base memory: {base_memory_gb} GB")
        logger.info(f"   Models: {num_models}")
        logger.info(f"   HP tuning: {'Enabled' if enable_hp_tuning else 'Disabled'}")
        
        return f"{base_memory_gb}g"
    
    def _calculate_driver_memory(self, job_config: Dict[str, Any]) -> str:
        """Calculate optimal driver memory based on data size, respecting Dataproc Serverless limits."""
        data_size_mb = job_config.get('data_size_mb', 100)
        data_size_gb = data_size_mb / 1024
        estimated_rows = job_config.get('estimated_rows', 10000)
        
        # Driver needs more memory for coordination and result collection
        # For large datasets (>1M rows), we need significantly more driver memory
        # But must respect Dataproc Serverless limits: max ~7GB per core with overhead
        # Standard tier has 4 cores, so max effective driver memory is ~16GB
        
        if estimated_rows > 2000000:  # > 2M rows - very large dataset
            driver_memory_gb = 16  # Maximum for large datasets
        elif estimated_rows > 1000000:  # > 1M rows - large dataset
            driver_memory_gb = 14
        elif estimated_rows > 500000:  # > 500K rows - medium-large dataset
            driver_memory_gb = 12
        elif data_size_gb < 0.1:
            driver_memory_gb = 6
        elif data_size_gb < 1.0:
            driver_memory_gb = 8
        elif data_size_gb < 10.0:
            driver_memory_gb = 10
        else:
            driver_memory_gb = 12
        
        # Ensure we don't exceed Dataproc Serverless limits
        max_driver_memory = 16  # Conservative limit for standard tier
        driver_memory_gb = min(driver_memory_gb, max_driver_memory)
        
        logger.info(f"🚗 Driver memory: {driver_memory_gb} GB (optimized for {estimated_rows:,} rows, {data_size_gb:.1f} GB)")
        return f"{driver_memory_gb}g"
    
    def _store_job_metadata(
        self,
        job_id: str,
        batch_id: str,
        batch: dataproc_v1.Batch,
        job_config: Dict[str, Any]
    ):
        """Store job metadata for tracking."""
        metadata = {
            'job_id': job_id,
            'batch_id': batch_id,
            'status': 'SUBMITTED',
            'submission_time': datetime.now().isoformat(),
            'batch_uri': batch.name,
            'job_config': job_config
        }
        
        self.job_history.append(metadata)
        
        # Store in Cloud Storage for persistence
        bucket = self.storage_client.bucket(self.config['temp_bucket'])
        blob = bucket.blob(f"jobs/{job_id}/metadata.json")
        blob.upload_from_string(
            json.dumps(metadata, indent=2, default=str),
            content_type='application/json'
        )
    
    def _store_job_metadata_async(
        self,
        job_id: str,
        batch_id: str,
        operation: Any,
        job_config: Dict[str, Any]
    ):
        """Store job metadata without waiting for batch creation to complete."""
        import threading
        
        # Store basic metadata immediately
        metadata = {
            'job_id': job_id,
            'batch_id': batch_id,
            'status': 'SUBMITTED',
            'submission_time': datetime.now().isoformat(),
            'batch_uri': f"projects/{self.config['project_id']}/locations/{self.config['region']}/batches/{batch_id}",
            'job_config': job_config
        }
        
        self.job_history.append(metadata)
        
        # Store basic metadata in Cloud Storage immediately
        try:
            bucket = self.storage_client.bucket(self.config['temp_bucket'])
            blob = bucket.blob(f"jobs/{job_id}/metadata.json")
            blob.upload_from_string(
                json.dumps(metadata, indent=2, default=str),
                content_type='application/json'
            )
            logger.info(f"✅ Basic job metadata stored for {job_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store basic metadata for {job_id}: {e}")
        
        # Update metadata with batch details in background thread
        def update_metadata_with_batch():
            try:
                # Wait for batch creation in background
                batch = operation.result(timeout=60)  # 1 minute timeout
                
                # Update metadata with batch details
                updated_metadata = metadata.copy()
                updated_metadata.update({
                    'batch_uri': batch.name,
                    'batch_creation_time': batch.create_time.isoformat() if batch.create_time else None,
                    'status': 'RUNNING' if batch.state.name == 'RUNNING' else batch.state.name
                })
                
                # Update in job history
                for i, job_meta in enumerate(self.job_history):
                    if job_meta['job_id'] == job_id:
                        self.job_history[i] = updated_metadata
                        break
                
                # Update in Cloud Storage
                blob.upload_from_string(
                    json.dumps(updated_metadata, indent=2, default=str),
                    content_type='application/json'
                )
                logger.info(f"✅ Updated job metadata with batch details for {job_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to update metadata with batch details for {job_id}: {e}")
        
        # Start background thread to update metadata
        thread = threading.Thread(target=update_metadata_with_batch, daemon=True)
        thread.start()
    
    def _get_or_create_cached_package(self, bucket, job_id: str) -> str:
        """Get cached automl_pyspark package or create one if needed."""
        import tempfile
        import zipfile
        import hashlib
        import os
        
        try:
            # Calculate package hash to determine if we need to rebuild
            base_dir = os.path.dirname(os.path.abspath(__file__))
            package_hash = self._calculate_package_hash(base_dir)
            
            # Check if cached package exists
            cached_blob_name = f"cached_packages/automl_pyspark_{package_hash}.zip"
            cached_blob = bucket.blob(cached_blob_name)
            
            if cached_blob.exists():
                logger.info(f"📦 Found cached package: {cached_blob_name}")
                return f"gs://{self.config['temp_bucket']}/{cached_blob_name}"
            
            # Create new package
            logger.info(f"📦 Creating new cached package: {cached_blob_name}")
            
            # Create a temporary zip file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                zip_path = tmp_zip.name
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(base_dir):
                    # Skip __pycache__ and libs/jars that are handled separately
                    if '__pycache__' in dirs:
                        dirs.remove('__pycache__')
                    # Skip libs directory containing large JARs – JARs are passed via jar_file_uris
                    if root == os.path.join(base_dir, 'libs'):
                        continue
                    for file in files:
                        if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.csv') or file.endswith('.txt') or file.endswith('.md'):
                            full_path = os.path.join(root, file)
                            # Create proper automl_pyspark/ directory structure in zip
                            arcname = os.path.join('automl_pyspark', os.path.relpath(full_path, base_dir))
                            zf.write(full_path, arcname=arcname)
            
            # Upload the cached package
            cached_blob.upload_from_filename(zip_path, content_type='application/zip')
            
            # Clean up temp file
            os.unlink(zip_path)
            
            logger.info(f"📦 Created and cached package: {cached_blob_name}")
            return f"gs://{self.config['temp_bucket']}/{cached_blob_name}"
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create cached package: {e}")
            return None
    
    def _calculate_package_hash(self, package_dir: str) -> str:
        """Calculate hash of package contents for caching."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        
        # Hash key Python files to detect changes
        key_files = [
            'automl_classifier.py',
            'automl_regressor.py', 
            'automl_clusterer.py',
            'data_processor.py',
            'unified_job_script_generator.py'
        ]
        
        for root, dirs, files in os.walk(package_dir):
            # Skip __pycache__ directories
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in sorted(files):  # Sort for consistent hashing
                if file.endswith('.py') and (file in key_files or 'automl' in file.lower()):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            hash_md5.update(f.read())
                    except Exception:
                        continue  # Skip files that can't be read
        
        return hash_md5.hexdigest()[:8]  # Use first 8 characters
    
    def _get_or_upload_cached_jars(self, bucket, libs_dir: str, include_bigquery: bool) -> List[str]:
        """Get cached JAR files or upload them if needed."""
        jar_uris = []
        
        try:
            for fname in os.listdir(libs_dir):
                if fname.lower().endswith('.jar'):
                    # Skip BigQuery connector - use built-in instead
                    if 'bigquery' in fname.lower():
                        logger.info(f"⏭️ Skipping BigQuery connector JAR: {fname} (using built-in)")
                        continue
                    
                    # Check if cached JAR exists
                    cached_jar_name = f"cached_jars/{fname}"
                    cached_jar_blob = bucket.blob(cached_jar_name)
                    
                    if cached_jar_blob.exists():
                        logger.info(f"📦 Using cached JAR: {fname}")
                        jar_uris.append(f"gs://{self.config['temp_bucket']}/{cached_jar_name}")
                    else:
                        # Upload new JAR to cache
                        local_jar_path = os.path.join(libs_dir, fname)
                        cached_jar_blob.upload_from_filename(local_jar_path)
                        jar_uris.append(f"gs://{self.config['temp_bucket']}/{cached_jar_name}")
                        logger.info(f"📦 Cached new JAR: {fname}")
            
            return jar_uris
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process cached JARs: {e}")
            return []
    
    def get_job_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the current status of a Dataproc Serverless job."""
        try:
            batch = self.dataproc_client.get_batch(name=batch_id)
            
            status_info = {
                'batch_id': batch_id,
                'state': batch.state.name,
                'state_message': batch.state_message,
                'create_time': batch.create_time.isoformat() if batch.create_time else None,
                'update_time': batch.update_time.isoformat() if batch.update_time else None,
                'operation': batch.operation
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get status for batch {batch_id}: {e}")
            return {'error': str(e)}

    def get_job_logs(self, batch_id: str, max_lines: int = 100) -> List[str]:
        """Retrieve the driver output logs for a Dataproc Serverless job.

        Dataproc Serverless writes the driver output (stdout and stderr) to a
        Cloud Storage location pointed to by the ``driver_output_resource_uri``
        field of the Batch object. This method fetches the file contents from
        GCS and returns the last ``max_lines`` lines to support log tailing in
        the Streamlit UI.

        Args:
            batch_id: The full Dataproc batch resource name (projects/*/locations/*/batches/*).
            max_lines: Maximum number of log lines to return.  Defaults to 100.

        Returns:
            A list of log lines (strings).  Returns an empty list if logs are
            unavailable or an error occurs.
        """
        try:
            # Fetch the batch details to get the driver output URI
            batch = self.dataproc_client.get_batch(name=batch_id)
            output_uri = getattr(batch, 'driver_output_resource_uri', None)
            if not output_uri:
                return []

            # The driver output URI is a Cloud Storage path (gs://bucket/object)
            if not output_uri.startswith('gs://'):
                return []
            # Parse the bucket and blob name
            # The URI may refer directly to a file or to a prefix.  In most cases
            # Dataproc writes the driver output to a single file at this URI.
            parts = output_uri[5:].split('/', 1)
            if len(parts) != 2:
                return []
            bucket_name, blob_name = parts
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            # Download the log content as text
            data = blob.download_as_bytes()
            content = data.decode('utf-8', errors='replace') if data else ''
            lines = content.splitlines()
            # Return the tail of the log
            if max_lines > 0 and len(lines) > max_lines:
                return lines[-max_lines:]
            return lines
        except Exception as e:
            logger.error(f"❌ Failed to retrieve logs for batch {batch_id}: {e}")
            return []
    
    def organize_failed_job_logs(self, job_id: str, batch_id: str) -> bool:
        """Organize logs and metadata for a failed job into the proper folder structure.
        
        Args:
            job_id: The job ID
            batch_id: The Dataproc batch ID
            
        Returns:
            True if logs were successfully organized, False otherwise
        """
        try:
            logger.info(f"📝 Organizing logs for failed job {job_id}")
            
            # Get the batch details
            batch = self.dataproc_client.get_batch(name=batch_id)
            
            # Try to get output URI from different possible locations
            output_uri = None
            if hasattr(batch, 'driver_output_resource_uri') and batch.driver_output_resource_uri:
                output_uri = batch.driver_output_resource_uri
            elif hasattr(batch, 'runtime_info') and batch.runtime_info and hasattr(batch.runtime_info, 'output_uri'):
                output_uri = batch.runtime_info.output_uri
            elif hasattr(batch, 'runtimeInfo') and batch.runtimeInfo and hasattr(batch.runtimeInfo, 'outputUri'):
                output_uri = batch.runtimeInfo.outputUri
            
            if not output_uri:
                logger.warning(f"⚠️ No driver output URI found for batch {batch_id}")
                return False
            
            # Parse the output URI
            if not output_uri.startswith('gs://'):
                logger.warning(f"⚠️ Invalid output URI format: {output_uri}")
                return False
                
            parts = output_uri[5:].split('/', 1)
            if len(parts) != 2:
                logger.warning(f"⚠️ Could not parse output URI: {output_uri}")
                return False
                
            bucket_name, blob_prefix = parts
            bucket = self.storage_client.bucket(bucket_name)
            
            # Find all driver output files
            blobs = bucket.list_blobs(prefix=blob_prefix)
            driver_output_blobs = []
            
            for blob in blobs:
                if 'driveroutput' in blob.name:
                    driver_output_blobs.append(blob)
            
            if not driver_output_blobs:
                logger.warning(f"⚠️ No driver output files found for batch {batch_id}")
                return False
            
            # Download and combine all log content
            log_content = []
            for blob in driver_output_blobs:
                try:
                    content = blob.download_as_text()
                    log_content.append(content)
                    logger.info(f"✅ Downloaded log from: {blob.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download log from {blob.name}: {e}")
            
            if not log_content:
                logger.warning(f"⚠️ No log content could be downloaded for batch {batch_id}")
                return False
            
            # Combine all log content
            combined_logs = '\n'.join(log_content)
            
            # Upload logs to the proper automl_jobs structure
            results_bucket = self.config.get('results_bucket', self.config.get('temp_bucket'))
            if not results_bucket:
                logger.error("❌ No results bucket configured")
                return False
            
            results_bucket_obj = self.storage_client.bucket(results_bucket)
            
            # Upload structured log file
            log_blob = results_bucket_obj.blob(f"automl_jobs/{job_id}/logs/job_execution.log")
            log_blob.upload_from_string(combined_logs, content_type='text/plain')
            logger.info(f"✅ Structured log uploaded to GCS: gs://{results_bucket}/automl_jobs/{job_id}/logs/job_execution.log")
            
            # Upload compatibility log file
            compat_log_blob = results_bucket_obj.blob(f"automl_jobs/{job_id}/{job_id}_log.txt")
            compat_log_blob.upload_from_string(combined_logs, content_type='text/plain')
            logger.info(f"✅ Compatibility log uploaded to GCS: gs://{results_bucket}/automl_jobs/{job_id}/{job_id}_log.txt")
            
            # Create job status file
            status_blob = results_bucket_obj.blob(f"automl_jobs/{job_id}/{job_id}_status.txt")
            status_blob.upload_from_string("FAILED", content_type='text/plain')
            logger.info(f"✅ Status file uploaded to GCS: gs://{results_bucket}/automl_jobs/{job_id}/{job_id}_status.txt")
            
            # Create error summary
            error_summary = {
                'job_id': job_id,
                'batch_id': batch_id,
                'status': 'FAILED',
                'failure_time': datetime.now().isoformat(),
                'error_message': batch.state_message if hasattr(batch, 'state_message') else 'Unknown error',
                'logs_location': f"gs://{results_bucket}/automl_jobs/{job_id}/logs/job_execution.log"
            }
            
            error_blob = results_bucket_obj.blob(f"automl_jobs/{job_id}/{job_id}_error.json")
            error_blob.upload_from_string(
                json.dumps(error_summary, indent=2),
                content_type='application/json'
            )
            logger.info(f"✅ Error summary uploaded to GCS: gs://{results_bucket}/automl_jobs/{job_id}/{job_id}_error.json")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to organize logs for job {job_id}: {e}")
            return False
    
    def list_jobs(self, filter_expr: str = None) -> List[Dict[str, Any]]:
        """List all Dataproc Serverless jobs."""
        try:
            parent = f"projects/{self.config['project_id']}/locations/{self.config['region']}"
            
            if filter_expr:
                request = dataproc_v1.ListBatchesRequest(
                    parent=parent,
                    filter=filter_expr
                )
            else:
                request = dataproc_v1.ListBatchesRequest(parent=parent)
            
            page_result = self.dataproc_client.list_batches(request=request)
            
            jobs = []
            for batch in page_result:
                job_info = {
                    'batch_id': batch.name.split('/')[-1],
                    'state': batch.state.name,
                    'create_time': batch.create_time.isoformat() if batch.create_time else None,
                    'labels': dict(batch.labels) if batch.labels else {}
                }
                jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to list jobs: {e}")
            return []
    
    def cancel_job(self, batch_id: str) -> bool:
        """Cancel a running Dataproc Serverless job."""
        try:
            self.dataproc_client.delete_batch(name=batch_id)
            logger.info(f"✅ Job {batch_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel job {batch_id}: {e}")
            return False
    
    def get_cost_estimate(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the cost of running a job on Dataproc Serverless."""
        # This is a simplified cost estimation
        # In production, you'd want to use the actual pricing API
        
        executor_count = self._calculate_executor_count(job_config)
        estimated_duration_hours = job_config.get('estimated_duration_hours', 1)
        
        # Rough cost estimation (you should adjust based on actual pricing)
        cost_per_executor_hour = 0.10  # USD per executor hour
        estimated_cost = executor_count * estimated_duration_hours * cost_per_executor_hour
        
        return {
            'estimated_cost_usd': round(estimated_cost, 2),
            'executor_count': executor_count,
            'estimated_duration_hours': estimated_duration_hours,
            'cost_per_executor_hour': cost_per_executor_hour
        }
