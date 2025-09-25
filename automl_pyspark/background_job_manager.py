"""
Background Job Manager for AutoML PySpark

This module handles background job execution independently from the Streamlit UI,
preventing UI freezing and enabling real-time log streaming.

Supports three execution modes:
1. Local background threads (default)
2. External message queue (Pub/Sub or Cloud Tasks)
3. Dataproc Serverless (Google Cloud managed Spark)
"""

import os
import json
import time
import subprocess
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Union
import signal
import sys

class BackgroundJobManager:
    """
    Manages background AutoML job execution.  By default jobs are
    executed in local background threads.  Optionally, jobs can be
    dispatched to an external message queue (e.g. Google Cloud Pub/Sub
    or Cloud Tasks) or to Dataproc Serverless for managed Spark execution.
    
    Execution modes:
    - Local: Set no environment variables (default)
    - Queue: Set USE_GCP_QUEUE=true
    - Dataproc: Set ENABLE_DATAPROC_SERVERLESS=true
    """

    def __init__(self, jobs_dir: str = None, use_gcp_queue: Union[bool, None] = None):
        # Determine execution mode from environment variables
        env_flag = os.getenv("USE_GCP_QUEUE", "false").lower() in ("1", "true", "yes")
        self.use_gcp_queue = use_gcp_queue if use_gcp_queue is not None else env_flag
        
        # Check if Dataproc Serverless is enabled
        self.use_dataproc_serverless = (
            os.getenv("ENABLE_DATAPROC_SERVERLESS", "false").lower() in ("1", "true", "yes") or
            os.getenv("USE_DATAPROC_SERVERLESS", "false").lower() in ("1", "true", "yes")
        )
        
        # Set jobs directory based on execution mode
        if self.use_dataproc_serverless:
            # For Dataproc Serverless, use current working directory
            self.jobs_dir = os.path.join(os.getcwd(), 'automl_jobs')
        elif jobs_dir is not None:
            # Use provided jobs directory
            self.jobs_dir = jobs_dir
        else:
            # For local execution, use the directory containing this file
            self.jobs_dir = os.path.join(os.path.dirname(__file__), 'automl_jobs')
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.job_logs: Dict[str, queue.Queue] = {}
        self.job_threads: Dict[str, threading.Thread] = {}
        
        # Initialize Dataproc Serverless manager if enabled
        self.dataproc_manager = None
        if self.use_dataproc_serverless:
            try:
                from dataproc_serverless_manager import DataprocServerlessManager
                self.dataproc_manager = DataprocServerlessManager()
                print("✅ Dataproc Serverless Manager initialized")
            except ImportError as e:
                print(f"⚠️ Dataproc Serverless Manager not available: {e}")
                self.use_dataproc_serverless = False
            except Exception as e:
                print(f"❌ Failed to initialize Dataproc Serverless Manager: {e}")
                self.use_dataproc_serverless = False
        
        # Log current execution mode
        if self.use_dataproc_serverless:
            print("🚀 Execution mode: Dataproc Serverless")
        elif self.use_gcp_queue:
            print("📤 Execution mode: External Queue (Pub/Sub/Cloud Tasks)")
        else:
            print("🏠 Execution mode: Local Background Threads")
    
    def _clean_config_for_json(self, config: Dict) -> Dict:
        """Clean configuration to remove non-JSON-serializable objects."""
        import copy
        
        def clean_value(value):
            """Recursively clean a value to make it JSON serializable."""
            if value is None:
                return None
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            elif hasattr(value, 'name') and hasattr(value, 'size'):
                # This is likely an UploadedFile object
                return {
                    'type': 'uploaded_file',
                    'name': value.name,
                    'size': value.size,
                    'file_type': getattr(value, 'type', 'unknown')
                }
            elif hasattr(value, '__dict__'):
                # Handle other objects with attributes
                try:
                    return str(value)
                except:
                    return f"<{type(value).__name__} object>"
            else:
                # Try to copy normally
                try:
                    return copy.deepcopy(value)
                except:
                    return str(value)
        
        return clean_value(config)
    
    def _convert_paths_for_dataproc(self, config: Dict, job_id: str) -> Dict:
        """Convert Docker paths to Dataproc Serverless compatible paths."""
        import copy
        
        # Create a deep copy to avoid modifying the original config
        dataproc_config = copy.deepcopy(config)
        
        # Convert output_dir from /app/automl_pyspark/automl_results/job_id to /tmp/automl_results/job_id
        if 'output_dir' in dataproc_config:
            original_output_dir = dataproc_config['output_dir']
            if '/app/automl_pyspark/automl_results/' in original_output_dir:
                dataproc_config['output_dir'] = f"/tmp/automl_results/{job_id}"
                print(f"🔄 Converted output_dir: {original_output_dir} -> {dataproc_config['output_dir']}")
        
        # Convert config_path from /app/automl_pyspark/config.yaml to /tmp/automl_pyspark/config.yaml
        if 'config_path' in dataproc_config:
            original_config_path = dataproc_config['config_path']
            if '/app/automl_pyspark/config.yaml' in original_config_path:
                dataproc_config['config_path'] = "/tmp/automl_pyspark/config.yaml"
                print(f"🔄 Converted config_path: {original_config_path} -> {dataproc_config['config_path']}")
        
        # Convert all data file paths to GCS paths based on source type
        enhanced_data_config = dataproc_config.get('enhanced_data_config', {})
        source_type = enhanced_data_config.get('source_type', 'existing')
        
        # Convert main data_file
        if 'data_file' in dataproc_config:
            original_data_file = dataproc_config['data_file']
            
            # Handle case where data_file might be a dictionary (flexible data input)
            if isinstance(original_data_file, dict):
                # Extract the actual file path from the dictionary
                if 'data_source' in original_data_file:
                    original_data_file = original_data_file['data_source']
                elif 'file_path' in original_data_file:
                    original_data_file = original_data_file['file_path']
                else:
                    print(f"⚠️ Unable to extract file path from data_file dict: {original_data_file}")
                    original_data_file = str(original_data_file)
            
            # Ensure original_data_file is a string
            if not isinstance(original_data_file, str):
                print(f"⚠️ data_file is not a string: {type(original_data_file)} - {original_data_file}")
                original_data_file = str(original_data_file)
            
            if source_type == 'upload':
                # For uploaded files, they're in automl_results/job_id/
                if original_data_file.startswith('/app/automl_pyspark/automl_results/'):
                    filename = os.path.basename(original_data_file)
                    dataproc_config['data_file'] = f"gs://rapid_modeler_app/automl_results/{job_id}/{filename}"
                    print(f"🔄 Converted upload data_file: {original_data_file} -> {dataproc_config['data_file']}")
            elif source_type == 'existing':
                # For existing files, they're in the data directory
                if not original_data_file.startswith('gs://'):
                    filename = os.path.basename(original_data_file)
                    dataproc_config['data_file'] = f"gs://rapid_modeler_app/data/{filename}"
                    print(f"🔄 Converted existing data_file: {original_data_file} -> {dataproc_config['data_file']}")
        
        # Convert OOT files
        for oot_key in ['oot1_file', 'oot2_file']:
            if oot_key in dataproc_config and dataproc_config[oot_key]:
                original_oot_file = dataproc_config[oot_key]
                
                # Handle case where OOT file might be a dictionary
                if isinstance(original_oot_file, dict):
                    # Extract the actual file path from the dictionary
                    if 'data_source' in original_oot_file:
                        original_oot_file = original_oot_file['data_source']
                    elif 'file_path' in original_oot_file:
                        original_oot_file = original_oot_file['file_path']
                    else:
                        print(f"⚠️ Unable to extract file path from {oot_key} dict: {original_oot_file}")
                        original_oot_file = str(original_oot_file)
                
                # Ensure original_oot_file is a string
                if not isinstance(original_oot_file, str):
                    print(f"⚠️ {oot_key} is not a string: {type(original_oot_file)} - {original_oot_file}")
                    original_oot_file = str(original_oot_file)
                
                # Get source type for this OOT file
                oot_config_key = oot_key.replace('_file', '_config')
                oot_source_type = 'existing'  # default
                if oot_config_key in dataproc_config and dataproc_config[oot_config_key]:
                    oot_source_type = dataproc_config[oot_config_key].get('source_type', 'existing')
                elif source_type:  # fallback to main data source type
                    oot_source_type = source_type
                
                if oot_source_type == 'upload':
                    # For uploaded OOT files, they're in automl_results/job_id/
                    if original_oot_file.startswith('/app/automl_pyspark/automl_results/'):
                        filename = os.path.basename(original_oot_file)
                        dataproc_config[oot_key] = f"gs://rapid_modeler_app/automl_results/{job_id}/{filename}"
                        print(f"🔄 Converted upload {oot_key}: {original_oot_file} -> {dataproc_config[oot_key]}")
                elif oot_source_type == 'existing':
                    # For existing OOT files, they're in the data directory
                    if not original_oot_file.startswith('gs://'):
                        filename = os.path.basename(original_oot_file)
                        dataproc_config[oot_key] = f"gs://rapid_modeler_app/data/{filename}"
                        print(f"🔄 Converted existing {oot_key}: {original_oot_file} -> {dataproc_config[oot_key]}")
        
        # Convert oot_config data_source paths
        for oot_config_key in ['oot1_config', 'oot2_config']:
            if oot_config_key in dataproc_config and dataproc_config[oot_config_key]:
                oot_config = dataproc_config[oot_config_key]
                if 'data_source' in oot_config:
                    original_data_source = oot_config['data_source']
                    
                    # Handle case where data_source might be a dictionary
                    if isinstance(original_data_source, dict):
                        # Extract the actual file path from the dictionary
                        if 'data_source' in original_data_source:
                            original_data_source = original_data_source['data_source']
                        elif 'file_path' in original_data_source:
                            original_data_source = original_data_source['file_path']
                        else:
                            print(f"⚠️ Unable to extract file path from {oot_config_key}.data_source dict: {original_data_source}")
                            original_data_source = str(original_data_source)
                    
                    # Ensure original_data_source is a string
                    if not isinstance(original_data_source, str):
                        print(f"⚠️ {oot_config_key}.data_source is not a string: {type(original_data_source)} - {original_data_source}")
                        original_data_source = str(original_data_source)
                    
                    oot_source_type = oot_config.get('source_type', 'existing')
                    
                    if oot_source_type == 'upload':
                        # For uploaded files, they're in automl_results/job_id/
                        if original_data_source.startswith('/app/automl_pyspark/automl_results/'):
                            filename = os.path.basename(original_data_source)
                            oot_config['data_source'] = f"gs://rapid_modeler_app/automl_results/{job_id}/{filename}"
                            print(f"🔄 Converted upload {oot_config_key}.data_source: {original_data_source} -> {oot_config['data_source']}")
                    elif oot_source_type == 'existing':
                        # For existing files, they're in the data directory
                        if not original_data_source.startswith('gs://'):
                            filename = os.path.basename(original_data_source)
                            oot_config['data_source'] = f"gs://rapid_modeler_app/data/{filename}"
                            print(f"🔄 Converted existing {oot_config_key}.data_source: {original_data_source} -> {oot_config['data_source']}")
        
        return dataproc_config
    
    def _convert_paths_for_local(self, config: Dict, job_id: str) -> Dict:
        """Convert Docker paths to local execution compatible paths."""
        import copy
        
        # Create a deep copy to avoid modifying the original config
        local_config = copy.deepcopy(config)
        
        # Detect if we're running in Docker
        is_docker = os.path.exists('/app') and os.path.isdir('/app')
        
        if is_docker:
            # If we're in Docker, keep the Docker paths as they are
            print(f"🐳 Running in Docker - keeping original paths")
            return local_config
        
        # Convert output_dir from /app/automl_pyspark/automl_results/job_id to local path
        if 'output_dir' in local_config:
            original_output_dir = local_config['output_dir']
            if '/app/automl_pyspark/automl_results/' in original_output_dir:
                local_config['output_dir'] = f"{os.getcwd()}/automl_results/{job_id}"
                print(f"🔄 Converted output_dir for local: {original_output_dir} -> {local_config['output_dir']}")
        
        # Convert config_path from /app/automl_pyspark/config.yaml to local path
        if 'config_path' in local_config:
            original_config_path = local_config['config_path']
            if '/app/automl_pyspark/config.yaml' in original_config_path:
                local_config['config_path'] = f"{os.path.dirname(__file__)}/config.yaml"
                print(f"🔄 Converted config_path for local: {original_config_path} -> {local_config['config_path']}")
        
        return local_config
        
    def start_job(self, job_id: str, config: Dict) -> bool:
        """Start a job in the background."""
        try:
            print(f"🔄 DEBUG: Starting job {job_id}")
            # Create jobs directory if it doesn't exist
            os.makedirs(self.jobs_dir, exist_ok=True)
            print(f"✅ DEBUG: Jobs directory created: {self.jobs_dir}")
            
            # Create organized job directory structure
            job_dir = os.path.join(self.jobs_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            print(f"✅ DEBUG: Job directory created: {job_dir}")
            
            # Save job configuration first (clean non-serializable objects)
            config_file = os.path.join(job_dir, f"{job_id}.json")
            clean_config = self._clean_config_for_json(config)
            with open(config_file, 'w') as f:
                json.dump(clean_config, f, indent=2)
            print(f"✅ DEBUG: Config file saved: {config_file}")
            
            # Create job script using the cleaned config
            print(f"🔄 DEBUG: Creating job script...")
            script_content = self._create_job_script(job_id, clean_config)
            print(f"✅ DEBUG: Script content created, length: {len(script_content)}")
            script_file = os.path.join(job_dir, f"{job_id}_script.py")
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            print(f"✅ DEBUG: Script file written: {script_file}")
            
            # Create log queue for this job
            self.job_logs[job_id] = queue.Queue()

            if self.use_gcp_queue:
                # When using an external queue, publish the job configuration
                # instead of launching it locally.  The worker environment
                # should retrieve the config file from jobs_dir and execute
                # the job.  Publishing is wrapped in a try/except to avoid
                # crashing the UI if credentials are not configured.  Logs
                # will reflect whether publishing was attempted.
                try:
                    self._publish_job_to_queue(job_id, clean_config)
                    self._update_job_status(job_id, "Submitted")
                    self._log_job_message(job_id, f"📤 Job {job_id} published to message queue at {datetime.now().isoformat()}")
                except Exception as e:
                    self._log_job_message(job_id, f"❌ Failed to publish job to queue: {e}")
                    self._update_job_status(job_id, "Failed")
                    return False
            elif self.use_dataproc_serverless:
                # When using Dataproc Serverless, submit the job to Dataproc
                try:
                    # Convert paths for Dataproc Serverless
                    dataproc_config = self._convert_paths_for_dataproc(clean_config, job_id)
                    
                    # Prepare data files for Dataproc submission
                    data_files = self._extract_data_files_from_config(dataproc_config)
                    
                    # Submit to Dataproc Serverless
                    batch_id = self.dataproc_manager.submit_spark_job(
                        job_config=dataproc_config,
                        job_id=job_id,
                        data_files=data_files
                    )
                    
                    self._update_job_status(job_id, "Submitted")
                    self._log_job_message(job_id, f"🚀 Job {job_id} submitted to Dataproc Serverless (Batch ID: {batch_id}) at {datetime.now().isoformat()}")
                    
                    # Store batch ID for tracking
                    self._store_dataproc_job_info(job_id, batch_id, clean_config)
                    
                except Exception as e:
                    self._log_job_message(job_id, f"❌ Failed to submit job to Dataproc Serverless: {e}")
                    self._update_job_status(job_id, "Failed")
                    return False
            else:
                # Start job in background thread locally
                # Convert paths for local execution
                local_config = self._convert_paths_for_local(clean_config, job_id)
                
                # Update the script file with converted paths
                script_content = self._create_job_script(job_id, local_config)
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                thread = threading.Thread(
                    target=self._run_job_background,
                    args=(job_id, script_file),
                    daemon=True
                )
                thread.start()
                self.job_threads[job_id] = thread
                # Update status to Running (not Submitted)
                self._update_job_status(job_id, "Running")
                self._log_job_message(job_id, f"🚀 Job {job_id} started at {datetime.now().isoformat()}")
            return True
        except Exception as e:
            # Log error and update status
            self._log_job_message(job_id, f"❌ Failed to start job: {e}")
            self._update_job_status(job_id, "Failed")
            return False

    def _publish_job_to_queue(self, job_id: str, config: Dict) -> None:
        """
        Publish a job configuration to an external message queue.  The
        method supports two queue backends: Google Cloud Pub/Sub and
        Cloud Tasks.  When the environment variable ``USE_GCP_TASKS``
        is set to ``true`` (case-insensitive), the job will be
        dispatched via Cloud Tasks.  Otherwise Pub/Sub is used by
        default.  Required environment variables for each mode are
        documented below.

        **Pub/Sub Mode (default)**
            Set ``GCP_PUBSUB_TOPIC`` to the full topic path (e.g.
            ``projects/my-project/topics/my-topic``).  The
            ``google-cloud-pubsub`` library must be installed.

        **Cloud Tasks Mode**
            Set ``USE_GCP_TASKS=true`` and provide the following
            variables:

            * ``GCP_TASKS_PROJECT`` – project ID where the queue
              resides.
            * ``GCP_TASKS_LOCATION`` – location/region of the queue.
            * ``GCP_TASKS_QUEUE`` – name of the queue.
            * ``CLOUD_RUN_BASE_URL`` – base URL of the Cloud Run
              service that will process tasks.
            * ``SERVICE_ACCOUNT_EMAIL`` – (optional) service account
              email used for the OIDC token.  If omitted the email is
              inferred from the service account key.

        The job configuration is JSON serialised and included in the
        message or HTTP task body along with the job ID.
        """
        try:
            import json
            # Determine whether to use Cloud Tasks
            use_tasks = os.getenv('USE_GCP_TASKS', 'false').lower() in ('1', 'true', 'yes')
            if use_tasks:
                # Defer to Cloud Tasks helper.  Import lazily to avoid
                # dependency if unused.
                try:
                    from automl_pyspark.gcp_helpers import create_http_task  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "Cloud Tasks dispatch requested but google-cloud-tasks is not installed. "
                        "Install via 'pip install google-cloud-tasks' and set USE_GCP_TASKS=false to fall back to Pub/Sub."
                    ) from e
                project = os.getenv('GCP_TASKS_PROJECT')
                location = os.getenv('GCP_TASKS_LOCATION')
                queue = os.getenv('GCP_TASKS_QUEUE')
                base_url = os.getenv('CLOUD_RUN_BASE_URL')
                if not all([project, location, queue, base_url]):
                    raise RuntimeError(
                        "GCP_TASKS_PROJECT, GCP_TASKS_LOCATION, GCP_TASKS_QUEUE and CLOUD_RUN_BASE_URL must be set "
                        "when USE_GCP_TASKS is true."
                    )
                # Determine the target path from config.  For clustering,
                # classification and regression jobs, we post to a generic
                # '/run-job' endpoint.  Downstream services can inspect the
                # payload to decide how to handle the job.
                target_path = os.getenv('GCP_TASKS_TARGET_PATH', '/run-job')
                service_account_email = os.getenv('SERVICE_ACCOUNT_EMAIL')
                payload = {'job_id': job_id, 'config': config}
                # Create HTTP task
                response = create_http_task(
                    project=project,
                    location=location,
                    queue=queue,
                    target_path=target_path,
                    json_payload=payload,
                    task_id=job_id,
                    service_account_email=service_account_email or None
                )
                # We don't wait for result; Cloud Tasks returns immediately
                return
            else:
                # Use Pub/Sub
                try:
                    from google.cloud import pubsub_v1  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "google-cloud-pubsub library is not installed or configured. "
                        "Install via 'pip install google-cloud-pubsub' or set USE_GCP_TASKS=true to use Cloud Tasks."
                    ) from e
                topic_path = os.getenv('GCP_PUBSUB_TOPIC')
                if not topic_path:
                    raise RuntimeError("GCP_PUBSUB_TOPIC environment variable must be set when using Pub/Sub dispatch.")
                publisher = pubsub_v1.PublisherClient()
                message_bytes = json.dumps({'job_id': job_id, 'config': config}).encode('utf-8')
                future = publisher.publish(topic_path, message_bytes, job_id=job_id)
                future.result(timeout=10)
                
        except Exception as e:
            # Log detailed error information
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            self._log_job_message(job_id, f"❌ Failed to start job: {error_msg}")
            self._log_job_message(job_id, f"🔍 Full traceback: {traceback_msg}")
            self._update_job_status(job_id, "Failed")
            return False
    
    def _create_job_script(self, job_id: str, config: Dict) -> str:
        """Create the job execution script using unified generator."""
        import sys
        import os
        
        # Add multiple possible paths for robust importing
        current_dir = os.path.dirname(__file__)
        sys.path.insert(0, current_dir)
        sys.path.insert(0, os.path.dirname(current_dir))  # Parent directory
        sys.path.insert(0, '/app')  # Docker root
        sys.path.insert(0, '/app/automl_pyspark')  # Docker automl_pyspark
        
        # Try multiple import strategies
        try:
            from automl_pyspark.unified_job_script_generator import UnifiedJobScriptGenerator
        except ImportError:
            try:
                from unified_job_script_generator import UnifiedJobScriptGenerator
            except ImportError as e:
                raise ImportError(f"Could not import UnifiedJobScriptGenerator: {e}. Current dir: {current_dir}, sys.path: {sys.path[:5]}")
        
        generator = UnifiedJobScriptGenerator()
        return generator.generate_job_script(job_id, config, execution_mode="local", jobs_dir=self.jobs_dir)

    def _run_job_background(self, job_id: str, script_file: str):
        """Run job in background thread."""
        try:
            # Set up environment with job_id
            env = os.environ.copy()
            env['JOB_ID'] = job_id
            
            # Use absolute path to the script file
            process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=os.getcwd()  # Use current working directory instead of hardcoded /app
            )
            
            self.running_jobs[job_id] = process
            
            # Stream output in real-time, capturing all script output
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        # Capture all non-empty output
                        if line_stripped:
                            self._log_job_message(job_id, line_stripped)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self._log_job_message(job_id, "✅ Job process completed successfully")
            else:
                self._log_job_message(job_id, f"❌ Job process failed with return code: {return_code}")
                
        except Exception as e:
            self._log_job_message(job_id, f"❌ Background job execution error: {str(e)}")
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            if job_id in self.job_threads:
                del self.job_threads[job_id]
    
    def stop_job(self, job_id: str) -> bool:
        """Stop a running job."""
        if job_id in self.running_jobs:
            try:
                process = self.running_jobs[job_id]
                process.terminate()
                self._log_job_message(job_id, "🛑 Job stopped by user")
                self._update_job_status(job_id, "Stopped")
                return True
            except Exception as e:
                self._log_job_message(job_id, f"❌ Failed to stop job: {str(e)}")
                return False
        return False
    
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        status_file = os.path.join(self.jobs_dir, f"{job_id}_status.txt")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return f.read().strip()
        return "Submitted"
    
    def get_job_logs(self, job_id: str, max_lines: int = 100) -> List[str]:
        """Get recent job logs."""
        log_file = os.path.join(self.jobs_dir, job_id, f"{job_id}_log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-max_lines:] if len(lines) > max_lines else lines
            except Exception:
                return []
        return []
    
    def get_job_progress(self, job_id: str) -> Dict:
        """Get job progress information."""
        # First try to get progress from file
        progress_file = os.path.join(self.jobs_dir, f"{job_id}_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # If job is running, try to detect real progress from logs
                if progress_data.get('current_task') != 'Completed':
                    # Get task type from job config
                    job_config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
                    if os.path.exists(job_config_file):
                        try:
                            with open(job_config_file, 'r') as f:
                                config = json.load(f)
                            task_type = config.get('task_type', 'classification')
                            
                            # Detect progress from logs
                            current_step, current_task = self._detect_progress_from_logs(job_id, task_type)
                            
                            # Update progress if we found more recent activity
                            if current_step > progress_data.get('current_step', 0):
                                progress_data['current_step'] = current_step
                                progress_data['current_task'] = current_task
                                
                                # Calculate progress percentage with special handling for hyperparameter tuning
                                if "Hyperparameter Tuning" in current_task:
                                    # Extract tuning progress if available
                                    import re
                                    tuning_progress = re.search(r'(\d+)%', current_task)
                                    if tuning_progress:
                                        tuning_pct = int(tuning_progress.group(1))
                                        # Base progress is step 5 (Model Building), add tuning progress
                                        base_progress = (5 / progress_data.get('total_steps', 8)) * 100
                                        tuning_adjustment = (tuning_pct / 100) * 0.2  # Tuning is ~20% of total time
                                        progress_data['progress_percentage'] = round(base_progress + tuning_adjustment, 1)
                                    else:
                                        progress_data['progress_percentage'] = round((current_step / progress_data.get('total_steps', 8)) * 100, 1)
                                else:
                                    progress_data['progress_percentage'] = round((current_step / progress_data.get('total_steps', 8)) * 100, 1)
                                
                                progress_data['timestamp'] = datetime.now().isoformat()
                                
                                # Save updated progress
                                with open(progress_file, 'w') as f:
                                    json.dump(progress_data, f, indent=2)
                        except:
                            pass
                
                return progress_data
            except:
                pass
        
        # Fallback to log-based detection
        job_config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
        if os.path.exists(job_config_file):
            try:
                with open(job_config_file, 'r') as f:
                    config = json.load(f)
                task_type = config.get('task_type', 'classification')
                current_step, current_task = self._detect_progress_from_logs(job_id, task_type)
                
                total_steps = 8 if task_type in ['classification', 'regression'] else 6
                return {
                    'current_step': current_step,
                    'total_steps': total_steps,
                    'current_task': current_task,
                    'progress_percentage': round((current_step / total_steps) * 100, 1),
                    'timestamp': datetime.now().isoformat()
                }
            except:
                pass
        
        return {
            'current_step': 0,
            'total_steps': 8,
            'current_task': 'Initializing...',
            'progress_percentage': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_progress_from_logs(self, job_id: str, task_type: str) -> tuple:
        """Detect progress based on log patterns."""
        log_file = os.path.join(self.jobs_dir, job_id, f"{job_id}_log.txt")
        if not os.path.exists(log_file):
            return 0, "Initializing..."
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Define progress patterns for different task types
            if task_type == 'classification':
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data"]),
                    (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features", "2. Feature Selection"]),
                    (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid", "3. Data Splitting"]),
                    (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot", "4. Preparing Out-of-Time"]),
                    (5, "Model Building and Validation", ["Model Building", "training", "XGBoost", "Random Forest", "Logistic", "Gradient", "5. Model Building", "Building models", "Training", "Building clustering models"]),
                    (6, "Model Selection", ["Model Selection", "selecting best", "best model", "6. Model Selection", "Selecting best", "Selecting best clustering model"]),
                    (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model", "7. Generating Scoring", "Generating scoring scripts"]),
                    (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "8. Save model", "Saving model", "Complete model saved"])
                ]
            elif task_type == 'regression':
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data"]),
                    (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features", "2. Feature Selection"]),
                    (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid", "3. Data Splitting"]),
                    (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot", "4. Preparing Out-of-Time"]),
                    (5, "Model Building and Validation", ["Model Building", "training", "Linear Regression", "Random Forest", "Gradient", "5. Model Building", "Building models", "Training"]),
                    (6, "Model Selection", ["Model Selection", "selecting best", "best model", "6. Model Selection", "Selecting best"]),
                    (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model", "7. Generating Scoring", "Generating scoring scripts"]),
                    (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "8. Save model", "Saving model", "Complete model saved"])
                ]
            else:  # clustering
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data", "Loading and processing data"]),
                    (2, "Feature Scaling", ["Feature Scaling", "scaling", "normalize", "2. Feature Scaling", "Feature selection"]),
                    (3, "Clustering Analysis", ["Clustering Analysis", "K-Means", "DBSCAN", "clustering", "3. Clustering Analysis", "Building clustering models", "Training clustering models"]),
                    (4, "Model Building and Validation", ["Model Building", "training", "cluster", "4. Model Building", "Building models", "Training"]),
                    (5, "Model Selection", ["Model Selection", "selecting best", "best model", "5. Model Selection", "Selecting best clustering model", "Selecting best"]),
                    (6, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "6. Save model", "Saving model", "Complete clustering model saved"])
                ]
            
            # Check for completion first
            if "completed successfully" in log_content.lower() or "job completed successfully" in log_content.lower():
                return len(progress_patterns), "Completed"
            
            # Check for hyperparameter tuning progress
            if "best trial:" in log_content.lower() and "100%" in log_content.lower():
                # If hyperparameter tuning is complete, we're likely in model building/validation phase
                if task_type in ['classification', 'regression']:
                    return 5, "Model Building and Validation (Hyperparameter Tuning Complete)"
                else:
                    return 4, "Model Building and Validation (Hyperparameter Tuning Complete)"
            
            # Check for active hyperparameter tuning
            if "best trial:" in log_content.lower() and any(f"{i}%" in log_content for i in range(10, 100, 10)):
                # Extract progress percentage from tuning logs
                import re
                progress_matches = re.findall(r'(\d+)%', log_content)
                if progress_matches:
                    latest_progress = max(int(p) for p in progress_matches)
                    if task_type in ['classification', 'regression']:
                        return 5, "Model Building and Validation (Hyperparameter Tuning: {latest_progress}%)"
                    else:
                        return 4, "Model Building and Validation (Hyperparameter Tuning: {latest_progress}%)"
            
            # Find the highest completed step
            current_step = 0
            current_task = "Initializing..."
            
            for step, task, patterns in progress_patterns:
                if any(pattern.lower() in log_content.lower() for pattern in patterns):
                    current_step = step
                    current_task = task
            
            return current_step, current_task
            
        except Exception:
            return 0, "Initializing..."
    
    def _update_job_status(self, job_id: str, status: str):
        """Update job status."""
        # Create organized job directory if it doesn't exist
        job_dir = os.path.join(self.jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        status_file = os.path.join(job_dir, f"{job_id}_status.txt")
        with open(status_file, 'w') as f:
            f.write(status)
    
    def _log_job_message(self, job_id: str, message: str):
        """Log a message for the job."""
        # Create structured log directory within job folder
        log_dir = os.path.join(self.jobs_dir, job_id, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'job_execution.log')
        
        # Also create compatibility file inside job folder (not outside)
        compat_log_file = os.path.join(self.jobs_dir, job_id, f"{job_id}_log.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Write to structured log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Write to compatibility log file
        with open(compat_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Also add to queue for real-time access
        if job_id in self.job_logs:
            try:
                self.job_logs[job_id].put_nowait(message)
            except queue.Full:
                pass  # Queue is full, skip
    
    def get_running_jobs(self) -> List[str]:
        """Get list of currently running job IDs."""
        return list(self.running_jobs.keys())
    
    def cleanup_completed_jobs(self):
        """Clean up completed job references."""
        completed_jobs = []
        for job_id in list(self.running_jobs.keys()):
            if self.get_job_status(job_id) in ['Completed', 'Failed', 'Stopped']:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            if job_id in self.job_threads:
                del self.job_threads[job_id]
            if job_id in self.job_logs:
                del self.job_logs[job_id]

    def _extract_data_files_from_config(self, config: Dict) -> List[str]:
        """Extract data file paths from job configuration."""
        data_files = []
        
        # Check for data source configuration
        if 'data_source' in config:
            data_source = config['data_source']
            if isinstance(data_source, str) and os.path.exists(data_source):
                data_files.append(data_source)
            elif isinstance(data_source, dict):
                # Handle flexible data input configuration
                if 'file_path' in data_source and os.path.exists(data_source['file_path']):
                    data_files.append(data_source['file_path'])
                elif 'uploaded_file' in data_source:
                    # Handle uploaded file information
                    pass  # Uploaded files are handled differently
        
        # Check for existing files configuration
        if 'existing_files' in config:
            existing_files = config['existing_files']
            if isinstance(existing_files, list):
                for file_path in existing_files:
                    if os.path.exists(file_path):
                        data_files.append(file_path)
        
        # Check for BigQuery configuration (no local files)
        if 'bigquery' in config:
            # BigQuery doesn't have local files to upload
            pass
        
        return data_files
    
    def _store_dataproc_job_info(self, job_id: str, batch_id: str, config: Dict):
        """Store Dataproc job information for tracking."""
        try:
            job_info = {
                'job_id': job_id,
                'batch_id': batch_id,
                'submission_time': datetime.now().isoformat(),
                'execution_mode': 'dataproc_serverless',
                'config': config
            }
            
            # Save to jobs directory
            info_file = os.path.join(self.jobs_dir, f"{job_id}_dataproc_info.json")
            with open(info_file, 'w') as f:
                json.dump(job_info, f, indent=2)
                
            print(f"✅ Dataproc job info stored: {info_file}")
            
        except Exception as e:
            print(f"⚠️ Could not store Dataproc job info: {e}")
    
    def get_dataproc_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a Dataproc Serverless job."""
        if not self.use_dataproc_serverless or not self.dataproc_manager:
            return None
            
        try:
            # Try to find batch ID from stored info
            info_file = os.path.join(self.jobs_dir, f"{job_id}_dataproc_info.json")
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    job_info = json.load(f)
                    batch_id = job_info.get('batch_id')
                    
                    if batch_id:
                        return self.dataproc_manager.get_job_status(batch_id)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not get Dataproc job status: {e}")
            return None

# Global job manager instance
job_manager = BackgroundJobManager() 