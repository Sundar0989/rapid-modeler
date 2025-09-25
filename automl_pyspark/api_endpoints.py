#!/usr/bin/env python3
"""
REST API endpoints for AutoML backend.
Provides decoupled interface between Streamlit frontend and AutoML processing.
"""

from flask import Flask, request, jsonify
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from google.cloud import storage
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class AutoMLAPI:
    """API handler for AutoML operations."""
    
    def __init__(self):
        self.bucket_name = os.environ.get('GCP_RESULTS_BUCKET', 'rapid_modeler_app')
        self.project_id = os.environ.get('GCP_PROJECT_ID', 'atus-prism-dev')
        self.region = os.environ.get('GCP_REGION', 'us-east1')
        
        # Initialize GCS client
        try:
            self.storage_client = storage.Client(project=self.project_id)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logger.info(f"✅ GCS client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCS client: {e}")
            self.storage_client = None
            self.bucket = None

    def get_standardized_paths(self, job_id: str) -> Dict[str, str]:
        """
        Get standardized paths for all job outputs.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with all standardized paths
        """
        base_path = f"automl_jobs/{job_id}"
        
        paths = {
            # Job metadata and configuration
            'job_config': f"{base_path}/job_config.json",
            'job_status': f"{base_path}/job_status.json",
            'job_metadata': f"{base_path}/job_metadata.json",
            
            # Logs
            'logs_dir': f"{base_path}/logs",
            'main_log': f"{base_path}/logs/automl_main.log",
            'spark_log': f"{base_path}/logs/spark_driver.log",
            'error_log': f"{base_path}/logs/error.log",
            'dataproc_log': f"{base_path}/logs/dataproc_batch.log",
            
            # Data processing outputs
            'data_profile': f"{base_path}/data_analysis/data_profile.json",
            'feature_profiling': f"{base_path}/data_analysis/feature_profiling.json",
            'variable_tracking': f"{base_path}/data_analysis/variable_tracking_report.xlsx",
            'data_quality': f"{base_path}/data_analysis/data_quality_report.json",
            
            # Model outputs
            'models_dir': f"{base_path}/models",
            'model_comparison': f"{base_path}/models/model_comparison.json",
            'best_model': f"{base_path}/models/best_model",
            'model_metrics': f"{base_path}/models/model_metrics.json",
            'feature_importance': f"{base_path}/models/feature_importance.xlsx",
            
            # Results and reports
            'results_dir': f"{base_path}/results",
            'final_results': f"{base_path}/results/final_results.json",
            'performance_report': f"{base_path}/results/performance_report.json",
            'validation_results': f"{base_path}/results/validation_results.json",
            
            # Visualizations
            'plots_dir': f"{base_path}/plots",
            'feature_importance_plot': f"{base_path}/plots/feature_importance.png",
            'model_performance_plot': f"{base_path}/plots/model_performance.png",
            'confusion_matrix': f"{base_path}/plots/confusion_matrix.png",
            
            # Scoring artifacts
            'scoring_dir': f"{base_path}/scoring",
            'scoring_code': f"{base_path}/scoring/scoring_code.py",
            'model_pipeline': f"{base_path}/scoring/model_pipeline.pkl",
            
            # Raw outputs (for debugging)
            'raw_outputs': f"{base_path}/raw_outputs",
            'spark_history': f"{base_path}/raw_outputs/spark_history",
        }
        
        # Add GCS URLs for easy access
        gcs_paths = {}
        for key, path in paths.items():
            gcs_paths[f"{key}_gcs_url"] = f"gs://{self.bucket_name}/{path}"
        
        paths.update(gcs_paths)
        return paths

api_handler = AutoMLAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'AutoML API',
        'version': '1.0.0'
    })

@app.route('/api/v1/jobs', methods=['POST'])
def submit_job():
    """
    Submit a new AutoML job.
    
    Expected payload:
    {
        "job_id": "unique_job_id",
        "data_source": "bigquery_table_or_gcs_path",
        "target_column": "target_column_name",
        "task_type": "classification|regression",
        "model_params": {...},
        "advanced_params": {...}
    }
    """
    try:
        job_data = request.get_json()
        
        if not job_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['job_id', 'data_source', 'target_column', 'task_type']
        missing_fields = [field for field in required_fields if field not in job_data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        job_id = job_data['job_id']
        
        # Get standardized paths
        paths = api_handler.get_standardized_paths(job_id)
        
        # Create job metadata
        job_metadata = {
            'job_id': job_id,
            'submission_time': datetime.now().isoformat(),
            'status': 'submitted',
            'task_type': job_data['task_type'],
            'data_source': job_data['data_source'],
            'target_column': job_data['target_column'],
            'paths': paths,
            'estimated_duration': '15-45 minutes',
            'progress': {
                'current_step': 'job_submitted',
                'completed_steps': [],
                'total_steps': [
                    'data_loading',
                    'data_preprocessing', 
                    'feature_engineering',
                    'model_training',
                    'model_validation',
                    'results_generation'
                ]
            }
        }
        
        # Submit job to Dataproc (using existing manager)
        try:
            from dataproc_serverless_manager import DataprocServerlessManager
            
            dataproc_manager = DataprocServerlessManager()
            
            # Enhanced job config with standardized paths
            enhanced_job_config = {
                **job_data,
                'output_paths': paths,
                'api_mode': True,
                'structured_output': True
            }
            
            batch_id = dataproc_manager.submit_spark_job(
                job_config=enhanced_job_config,
                job_id=job_id,
                data_files=None,
                dependencies=None
            )
            
            if batch_id:
                job_metadata['batch_id'] = batch_id
                job_metadata['status'] = 'running'
                job_metadata['dataproc_console_url'] = f"https://console.cloud.google.com/dataproc/batches/{api_handler.region}/{batch_id}?project={api_handler.project_id}"
                
                # Save job metadata to GCS
                if api_handler.bucket:
                    try:
                        blob = api_handler.bucket.blob(paths['job_metadata'])
                        blob.upload_from_string(json.dumps(job_metadata, indent=2))
                        logger.info(f"✅ Job metadata saved to {paths['job_metadata_gcs_url']}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save job metadata: {e}")
                
                return jsonify({
                    'success': True,
                    'job_id': job_id,
                    'batch_id': batch_id,
                    'status': 'running',
                    'message': 'Job submitted successfully',
                    'paths': paths,
                    'monitoring': {
                        'status_endpoint': f"/api/v1/jobs/{job_id}/status",
                        'logs_endpoint': f"/api/v1/jobs/{job_id}/logs",
                        'results_endpoint': f"/api/v1/jobs/{job_id}/results"
                    },
                    'estimated_completion': '15-45 minutes'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to submit job to Dataproc'
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error submitting to Dataproc: {e}")
            return jsonify({
                'success': False,
                'error': f'Dataproc submission failed: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"❌ Error in submit_job: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/v1/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id: str):
    """Get job status and progress."""
    try:
        paths = api_handler.get_standardized_paths(job_id)
        
        # Try to get job metadata from GCS
        job_metadata = None
        if api_handler.bucket:
            try:
                blob = api_handler.bucket.blob(paths['job_metadata'])
                if blob.exists():
                    job_metadata = json.loads(blob.download_as_text())
            except Exception as e:
                logger.warning(f"⚠️ Could not load job metadata: {e}")
        
        # Get Dataproc batch status if we have batch_id
        dataproc_status = None
        if job_metadata and 'batch_id' in job_metadata:
            try:
                from dataproc_serverless_manager import DataprocServerlessManager
                dataproc_manager = DataprocServerlessManager()
                dataproc_status = dataproc_manager.get_job_status(job_metadata['batch_id'])
            except Exception as e:
                logger.warning(f"⚠️ Could not get Dataproc status: {e}")
        
        # Check for results to determine completion
        results_available = False
        if api_handler.bucket:
            try:
                results_blob = api_handler.bucket.blob(paths['final_results'])
                results_available = results_blob.exists()
            except:
                pass
        
        # Determine overall status
        if results_available:
            overall_status = 'completed'
        elif dataproc_status == 'FAILED':
            overall_status = 'failed'
        elif dataproc_status in ['RUNNING', 'PENDING']:
            overall_status = 'running'
        elif dataproc_status == 'SUCCEEDED' and not results_available:
            overall_status = 'processing_results'
        else:
            overall_status = 'unknown'
        
        response = {
            'job_id': job_id,
            'status': overall_status,
            'dataproc_status': dataproc_status,
            'results_available': results_available,
            'timestamp': datetime.now().isoformat(),
            'paths': paths
        }
        
        if job_metadata:
            response.update({
                'submission_time': job_metadata.get('submission_time'),
                'task_type': job_metadata.get('task_type'),
                'progress': job_metadata.get('progress', {}),
                'batch_id': job_metadata.get('batch_id'),
                'dataproc_console_url': job_metadata.get('dataproc_console_url')
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting job status: {e}")
        return jsonify({
            'error': f'Failed to get job status: {str(e)}'
        }), 500

@app.route('/api/v1/jobs/<job_id>/logs', methods=['GET'])
def get_job_logs(job_id: str):
    """Get job logs."""
    try:
        paths = api_handler.get_standardized_paths(job_id)
        
        logs = {}
        log_files = {
            'main': paths['main_log'],
            'spark': paths['spark_log'], 
            'error': paths['error_log'],
            'dataproc': paths['dataproc_log']
        }
        
        if api_handler.bucket:
            for log_type, log_path in log_files.items():
                try:
                    blob = api_handler.bucket.blob(log_path)
                    if blob.exists():
                        # Get last 1000 lines for web display
                        log_content = blob.download_as_text()
                        log_lines = log_content.split('\n')
                        logs[log_type] = {
                            'content': '\n'.join(log_lines[-1000:]) if len(log_lines) > 1000 else log_content,
                            'total_lines': len(log_lines),
                            'truncated': len(log_lines) > 1000,
                            'gcs_url': f"gs://{api_handler.bucket_name}/{log_path}"
                        }
                    else:
                        logs[log_type] = {
                            'content': 'Log file not yet available',
                            'total_lines': 0,
                            'truncated': False,
                            'gcs_url': f"gs://{api_handler.bucket_name}/{log_path}"
                        }
                except Exception as e:
                    logs[log_type] = {
                        'content': f'Error reading log: {str(e)}',
                        'error': True
                    }
        
        return jsonify({
            'job_id': job_id,
            'logs': logs,
            'timestamp': datetime.now().isoformat(),
            'log_locations': {
                'gcs_base_path': f"gs://{api_handler.bucket_name}/{paths['logs_dir']}",
                'individual_logs': {k: f"gs://{api_handler.bucket_name}/{v}" for k, v in log_files.items()}
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting job logs: {e}")
        return jsonify({
            'error': f'Failed to get job logs: {str(e)}'
        }), 500

@app.route('/api/v1/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id: str):
    """Get job results and outputs."""
    try:
        paths = api_handler.get_standardized_paths(job_id)
        
        results = {}
        result_files = {
            'final_results': paths['final_results'],
            'model_comparison': paths['model_comparison'],
            'model_metrics': paths['model_metrics'],
            'performance_report': paths['performance_report'],
            'data_profile': paths['data_profile'],
            'feature_profiling': paths['feature_profiling']
        }
        
        if api_handler.bucket:
            for result_type, result_path in result_files.items():
                try:
                    blob = api_handler.bucket.blob(result_path)
                    if blob.exists():
                        content = blob.download_as_text()
                        try:
                            results[result_type] = json.loads(content)
                        except json.JSONDecodeError:
                            results[result_type] = {'raw_content': content}
                        results[f"{result_type}_gcs_url"] = f"gs://{api_handler.bucket_name}/{result_path}"
                    else:
                        results[result_type] = None
                except Exception as e:
                    results[result_type] = {'error': str(e)}
        
        # Get available artifacts
        artifacts = {}
        artifact_paths = {
            'variable_tracking': paths['variable_tracking'],
            'feature_importance': paths['feature_importance'],
            'feature_importance_plot': paths['feature_importance_plot'],
            'model_performance_plot': paths['model_performance_plot'],
            'scoring_code': paths['scoring_code']
        }
        
        if api_handler.bucket:
            for artifact_type, artifact_path in artifact_paths.items():
                try:
                    blob = api_handler.bucket.blob(artifact_path)
                    artifacts[artifact_type] = {
                        'available': blob.exists(),
                        'gcs_url': f"gs://{api_handler.bucket_name}/{artifact_path}",
                        'size_bytes': blob.size if blob.exists() else 0
                    }
                except:
                    artifacts[artifact_type] = {'available': False}
        
        return jsonify({
            'job_id': job_id,
            'results': results,
            'artifacts': artifacts,
            'download_urls': {
                'all_results': f"/api/v1/jobs/{job_id}/download/results",
                'logs': f"/api/v1/jobs/{job_id}/download/logs",
                'models': f"/api/v1/jobs/{job_id}/download/models"
            },
            'gcs_locations': {
                'base_path': f"gs://{api_handler.bucket_name}/automl_jobs/{job_id}",
                'results_dir': paths['results_dir_gcs_url'],
                'models_dir': paths['models_dir_gcs_url'],
                'logs_dir': paths['logs_dir_gcs_url']
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting job results: {e}")
        return jsonify({
            'error': f'Failed to get job results: {str(e)}'
        }), 500

@app.route('/api/v1/jobs', methods=['GET'])
def list_jobs():
    """List all jobs with their status."""
    try:
        jobs = []
        
        if api_handler.bucket:
            # List all job directories
            blobs = api_handler.bucket.list_blobs(prefix='automl_jobs/', delimiter='/')
            
            for prefix in blobs.prefixes:
                job_id = prefix.split('/')[-2]  # Extract job_id from path
                
                # Try to get job metadata
                try:
                    metadata_blob = api_handler.bucket.blob(f"{prefix}job_metadata.json")
                    if metadata_blob.exists():
                        metadata = json.loads(metadata_blob.download_as_text())
                        
                        # Check if results are available
                        results_blob = api_handler.bucket.blob(f"{prefix}results/final_results.json")
                        results_available = results_blob.exists()
                        
                        jobs.append({
                            'job_id': job_id,
                            'submission_time': metadata.get('submission_time'),
                            'task_type': metadata.get('task_type'),
                            'status': 'completed' if results_available else metadata.get('status', 'unknown'),
                            'data_source': metadata.get('data_source'),
                            'target_column': metadata.get('target_column'),
                            'results_available': results_available
                        })
                except:
                    # If metadata is not available, just list the job
                    jobs.append({
                        'job_id': job_id,
                        'status': 'unknown',
                        'results_available': False
                    })
        
        return jsonify({
            'jobs': sorted(jobs, key=lambda x: x.get('submission_time', ''), reverse=True),
            'total_jobs': len(jobs),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error listing jobs: {e}")
        return jsonify({
            'error': f'Failed to list jobs: {str(e)}'
        }), 500

@app.route('/api/v1/paths/<job_id>', methods=['GET'])
def get_job_paths(job_id: str):
    """Get all standardized paths for a job."""
    try:
        paths = api_handler.get_standardized_paths(job_id)
        
        return jsonify({
            'job_id': job_id,
            'paths': paths,
            'bucket': api_handler.bucket_name,
            'project': api_handler.project_id,
            'region': api_handler.region,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get job paths: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
