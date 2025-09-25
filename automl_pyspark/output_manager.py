#!/usr/bin/env python3
"""
Output Manager for standardized file locations and logging.
Ensures all AutoML outputs are in predictable locations for Streamlit UI consumption.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from google.cloud import storage
import tempfile

class OutputManager:
    """
    Manages all AutoML outputs with standardized locations and formats.
    """
    
    def __init__(self, job_id: str, bucket_name: str = None, project_id: str = None):
        self.job_id = job_id
        self.bucket_name = bucket_name or os.environ.get('GCP_RESULTS_BUCKET', 'rapid_modeler_app')
        self.project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'atus-prism-dev')
        
        # Initialize GCS client
        try:
            self.storage_client = storage.Client(project=self.project_id)
            self.bucket = self.storage_client.bucket(self.bucket_name)
        except Exception as e:
            logging.warning(f"⚠️ Could not initialize GCS client: {e}")
            self.storage_client = None
            self.bucket = None
        
        # Get all standardized paths
        self.paths = self._generate_paths()
        
        # Setup logging
        self._setup_logging()
    
    def _generate_paths(self) -> Dict[str, str]:
        """Generate all standardized paths for the job."""
        base_path = f"automl_jobs/{self.job_id}"
        
        return {
            # Job metadata and configuration
            'job_config': f"{base_path}/job_config.json",
            'job_status': f"{base_path}/job_status.json",
            'job_metadata': f"{base_path}/job_metadata.json",
            'job_progress': f"{base_path}/job_progress.json",
            
            # Logs (structured for easy parsing)
            'logs_dir': f"{base_path}/logs",
            'main_log': f"{base_path}/logs/automl_main.log",
            'spark_log': f"{base_path}/logs/spark_driver.log",
            'error_log': f"{base_path}/logs/error.log",
            'dataproc_log': f"{base_path}/logs/dataproc_batch.log",
            'step_logs': f"{base_path}/logs/steps",  # Individual step logs
            
            # Data processing outputs
            'data_analysis_dir': f"{base_path}/data_analysis",
            'data_profile': f"{base_path}/data_analysis/data_profile.json",
            'feature_profiling_train': f"{base_path}/data_analysis/feature_profiling_train.json",
            'feature_profiling_oot1': f"{base_path}/data_analysis/feature_profiling_oot1.json",
            'feature_profiling_oot2': f"{base_path}/data_analysis/feature_profiling_oot2.json",
            'feature_profiling_combined': f"{base_path}/data_analysis/feature_profiling_combined.json",
            'variable_tracking_classification': f"{base_path}/data_analysis/classification_variable_tracking_report.xlsx",
            'variable_tracking_regression': f"{base_path}/data_analysis/regression_variable_tracking_report.xlsx",
            'data_quality': f"{base_path}/data_analysis/data_quality_report.json",
            'cardinality_analysis': f"{base_path}/data_analysis/cardinality_analysis.json",
            
            # Model outputs
            'models_dir': f"{base_path}/models",
            'model_comparison': f"{base_path}/models/model_comparison.json",
            'best_model_dir': f"{base_path}/models/best_model",
            'model_metrics': f"{base_path}/models/model_metrics.json",
            'feature_importance': f"{base_path}/models/feature_importance.xlsx",
            'hyperparameter_results': f"{base_path}/models/hyperparameter_results.json",
            'cross_validation_results': f"{base_path}/models/cross_validation_results.json",
            
            # Individual model outputs
            'logistic_regression_model': f"{base_path}/models/logistic_regression",
            'random_forest_model': f"{base_path}/models/random_forest",
            'xgboost_model': f"{base_path}/models/xgboost",
            'lightgbm_model': f"{base_path}/models/lightgbm",
            'gradient_boosting_model': f"{base_path}/models/gradient_boosting",
            'decision_tree_model': f"{base_path}/models/decision_tree",
            'neural_network_model': f"{base_path}/models/neural_network",
            
            # Results and reports
            'results_dir': f"{base_path}/results",
            'final_results': f"{base_path}/results/final_results.json",
            'performance_report': f"{base_path}/results/performance_report.json",
            'validation_results_train': f"{base_path}/results/validation_results_train.json",
            'validation_results_oot1': f"{base_path}/results/validation_results_oot1.json",
            'validation_results_oot2': f"{base_path}/results/validation_results_oot2.json",
            'model_comparison_report': f"{base_path}/results/model_comparison_report.json",
            'business_metrics': f"{base_path}/results/business_metrics.json",
            
            # Visualizations
            'plots_dir': f"{base_path}/plots",
            'feature_importance_plot': f"{base_path}/plots/feature_importance.png",
            'model_performance_plot': f"{base_path}/plots/model_performance.png",
            'confusion_matrix_train': f"{base_path}/plots/confusion_matrix_train.png",
            'confusion_matrix_oot1': f"{base_path}/plots/confusion_matrix_oot1.png",
            'confusion_matrix_oot2': f"{base_path}/plots/confusion_matrix_oot2.png",
            'roc_curve_train': f"{base_path}/plots/roc_curve_train.png",
            'roc_curve_oot1': f"{base_path}/plots/roc_curve_oot1.png",
            'roc_curve_oot2': f"{base_path}/plots/roc_curve_oot2.png",
            'feature_distribution': f"{base_path}/plots/feature_distribution.png",
            'correlation_heatmap': f"{base_path}/plots/correlation_heatmap.png",
            
            # Scoring artifacts
            'scoring_dir': f"{base_path}/scoring",
            'scoring_code': f"{base_path}/scoring/scoring_code.py",
            'model_pipeline': f"{base_path}/scoring/model_pipeline.pkl",
            'preprocessing_pipeline': f"{base_path}/scoring/preprocessing_pipeline.pkl",
            'feature_transformer': f"{base_path}/scoring/feature_transformer.pkl",
            
            # Raw outputs and debugging
            'raw_outputs': f"{base_path}/raw_outputs",
            'spark_history': f"{base_path}/raw_outputs/spark_history",
            'temp_tables': f"{base_path}/raw_outputs/temp_tables.json",
            'debug_info': f"{base_path}/raw_outputs/debug_info.json",
        }
    
    def _setup_logging(self):
        """Setup structured logging for the job."""
        # Configure main logger
        self.logger = logging.getLogger(f"automl_job_{self.job_id}")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"🚀 OutputManager initialized for job: {self.job_id}")
        self.logger.info(f"📁 Base path: gs://{self.bucket_name}/automl_jobs/{self.job_id}")
    
    def get_gcs_url(self, path_key: str) -> str:
        """Get GCS URL for a given path key."""
        if path_key in self.paths:
            return f"gs://{self.bucket_name}/{self.paths[path_key]}"
        else:
            raise ValueError(f"Unknown path key: {path_key}")
    
    def save_json(self, path_key: str, data: Dict[Any, Any], pretty: bool = True) -> bool:
        """
        Save JSON data to GCS with standardized formatting.
        
        Args:
            path_key: Key from self.paths
            data: Data to save
            pretty: Whether to format JSON nicely
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if path_key not in self.paths:
                raise ValueError(f"Unknown path key: {path_key}")
            
            # Add metadata to all JSON files
            enhanced_data = {
                'job_id': self.job_id,
                'timestamp': datetime.now().isoformat(),
                'file_type': path_key,
                'data': data
            }
            
            json_str = json.dumps(enhanced_data, indent=2 if pretty else None, default=str)
            
            if self.bucket:
                blob = self.bucket.blob(self.paths[path_key])
                blob.upload_from_string(json_str, content_type='application/json')
                self.logger.info(f"✅ Saved {path_key} to {self.get_gcs_url(path_key)}")
                return True
            else:
                self.logger.warning(f"⚠️ GCS not available, could not save {path_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save {path_key}: {e}")
            return False
    
    def save_text(self, path_key: str, content: str, content_type: str = 'text/plain') -> bool:
        """Save text content to GCS."""
        try:
            if path_key not in self.paths:
                raise ValueError(f"Unknown path key: {path_key}")
            
            if self.bucket:
                blob = self.bucket.blob(self.paths[path_key])
                blob.upload_from_string(content, content_type=content_type)
                self.logger.info(f"✅ Saved {path_key} to {self.get_gcs_url(path_key)}")
                return True
            else:
                self.logger.warning(f"⚠️ GCS not available, could not save {path_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save {path_key}: {e}")
            return False
    
    def save_file(self, path_key: str, local_file_path: str, content_type: str = None) -> bool:
        """Upload a local file to GCS."""
        try:
            if path_key not in self.paths:
                raise ValueError(f"Unknown path key: {path_key}")
            
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")
            
            if self.bucket:
                blob = self.bucket.blob(self.paths[path_key])
                
                if content_type:
                    blob.upload_from_filename(local_file_path, content_type=content_type)
                else:
                    blob.upload_from_filename(local_file_path)
                
                self.logger.info(f"✅ Uploaded {local_file_path} to {self.get_gcs_url(path_key)}")
                return True
            else:
                self.logger.warning(f"⚠️ GCS not available, could not upload {path_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to upload {path_key}: {e}")
            return False
    
    def update_progress(self, current_step: str, completed_steps: List[str], 
                       progress_percentage: float = None, message: str = None) -> bool:
        """Update job progress for real-time monitoring."""
        try:
            progress_data = {
                'current_step': current_step,
                'completed_steps': completed_steps,
                'progress_percentage': progress_percentage,
                'message': message,
                'last_updated': datetime.now().isoformat()
            }
            
            return self.save_json('job_progress', progress_data)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update progress: {e}")
            return False
    
    def log_step(self, step_name: str, message: str, level: str = 'INFO', 
                 data: Dict[Any, Any] = None) -> bool:
        """Log a processing step with structured data."""
        try:
            # Log to main logger
            getattr(self.logger, level.lower())(f"[{step_name}] {message}")
            
            # Save structured step log
            step_log = {
                'step_name': step_name,
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'data': data or {}
            }
            
            step_log_path = f"{self.paths['step_logs']}/{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            if self.bucket:
                blob = self.bucket.blob(step_log_path)
                blob.upload_from_string(json.dumps(step_log, indent=2, default=str))
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step {step_name}: {e}")
            return False
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all paths with GCS URLs."""
        gcs_paths = {}
        for key, path in self.paths.items():
            gcs_paths[key] = path
            gcs_paths[f"{key}_gcs_url"] = f"gs://{self.bucket_name}/{path}"
        
        return gcs_paths
    
    def create_summary_manifest(self) -> bool:
        """Create a summary manifest of all outputs for easy discovery."""
        try:
            manifest = {
                'job_id': self.job_id,
                'bucket': self.bucket_name,
                'project': self.project_id,
                'base_path': f"automl_jobs/{self.job_id}",
                'created': datetime.now().isoformat(),
                'paths': self.get_all_paths(),
                'key_outputs': {
                    'final_results': self.get_gcs_url('final_results'),
                    'model_comparison': self.get_gcs_url('model_comparison'),
                    'feature_profiling': self.get_gcs_url('feature_profiling_combined'),
                    'variable_tracking': self.get_gcs_url('variable_tracking_classification'),
                    'main_log': self.get_gcs_url('main_log'),
                    'best_model': self.get_gcs_url('best_model_dir'),
                    'scoring_code': self.get_gcs_url('scoring_code')
                },
                'streamlit_endpoints': {
                    'status': f"/api/v1/jobs/{self.job_id}/status",
                    'logs': f"/api/v1/jobs/{self.job_id}/logs",
                    'results': f"/api/v1/jobs/{self.job_id}/results"
                }
            }
            
            return self.save_json('job_metadata', manifest)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create summary manifest: {e}")
            return False

def get_output_manager(job_id: str) -> OutputManager:
    """Factory function to create OutputManager instance."""
    return OutputManager(job_id)

# Example usage and testing
if __name__ == "__main__":
    # Test the OutputManager
    test_job_id = f"test_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_manager = get_output_manager(test_job_id)
    
    print("📁 All standardized paths:")
    for key, path in output_manager.paths.items():
        print(f"  {key}: gs://{output_manager.bucket_name}/{path}")
    
    # Test saving some data
    test_data = {
        'test': True,
        'timestamp': datetime.now().isoformat(),
        'message': 'This is a test of the OutputManager'
    }
    
    success = output_manager.save_json('job_config', test_data)
    print(f"\n✅ Test save successful: {success}")
    
    # Test progress update
    output_manager.update_progress(
        current_step='testing',
        completed_steps=['initialization'],
        progress_percentage=25.0,
        message='Testing OutputManager functionality'
    )
    
    # Create manifest
    output_manager.create_summary_manifest()
    
    print(f"\n🎯 Test job outputs available at:")
    print(f"   gs://{output_manager.bucket_name}/automl_jobs/{test_job_id}/")
