# 📁 AutoML Output Locations Guide

## Overview
This document provides the complete mapping of where all AutoML outputs, logs, and results are stored. The Streamlit UI can read from these standardized locations for real-time monitoring and results display.

## 🏗️ Architecture

```
┌─────────────────┐    API Calls    ┌─────────────────┐
│   Streamlit     │◄───────────────►│   AutoML        │
│   Frontend      │                 │   Backend       │
│   (Cloud Run)   │                 │   (Dataproc)    │
└─────────────────┘                 └─────────────────┘
        │                                    │
        │                                    │
        ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│              Google Cloud Storage                   │
│        gs://rapid_modeler_app/automl_jobs/         │
└─────────────────────────────────────────────────────┘
```

## 📂 Standardized Directory Structure

For each job with ID `{job_id}`, all outputs are organized under:
```
gs://rapid_modeler_app/automl_jobs/{job_id}/
```

### 🔧 Job Metadata & Configuration
```
gs://rapid_modeler_app/automl_jobs/{job_id}/
├── job_config.json          # Original job configuration
├── job_status.json          # Current job status
├── job_metadata.json        # Complete job metadata with all paths
└── job_progress.json        # Real-time progress updates
```

### 📋 Logs (Structured for Easy Parsing)
```
gs://rapid_modeler_app/automl_jobs/{job_id}/logs/
├── automl_main.log          # Main AutoML process log
├── spark_driver.log         # Spark driver logs
├── error.log                # Error logs and stack traces
├── dataproc_batch.log       # Dataproc batch execution logs
└── steps/                   # Individual step logs
    ├── data_loading_20240923_120001.json
    ├── preprocessing_20240923_120015.json
    ├── feature_selection_20240923_120030.json
    └── model_training_20240923_120045.json
```

### 📊 Data Analysis & Profiling
```
gs://rapid_modeler_app/automl_jobs/{job_id}/data_analysis/
├── data_profile.json                              # Dataset overview and statistics
├── feature_profiling_train.json                   # Training data feature analysis
├── feature_profiling_oot1.json                    # OOT1 data feature analysis
├── feature_profiling_oot2.json                    # OOT2 data feature analysis
├── feature_profiling_combined.json                # Combined feature analysis
├── classification_variable_tracking_report.xlsx   # Variable lifecycle tracking (classification)
├── regression_variable_tracking_report.xlsx       # Variable lifecycle tracking (regression)
├── data_quality_report.json                       # Data quality assessment
└── cardinality_analysis.json                      # Categorical variable cardinality
```

### 🤖 Model Outputs
```
gs://rapid_modeler_app/automl_jobs/{job_id}/models/
├── model_comparison.json                # Side-by-side model comparison
├── model_metrics.json                   # Detailed performance metrics
├── feature_importance.xlsx              # Feature importance analysis
├── hyperparameter_results.json          # Hyperparameter tuning results
├── cross_validation_results.json        # Cross-validation performance
├── best_model/                          # Best performing model artifacts
│   ├── model.pkl
│   ├── metadata.json
│   └── performance.json
└── individual_models/                   # All trained models
    ├── logistic_regression/
    ├── random_forest/
    ├── xgboost/
    ├── lightgbm/
    ├── gradient_boosting/
    ├── decision_tree/
    └── neural_network/
```

### 📈 Results & Reports
```
gs://rapid_modeler_app/automl_jobs/{job_id}/results/
├── final_results.json                   # Complete final results summary
├── performance_report.json              # Detailed performance analysis
├── validation_results_train.json        # Training set validation
├── validation_results_oot1.json         # OOT1 validation results
├── validation_results_oot2.json         # OOT2 validation results
├── model_comparison_report.json         # Model comparison details
└── business_metrics.json                # Business-relevant metrics
```

### 📊 Visualizations
```
gs://rapid_modeler_app/automl_jobs/{job_id}/plots/
├── feature_importance.png               # Feature importance chart
├── model_performance.png                # Model performance comparison
├── confusion_matrix_train.png           # Training confusion matrix
├── confusion_matrix_oot1.png            # OOT1 confusion matrix
├── confusion_matrix_oot2.png            # OOT2 confusion matrix
├── roc_curve_train.png                  # Training ROC curve
├── roc_curve_oot1.png                   # OOT1 ROC curve
├── roc_curve_oot2.png                   # OOT2 ROC curve
├── feature_distribution.png             # Feature distribution plots
└── correlation_heatmap.png              # Feature correlation heatmap
```

### 🎯 Scoring Artifacts
```
gs://rapid_modeler_app/automl_jobs/{job_id}/scoring/
├── scoring_code.py                      # Production scoring code
├── model_pipeline.pkl                   # Complete model pipeline
├── preprocessing_pipeline.pkl           # Data preprocessing pipeline
└── feature_transformer.pkl              # Feature transformation pipeline
```

### 🔍 Raw Outputs & Debugging
```
gs://rapid_modeler_app/automl_jobs/{job_id}/raw_outputs/
├── spark_history/                       # Spark execution history
├── temp_tables.json                     # Temporary table information
└── debug_info.json                      # Debugging information
```

## 🔗 API Endpoints for Streamlit UI

### Job Management
- `POST /api/v1/jobs` - Submit new job
- `GET /api/v1/jobs` - List all jobs
- `GET /api/v1/jobs/{job_id}/status` - Get job status and progress
- `GET /api/v1/paths/{job_id}` - Get all standardized paths

### Monitoring & Logs
- `GET /api/v1/jobs/{job_id}/logs` - Get structured logs
- `GET /api/v1/jobs/{job_id}/results` - Get results and artifacts

### Example API Response for Paths:
```json
{
  "job_id": "automl_job_20240923_120000",
  "paths": {
    "main_log": "automl_jobs/automl_job_20240923_120000/logs/automl_main.log",
    "main_log_gcs_url": "gs://rapid_modeler_app/automl_jobs/automl_job_20240923_120000/logs/automl_main.log",
    "final_results": "automl_jobs/automl_job_20240923_120000/results/final_results.json",
    "final_results_gcs_url": "gs://rapid_modeler_app/automl_jobs/automl_job_20240923_120000/results/final_results.json"
  }
}
```

## 📱 Streamlit UI Integration

### Real-time Monitoring
The Streamlit UI polls these endpoints every 10 seconds:
1. **Job Status**: `GET /api/v1/jobs/{job_id}/status`
2. **Progress Updates**: Reads `job_progress.json`
3. **Live Logs**: `GET /api/v1/jobs/{job_id}/logs`

### Results Display
When jobs complete, the UI automatically loads:
1. **Final Results**: `final_results.json`
2. **Model Comparison**: `model_comparison.json`
3. **Feature Analysis**: `feature_profiling_combined.json`
4. **Variable Tracking**: Excel reports for download
5. **Visualizations**: All PNG files in plots directory

### File Access Patterns
```python
# Streamlit UI can access files via:
# 1. API endpoints (recommended)
response = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}/results")

# 2. Direct GCS access (if needed)
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('rapid_modeler_app')
blob = bucket.blob(f'automl_jobs/{job_id}/results/final_results.json')
content = json.loads(blob.download_as_text())
```

## 🎯 Key Benefits

### For Streamlit UI:
- **Predictable Paths**: All outputs follow standard naming conventions
- **Real-time Updates**: Progress and logs update continuously
- **Structured Data**: All JSON files include metadata and timestamps
- **Easy Discovery**: Single API call gets all paths for a job

### For Users:
- **Transparency**: Complete visibility into all outputs
- **Accessibility**: All files accessible via GCS console or gsutil
- **Organization**: Logical grouping by function (logs, models, results)
- **Debugging**: Comprehensive logging at every step

### For Operations:
- **Monitoring**: Easy to set up alerts on log patterns
- **Cleanup**: Simple to implement retention policies by job age
- **Backup**: Standard GCS backup and versioning applies
- **Scaling**: No filesystem dependencies, pure cloud storage

## 🚀 Usage Examples

### Submit Job and Monitor
```python
# Submit job
job_config = {
    "job_id": "my_classification_job",
    "data_source": "project.dataset.table",
    "target_column": "target",
    "task_type": "classification"
}
response = requests.post(f"{API_BASE}/api/v1/jobs", json=job_config)

# Monitor progress
job_id = response.json()['job_id']
while True:
    status = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}/status").json()
    if status['status'] == 'completed':
        break
    time.sleep(10)

# Get results
results = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}/results").json()
```

### Access Specific Outputs
```bash
# Download variable tracking report
gsutil cp gs://rapid_modeler_app/automl_jobs/{job_id}/data_analysis/classification_variable_tracking_report.xlsx .

# View logs
gsutil cat gs://rapid_modeler_app/automl_jobs/{job_id}/logs/automl_main.log

# Download best model
gsutil -m cp -r gs://rapid_modeler_app/automl_jobs/{job_id}/models/best_model/ .
```

This standardized structure ensures the Streamlit UI always knows exactly where to find logs, results, and artifacts for real-time monitoring and analysis.
