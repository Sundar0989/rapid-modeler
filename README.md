# 🚀 Rapid Modeler - AutoML PySpark System

A comprehensive, production-ready automated machine learning platform built on Apache Spark, designed for scalable ML workflows across local, Docker, and Google Cloud environments.

## 📋 Overview

Rapid Modeler is an enterprise-grade AutoML system that provides:

- **🤖 Automated Machine Learning** for Classification, Regression, and Clustering
- **☁️ Multi-Environment Support** (Local, Docker, Google Cloud Dataproc Serverless)
- **📊 Interactive Web Interface** built with Streamlit
- **🔄 Real-time Job Monitoring** with live logs and status updates
- **📈 Automatic Resource Scaling** based on dataset size and complexity
- **📁 Flexible Data Input** (BigQuery, File Upload, Existing Datasets)
- **📊 Comprehensive Results** with metrics, visualizations, and Excel reports

## 🚀 Quick Start

### **1. Local Development**

```bash
# Clone and setup
git clone <repository-url>
cd rapid-modeler

# Install dependencies
pip install -r automl_pyspark/requirements.txt

# Run locally
./run.sh
```

Access at: http://localhost:8080

### **2. Docker Deployment**

```bash
# Build and run with Docker
./run.sh

# Or manually:
docker build . -t rapid_modeler
docker run -p 8080:8080 rapid_modeler
```

### **3. Google Cloud Deployment**

```bash
# Setup GCP infrastructure (one-time)
./setup_buckets.sh
./setup_dataproc_serverless.sh

# Deploy to Cloud Run
./gcp-push.sh
```

## 🏗️ Architecture

### **Execution Modes**

| Mode | Environment | Use Case | Resource Management |
|------|-------------|----------|-------------------|
| **Local** | Development machine | Testing, small datasets | Single machine resources |
| **Docker** | Container (Cloud Run) | Production web app | Container resources |
| **Dataproc** | GCP Serverless Spark | Large datasets, production ML | Auto-scaling (2-100+ executors) |

### **Core Components**

- **`streamlit_automl_app.py`** - Interactive web interface
- **`background_job_manager.py`** - Job orchestration and execution
- **`dataproc_serverless_manager.py`** - Google Cloud Dataproc integration
- **`unified_job_script_generator.py`** - Cross-platform job script generation
- **`classification/`**, **`regression/`**, **`clustering/`** - ML algorithm implementations

## 📊 Features

### **🤖 AutoML Capabilities**

#### **Classification**
- **Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Output**: Model files, scoring scripts, performance reports

#### **Regression**  
- **Algorithms**: Linear Regression, Random Forest, XGBoost, LightGBM, Neural Networks
- **Metrics**: RMSE, MAE, R², Adjusted R²
- **Output**: Model files, prediction scripts, performance plots

#### **Clustering**
- **Algorithms**: K-Means, Hierarchical, DBSCAN, Gaussian Mixture
- **Metrics**: Silhouette Score, Calinski-Harabasz Index
- **Output**: Cluster assignments, centroids, visualization

### **☁️ Automatic Resource Scaling**

```python
# Scaling based on data size:
100MB dataset → 2-3 executors, 4GB memory
1GB dataset   → 3-5 executors, 8GB memory  
10GB dataset  → 6-10 executors, 16GB memory
100GB+ dataset → 12-50+ executors, 32GB memory
```

## 🛠️ Configuration

### **Environment Variables**

```bash
# Dataproc Serverless Mode
export ENABLE_DATAPROC_SERVERLESS=true
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-east1
export GCP_TEMP_BUCKET=your-temp-bucket
export GCP_RESULTS_BUCKET=your-results-bucket
```

### **Configuration Files**

- **`automl_pyspark/config.yaml`** - Main AutoML configuration
- **`automl_pyspark/dataproc_config.yaml`** - Dataproc-specific settings
- **`env.dataproc`** - Environment variables for Dataproc

## 🎯 Usage Examples

### **Web Interface**
1. Navigate to the **Job Submission** page
2. Configure your data source (BigQuery, Upload, or Existing)
3. Select ML algorithms and parameters
4. Click **🚀 Submit Job** (with ⚡ Fast Submit for speed)
5. Monitor progress with **Real-time Monitoring**
6. View results in the **Results** page

### **Programmatic Usage**

```python
from automl_pyspark import AutoMLClassifier

classifier = AutoMLClassifier(
    output_dir='./results',
    config_path='config.yaml',
    environment='production'
)

results = classifier.fit(
    train_data='data.csv',
    target_column='target',
    run_xgboost=True,
    run_random_forest=True,
    hyperparameter_tuning=True
)
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Permission Errors**: ✅ Fixed with automatic path resolution
2. **Module Import Errors**: ✅ Fixed with runtime package installation  
3. **Dataproc Job Failures**: ✅ Fixed with robust error handling

### **Debugging**

- **Enable Real-time Monitoring** in the web interface
- **Check job logs** in the Results page
- **Use debug logging** in config.yaml
- **Test locally first** before deploying to cloud

## 📞 Support

- **Issues**: Create GitHub issues for bugs or feature requests
- **Documentation**: See `DATAPROC_INTEGRATION_README.md` for cloud setup
- **Streamlit Guide**: See `automl_pyspark/STREAMLIT_README.md`

---

**🚀 Ready to accelerate your machine learning workflows with Rapid Modeler!**