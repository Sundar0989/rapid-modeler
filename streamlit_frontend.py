#!/usr/bin/env python3
"""
Standalone Streamlit Frontend for AutoML.
Decoupled from the backend processing, connects via API calls.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from typing import Dict, Any, Optional, List

# Configure Streamlit page
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.environ.get('AUTOML_API_URL', 'http://localhost:8080')
REFRESH_INTERVAL = 10  # seconds

class AutoMLFrontend:
    """Frontend handler for AutoML operations."""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        
    def make_api_call(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict:
        """Make API call with error handling."""
        try:
            url = f"{self.api_base}{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {'error': str(e)}
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return {'error': str(e)}
    
    def submit_job(self, job_config: Dict) -> Dict:
        """Submit a new AutoML job."""
        return self.make_api_call('/api/v1/jobs', method='POST', data=job_config)
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status."""
        return self.make_api_call(f'/api/v1/jobs/{job_id}/status')
    
    def get_job_logs(self, job_id: str) -> Dict:
        """Get job logs."""
        return self.make_api_call(f'/api/v1/jobs/{job_id}/logs')
    
    def get_job_results(self, job_id: str) -> Dict:
        """Get job results."""
        return self.make_api_call(f'/api/v1/jobs/{job_id}/results')
    
    def list_jobs(self) -> Dict:
        """List all jobs."""
        return self.make_api_call('/api/v1/jobs')
    
    def get_job_paths(self, job_id: str) -> Dict:
        """Get all paths for a job."""
        return self.make_api_call(f'/api/v1/paths/{job_id}')

# Initialize frontend
frontend = AutoMLFrontend()

def main():
    """Main Streamlit application."""
    
    st.title("🤖 AutoML Platform")
    st.markdown("**Intelligent Machine Learning with Real-time Monitoring**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🚀 Submit Job", "📊 Monitor Jobs", "📈 Results & Analysis", "📁 Job Explorer", "⚙️ Settings"]
    )
    
    if page == "🚀 Submit Job":
        show_submit_job_page()
    elif page == "📊 Monitor Jobs":
        show_monitor_jobs_page()
    elif page == "📈 Results & Analysis":
        show_results_page()
    elif page == "📁 Job Explorer":
        show_job_explorer_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_submit_job_page():
    """Job submission page."""
    st.header("🚀 Submit New AutoML Job")
    
    with st.form("job_submission_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Configuration")
            
            job_id = st.text_input(
                "Job ID",
                value=f"automl_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Unique identifier for this job"
            )
            
            data_source = st.text_input(
                "Data Source",
                placeholder="project.dataset.table or gs://bucket/path/file.csv",
                help="BigQuery table or GCS file path"
            )
            
            target_column = st.text_input(
                "Target Column",
                placeholder="target",
                help="Name of the column to predict"
            )
            
            task_type = st.selectbox(
                "Task Type",
                ["classification", "regression"],
                help="Type of machine learning task"
            )
            
            where_clause = st.text_area(
                "WHERE Clause (Optional)",
                placeholder="date_column >= '2024-01-01'",
                help="SQL WHERE clause to filter data"
            )
        
        with col2:
            st.subheader("🤖 Model Configuration")
            
            st.markdown("**Select Models to Train:**")
            run_logistic = st.checkbox("Logistic Regression", value=True)
            run_rf = st.checkbox("Random Forest", value=True)
            run_xgboost = st.checkbox("XGBoost", value=True)
            run_lightgbm = st.checkbox("LightGBM", value=True)
            run_gb = st.checkbox("Gradient Boosting", value=False)
            run_dt = st.checkbox("Decision Tree", value=False)
            run_nn = st.checkbox("Neural Network", value=False)
            
            st.markdown("**Advanced Options:**")
            enable_hp_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            enable_intelligent_sampling = st.checkbox("Enable Intelligent Sampling", value=True)
            
            max_features = st.slider("Max Features", 5, 100, 30)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        submitted = st.form_submit_button("🚀 Submit Job", type="primary")
        
        if submitted:
            if not all([job_id, data_source, target_column]):
                st.error("Please fill in all required fields!")
                return
            
            # Prepare job configuration
            job_config = {
                "job_id": job_id,
                "data_source": data_source,
                "target_column": target_column,
                "task_type": task_type,
                "model_params": {
                    "run_logistic_regression": run_logistic,
                    "run_random_forest": run_rf,
                    "run_xgboost": run_xgboost,
                    "run_lightgbm": run_lightgbm,
                    "run_gradient_boosting": run_gb,
                    "run_decision_tree": run_dt,
                    "run_neural_network": run_nn
                },
                "advanced_params": {
                    "enable_hyperparameter_tuning": enable_hp_tuning,
                    "max_features": max_features,
                    "cv_folds": cv_folds
                },
                "enable_intelligent_sampling": enable_intelligent_sampling,
                "enhanced_data_config": {
                    "source_type": "bigquery" if "." in data_source and not data_source.startswith("gs://") else "gcs",
                    "data_source": data_source,
                    "options": {
                        "where_clause": where_clause if where_clause else None
                    }
                }
            }
            
            with st.spinner("Submitting job..."):
                result = frontend.submit_job(job_config)
            
            if result.get('success'):
                st.success(f"✅ Job submitted successfully!")
                st.info(f"**Job ID:** {result['job_id']}")
                st.info(f"**Batch ID:** {result['batch_id']}")
                st.info(f"**Estimated Completion:** {result['estimated_completion']}")
                
                # Show monitoring links
                st.markdown("### 📊 Monitoring Links")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Monitor Status"):
                        st.session_state.monitoring_job_id = job_id
                        st.experimental_rerun()
                
                with col2:
                    if st.button("📋 View Logs"):
                        st.session_state.logs_job_id = job_id
                        st.experimental_rerun()
                
                with col3:
                    if st.button("📁 Explore Paths"):
                        st.session_state.explorer_job_id = job_id
                        st.experimental_rerun()
                
                # Show all output paths
                with st.expander("📁 Output Locations"):
                    paths = result.get('paths', {})
                    for category in ['logs', 'data_analysis', 'models', 'results', 'plots']:
                        category_paths = {k: v for k, v in paths.items() if category in k and not k.endswith('_gcs_url')}
                        if category_paths:
                            st.markdown(f"**{category.title()}:**")
                            for key, path in category_paths.items():
                                gcs_url = paths.get(f"{key}_gcs_url", f"gs://rapid_modeler_app/{path}")
                                st.code(gcs_url, language="bash")
            else:
                st.error(f"❌ Job submission failed: {result.get('error', 'Unknown error')}")

def show_monitor_jobs_page():
    """Job monitoring page."""
    st.header("📊 Monitor AutoML Jobs")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=False)
    with col2:
        if st.button("🔄 Refresh Now"):
            st.experimental_rerun()
    with col3:
        show_completed = st.checkbox("Show Completed", value=True)
    
    # Get all jobs
    with st.spinner("Loading jobs..."):
        jobs_response = frontend.list_jobs()
    
    if 'error' in jobs_response:
        st.error(f"Failed to load jobs: {jobs_response['error']}")
        return
    
    jobs = jobs_response.get('jobs', [])
    
    if not jobs:
        st.info("No jobs found. Submit a new job to get started!")
        return
    
    # Filter jobs
    if not show_completed:
        jobs = [job for job in jobs if job.get('status') != 'completed']
    
    # Display jobs
    for job in jobs:
        job_id = job['job_id']
        status = job.get('status', 'unknown')
        
        # Status color coding
        if status == 'completed':
            status_color = "🟢"
        elif status == 'running':
            status_color = "🟡"
        elif status == 'failed':
            status_color = "🔴"
        else:
            status_color = "⚪"
        
        with st.expander(f"{status_color} {job_id} - {status.upper()}", expanded=(status == 'running')):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Job ID:** {job_id}")
                st.markdown(f"**Status:** {status}")
                st.markdown(f"**Task Type:** {job.get('task_type', 'Unknown')}")
                st.markdown(f"**Data Source:** {job.get('data_source', 'Unknown')}")
                st.markdown(f"**Target:** {job.get('target_column', 'Unknown')}")
                if job.get('submission_time'):
                    st.markdown(f"**Submitted:** {job['submission_time']}")
            
            with col2:
                # Action buttons
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    if st.button(f"📊 Status", key=f"status_{job_id}"):
                        show_job_status_details(job_id)
                
                with col2b:
                    if st.button(f"📋 Logs", key=f"logs_{job_id}"):
                        show_job_logs_details(job_id)
                
                with col2c:
                    if st.button(f"📈 Results", key=f"results_{job_id}") and status == 'completed':
                        show_job_results_details(job_id)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.experimental_rerun()

def show_job_status_details(job_id: str):
    """Show detailed job status."""
    st.subheader(f"📊 Job Status: {job_id}")
    
    with st.spinner("Loading status..."):
        status_response = frontend.get_job_status(job_id)
    
    if 'error' in status_response:
        st.error(f"Failed to get status: {status_response['error']}")
        return
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Status", status_response.get('status', 'Unknown'))
    
    with col2:
        st.metric("Dataproc Status", status_response.get('dataproc_status', 'Unknown'))
    
    with col3:
        st.metric("Results Available", "Yes" if status_response.get('results_available') else "No")
    
    with col4:
        if status_response.get('submission_time'):
            submission_time = datetime.fromisoformat(status_response['submission_time'].replace('Z', '+00:00'))
            elapsed = datetime.now() - submission_time.replace(tzinfo=None)
            st.metric("Elapsed Time", f"{elapsed.total_seconds()/60:.1f} min")
    
    # Progress information
    if 'progress' in status_response:
        progress = status_response['progress']
        st.markdown("### 📈 Progress")
        
        if 'progress_percentage' in progress and progress['progress_percentage']:
            st.progress(progress['progress_percentage'] / 100)
        
        if progress.get('current_step'):
            st.info(f"**Current Step:** {progress['current_step']}")
        
        if progress.get('completed_steps'):
            st.success(f"**Completed Steps:** {', '.join(progress['completed_steps'])}")
    
    # Console links
    if status_response.get('dataproc_console_url'):
        st.markdown(f"[🔗 View in Dataproc Console]({status_response['dataproc_console_url']})")

def show_job_logs_details(job_id: str):
    """Show detailed job logs."""
    st.subheader(f"📋 Job Logs: {job_id}")
    
    with st.spinner("Loading logs..."):
        logs_response = frontend.get_job_logs(job_id)
    
    if 'error' in logs_response:
        st.error(f"Failed to get logs: {logs_response['error']}")
        return
    
    logs = logs_response.get('logs', {})
    
    # Log tabs
    log_types = ['main', 'spark', 'error', 'dataproc']
    available_logs = [log_type for log_type in log_types if log_type in logs]
    
    if not available_logs:
        st.info("No logs available yet. Job may still be starting up.")
        return
    
    tabs = st.tabs([f"📋 {log_type.title()}" for log_type in available_logs])
    
    for i, log_type in enumerate(available_logs):
        with tabs[i]:
            log_data = logs[log_type]
            
            if log_data.get('error'):
                st.error(f"Error reading {log_type} log: {log_data.get('content', 'Unknown error')}")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if log_data.get('truncated'):
                        st.warning(f"Showing last 1000 lines of {log_data.get('total_lines', 0)} total lines")
                
                with col2:
                    if log_data.get('gcs_url'):
                        st.markdown(f"[📁 Full Log]({log_data['gcs_url']})")
                
                # Display log content
                log_content = log_data.get('content', 'No content available')
                st.code(log_content, language="text")

def show_job_results_details(job_id: str):
    """Show detailed job results."""
    st.subheader(f"📈 Job Results: {job_id}")
    
    with st.spinner("Loading results..."):
        results_response = frontend.get_job_results(job_id)
    
    if 'error' in results_response:
        st.error(f"Failed to get results: {results_response['error']}")
        return
    
    results = results_response.get('results', {})
    artifacts = results_response.get('artifacts', {})
    
    # Results tabs
    tabs = st.tabs(["📊 Summary", "🤖 Models", "📈 Performance", "📁 Artifacts"])
    
    with tabs[0]:  # Summary
        if results.get('final_results'):
            final_results = results['final_results'].get('data', {})
            
            # Key metrics
            if 'best_model' in final_results:
                best_model = final_results['best_model']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best Model", best_model.get('model_name', 'Unknown'))
                
                with col2:
                    st.metric("Best Score", f"{best_model.get('score', 0):.4f}")
                
                with col3:
                    st.metric("Metric", best_model.get('metric', 'Unknown'))
        
        # Feature importance
        if results.get('feature_profiling'):
            st.markdown("### 🎯 Top Features")
            # Display feature importance if available
    
    with tabs[1]:  # Models
        if results.get('model_comparison'):
            model_comparison = results['model_comparison'].get('data', {})
            
            if 'model_scores' in model_comparison:
                # Create comparison chart
                model_scores = model_comparison['model_scores']
                
                df = pd.DataFrame([
                    {'Model': model, 'Score': score}
                    for model, score in model_scores.items()
                ])
                
                fig = px.bar(df, x='Model', y='Score', title='Model Performance Comparison')
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Performance
        # Show validation results for different datasets
        for dataset in ['train', 'oot1', 'oot2']:
            validation_key = f'validation_results_{dataset}'
            if results.get(validation_key):
                st.markdown(f"### 📊 {dataset.upper()} Performance")
                validation_data = results[validation_key].get('data', {})
                # Display metrics
    
    with tabs[3]:  # Artifacts
        st.markdown("### 📁 Available Artifacts")
        
        for artifact_type, artifact_info in artifacts.items():
            if artifact_info.get('available'):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{artifact_type.replace('_', ' ').title()}**")
                
                with col2:
                    size_mb = artifact_info.get('size_bytes', 0) / (1024 * 1024)
                    st.text(f"{size_mb:.2f} MB")
                
                with col3:
                    if artifact_info.get('gcs_url'):
                        st.markdown(f"[📥 Download]({artifact_info['gcs_url']})")

def show_results_page():
    """Results and analysis page."""
    st.header("📈 Results & Analysis")
    
    # Job selector
    with st.spinner("Loading completed jobs..."):
        jobs_response = frontend.list_jobs()
    
    if 'error' in jobs_response:
        st.error(f"Failed to load jobs: {jobs_response['error']}")
        return
    
    completed_jobs = [job for job in jobs_response.get('jobs', []) if job.get('status') == 'completed']
    
    if not completed_jobs:
        st.info("No completed jobs found. Submit and complete a job to see results here.")
        return
    
    selected_job = st.selectbox(
        "Select a completed job",
        options=[job['job_id'] for job in completed_jobs],
        format_func=lambda x: f"{x} ({next(job['task_type'] for job in completed_jobs if job['job_id'] == x)})"
    )
    
    if selected_job:
        show_job_results_details(selected_job)

def show_job_explorer_page():
    """Job explorer page showing all paths and outputs."""
    st.header("📁 Job Explorer")
    
    job_id = st.text_input("Enter Job ID", placeholder="automl_job_20240923_120000")
    
    if job_id:
        with st.spinner("Loading job paths..."):
            paths_response = frontend.get_job_paths(job_id)
        
        if 'error' in paths_response:
            st.error(f"Failed to get paths: {paths_response['error']}")
            return
        
        paths = paths_response.get('paths', {})
        
        # Organize paths by category
        categories = {
            'Job Metadata': ['job_config', 'job_status', 'job_metadata', 'job_progress'],
            'Logs': [k for k in paths.keys() if 'log' in k and not k.endswith('_gcs_url')],
            'Data Analysis': [k for k in paths.keys() if 'data_analysis' in k or 'profiling' in k or 'tracking' in k and not k.endswith('_gcs_url')],
            'Models': [k for k in paths.keys() if 'model' in k and not k.endswith('_gcs_url')],
            'Results': [k for k in paths.keys() if 'result' in k and not k.endswith('_gcs_url')],
            'Plots': [k for k in paths.keys() if 'plot' in k or 'png' in k and not k.endswith('_gcs_url')],
            'Scoring': [k for k in paths.keys() if 'scoring' in k and not k.endswith('_gcs_url')]
        }
        
        for category, path_keys in categories.items():
            if path_keys:
                with st.expander(f"📁 {category}"):
                    for key in path_keys:
                        if key in paths:
                            gcs_url = paths.get(f"{key}_gcs_url", f"gs://rapid_modeler_app/{paths[key]}")
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.code(gcs_url, language="bash")
                            
                            with col2:
                                st.markdown(f"[📋 Copy](javascript:navigator.clipboard.writeText('{gcs_url}'))")

def show_settings_page():
    """Settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("🔗 API Configuration")
    current_api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("🧪 Test Connection"):
        try:
            response = requests.get(f"{current_api_url}/health", timeout=10)
            if response.status_code == 200:
                st.success("✅ API connection successful!")
                health_data = response.json()
                st.json(health_data)
            else:
                st.error(f"❌ API returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.subheader("📊 Display Settings")
    st.slider("Auto-refresh Interval (seconds)", 5, 60, REFRESH_INTERVAL)
    st.checkbox("Show Debug Information", value=False)
    st.checkbox("Enable Notifications", value=True)
    
    st.subheader("📁 Default Paths")
    st.info("All outputs are automatically organized in standardized GCS paths:")
    st.code("gs://rapid_modeler_app/automl_jobs/{job_id}/", language="bash")

if __name__ == "__main__":
    main()
