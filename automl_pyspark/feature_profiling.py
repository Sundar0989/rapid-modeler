"""
Feature Profiling Module for AutoML PySpark

This module provides comprehensive feature profiling capabilities including:
1. Univariate analysis for each selected feature
2. Bivariate analysis (each feature vs target variable)
3. Statistical summaries and distributions
4. Streamlit dashboard for interactive visualization

Only works on selected features (post feature selection) for efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.functions import percentile_approx, corr, when, isnan, isnull
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class FeatureProfiler:
    """
    Comprehensive feature profiling for selected features in AutoML pipeline.
    
    Performs univariate and bivariate analysis on features that passed feature selection,
    providing insights into feature distributions and relationships with target variable.
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize Feature Profiler.
        
        Args:
            spark_session: Active Spark session
        """
        self.spark = spark_session
        self.profiling_results = {}
        
    def profile_features(self, 
                        data: DataFrame, 
                        selected_features: List[str], 
                        target_column: str,
                        task_type: str = "classification",
                        existing_stats: Optional[Dict[str, Any]] = None,
                        categorical_vars: Optional[List[str]] = None,
                        numerical_vars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature profiling on selected features.
        
        Args:
            data: Input DataFrame with all data
            selected_features: List of selected feature names (post feature selection)
            target_column: Name of target variable
            task_type: Either "classification" or "regression"
            existing_stats: Optional dict with pre-computed statistics from feature selection
            categorical_vars: List of categorical variable names (from initial analysis)
            numerical_vars: List of numerical variable names (from initial analysis)
            
        Returns:
            Dictionary containing all profiling results
        """
        print(f"🔍 Starting feature profiling for {len(selected_features)} selected features...")
        
        # Ensure target column is included in analysis
        analysis_columns = selected_features + [target_column]
        analysis_data = data.select(analysis_columns)
        
        results = {
            'task_type': task_type,
            'target_column': target_column,
            'selected_features': selected_features,
            'univariate_analysis': {},
            'bivariate_analysis': {},
            'correlation_matrix': {},
            'feature_importance_insights': {},
            'stability_summary': {}
        }
        
        # 1. Combined Univariate Analysis + Data Quality Assessment (single pass)
        print("📊 Performing combined univariate analysis and data quality assessment...")
        if existing_stats and 'cardinality_analysis' in existing_stats:
            print("♻️ Reusing cardinality and missing value statistics...")
            combined_results = self._univariate_analysis_optimized(
                analysis_data, selected_features, target_column, existing_stats['cardinality_analysis']
            )
            results['univariate_analysis'] = combined_results['features']
            results['stability_summary'] = combined_results['stability_summary']
        else:
            raise ValueError("❌ Feature profiling requires existing cardinality statistics. Call analyze_all_features_cardinality() first.")
        
        # 2. Bivariate Analysis (Feature vs Target Distribution)
        print("📈 Performing bivariate analysis...")
        results['bivariate_analysis'] = self._bivariate_analysis(
            analysis_data, selected_features, target_column, task_type, categorical_vars, numerical_vars
        )
        
        # 3. Correlation Analysis
        print("🔗 Computing correlation matrix...")
        results['correlation_matrix'] = self._correlation_analysis(analysis_data, selected_features, target_column)
        
        # Store results for dashboard
        self.profiling_results = results
        
        print(f"✅ Feature profiling completed for {len(selected_features)} features")
        return results
    
    def enhance_with_oot_psi(self, train_data: DataFrame, oot1_data: DataFrame = None, 
                            oot2_data: DataFrame = None, selected_features: List[str] = None) -> Dict[str, Any]:
        """
        Enhance existing profiling results with actual PSI calculations using OOT data.
        
        Args:
            train_data: Training dataset
            oot1_data: First out-of-time dataset (optional)
            oot2_data: Second out-of-time dataset (optional)
            selected_features: List of features to calculate PSI for
            
        Returns:
            Enhanced profiling results with actual PSI scores
        """
        if not hasattr(self, 'profiling_results') or not self.profiling_results:
            print("⚠️ No existing profiling results found. Run profile_features() first.")
            return {}
        
        if not selected_features:
            selected_features = list(self.profiling_results.get('univariate_analysis', {}).get('features', {}).keys())
        
        print(f"🔄 Enhancing profiling with PSI calculations for {len(selected_features)} features...")
        
        # Determine feature types from existing results
        schema_dict = {field.name: field.dataType.typeName() for field in train_data.schema.fields}
        numerical_features = [f for f in selected_features if schema_dict.get(f, 'string') in ['integer', 'long', 'float', 'double', 'decimal']]
        categorical_features = [f for f in selected_features if f not in numerical_features]
        
        psi_results = {
            'train_vs_oot1': {},
            'train_vs_oot2': {},
            'oot1_vs_oot2': {},
            'overall_stability_assessment': {}
        }
        
        # Calculate PSI between datasets
        for feature in selected_features:
            is_numerical = feature in numerical_features
            feature_psi = {
                'feature': feature,
                'is_numerical': is_numerical,
                'psi_scores': {},
                'stability_status': 'stable',
                'distribution_shift': 'none'
            }
            
            # Train vs OOT1
            if oot1_data is not None:
                psi_score = self._calculate_psi_between_datasets(train_data, oot1_data, feature, is_numerical)
                feature_psi['psi_scores']['train_vs_oot1'] = psi_score
                psi_results['train_vs_oot1'][feature] = psi_score
            
            # Train vs OOT2
            if oot2_data is not None:
                psi_score = self._calculate_psi_between_datasets(train_data, oot2_data, feature, is_numerical)
                feature_psi['psi_scores']['train_vs_oot2'] = psi_score
                psi_results['train_vs_oot2'][feature] = psi_score
            
            # OOT1 vs OOT2
            if oot1_data is not None and oot2_data is not None:
                psi_score = self._calculate_psi_between_datasets(oot1_data, oot2_data, feature, is_numerical)
                feature_psi['psi_scores']['oot1_vs_oot2'] = psi_score
                psi_results['oot1_vs_oot2'][feature] = psi_score
            
            # Determine overall stability status based on PSI scores
            max_psi = max(feature_psi['psi_scores'].values()) if feature_psi['psi_scores'] else 0.0
            
            if max_psi < 0.1:
                feature_psi['stability_status'] = 'stable'
                feature_psi['distribution_shift'] = 'none'
            elif max_psi < 0.2:
                feature_psi['stability_status'] = 'moderately_stable'
                feature_psi['distribution_shift'] = 'minimal'
            elif max_psi < 0.25:
                feature_psi['stability_status'] = 'unstable'
                feature_psi['distribution_shift'] = 'moderate'
            else:
                feature_psi['stability_status'] = 'very_unstable'
                feature_psi['distribution_shift'] = 'significant'
            
            psi_results['overall_stability_assessment'][feature] = feature_psi
            
            # Update existing profiling results with actual PSI
            if 'univariate_analysis' in self.profiling_results and 'features' in self.profiling_results['univariate_analysis']:
                if feature in self.profiling_results['univariate_analysis']['features']:
                    self.profiling_results['univariate_analysis']['features'][feature]['psi_assessment'].update({
                        'psi_scores': feature_psi['psi_scores'],
                        'stability_status': feature_psi['stability_status'],
                        'distribution_shift': feature_psi['distribution_shift'],
                        'max_psi_score': max_psi
                    })
        
        # Calculate overall dataset stability
        all_psi_scores = []
        for comparison in ['train_vs_oot1', 'train_vs_oot2', 'oot1_vs_oot2']:
            if psi_results[comparison]:
                all_psi_scores.extend(psi_results[comparison].values())
        
        if all_psi_scores:
            avg_psi = np.mean(all_psi_scores)
            max_psi = max(all_psi_scores)
            
            psi_results['dataset_stability_summary'] = {
                'average_psi': avg_psi,
                'maximum_psi': max_psi,
                'total_features_analyzed': len(selected_features),
                'stable_features': len([f for f in psi_results['overall_stability_assessment'].values() if f['stability_status'] == 'stable']),
                'unstable_features': len([f for f in psi_results['overall_stability_assessment'].values() if f['stability_status'] in ['unstable', 'very_unstable']])
            }
            
            if avg_psi < 0.1:
                psi_results['dataset_stability_summary']['overall_stability'] = 'stable'
            elif avg_psi < 0.2:
                psi_results['dataset_stability_summary']['overall_stability'] = 'moderately_stable'
            else:
                psi_results['dataset_stability_summary']['overall_stability'] = 'unstable'
        
        # Add PSI results to existing profiling results
        self.profiling_results['psi_analysis'] = psi_results
        
        print(f"✅ PSI analysis completed")
        if all_psi_scores:
            print(f"   📊 Average PSI: {avg_psi:.4f}")
            print(f"   📊 Maximum PSI: {max_psi:.4f}")
            print(f"   📊 Overall stability: {psi_results['dataset_stability_summary']['overall_stability']}")
            print(f"   ✅ Stable features: {psi_results['dataset_stability_summary']['stable_features']}")
            print(f"   ⚠️ Unstable features: {psi_results['dataset_stability_summary']['unstable_features']}")
        
        return psi_results
    
    
    def _univariate_analysis_optimized(self, data: DataFrame, features: List[str], target_column: str, 
                                      existing_feature_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Highly optimized vectorized univariate analysis and data quality assessment.
        
        Eliminates feature-by-feature loops and uses batch DataFrame operations for maximum efficiency.
        
        Args:
            data: Input DataFrame
            features: List of feature names (all selected features)
            target_column: Target variable name
            existing_feature_stats: Pre-computed stats from analyze_all_features_cardinality()
            
        Returns:
            Dictionary with combined univariate analysis and data quality results
        """
        print(f"   🚀 Performing vectorized analysis for {len(features)} features...")
        
        total_rows = data.count()
        schema_dict = {field.name: field.dataType.typeName() for field in data.schema.fields}
        
        # Separate numerical and categorical features
        numerical_features = [f for f in features if schema_dict.get(f, 'string') in ['integer', 'long', 'float', 'double', 'decimal']]
        categorical_features = [f for f in features if f not in numerical_features]
        
        print(f"   📊 Processing {len(numerical_features)} numerical and {len(categorical_features)} categorical features...")
        
        # ========================================================================
        # VECTORIZED NUMERICAL STATISTICS (single DataFrame operation)
        # ========================================================================
        numerical_stats = {}
        if numerical_features:
            print(f"   🧮 Computing all numerical statistics in single operation...")
            
            # Build comprehensive aggregation for all numerical features at once
            agg_exprs = []
            for feature in numerical_features:
                agg_exprs.extend([
                    count(feature).alias(f'{feature}_count'),
                    mean(feature).alias(f'{feature}_mean'),
                    stddev(feature).alias(f'{feature}_std'),
                    spark_min(feature).alias(f'{feature}_min'),
                    spark_max(feature).alias(f'{feature}_max'),
                    percentile_approx(feature, 0.25).alias(f'{feature}_q25'),
                    percentile_approx(feature, 0.5).alias(f'{feature}_median'),
                    percentile_approx(feature, 0.75).alias(f'{feature}_q75')
                ])
            
            # Single DataFrame operation for all numerical features
            batch_stats = data.agg(*agg_exprs).collect()[0]
            
            # Parse results into feature-specific dictionaries
            for feature in numerical_features:
                numerical_stats[feature] = {
                    'count': batch_stats[f'{feature}_count'],
                    'mean': float(batch_stats[f'{feature}_mean']) if batch_stats[f'{feature}_mean'] is not None else None,
                    'std': float(batch_stats[f'{feature}_std']) if batch_stats[f'{feature}_std'] is not None else None,
                    'min': float(batch_stats[f'{feature}_min']) if batch_stats[f'{feature}_min'] is not None else None,
                    'max': float(batch_stats[f'{feature}_max']) if batch_stats[f'{feature}_max'] is not None else None,
                    'q25': float(batch_stats[f'{feature}_q25']) if batch_stats[f'{feature}_q25'] is not None else None,
                    'median': float(batch_stats[f'{feature}_median']) if batch_stats[f'{feature}_median'] is not None else None,
                    'q75': float(batch_stats[f'{feature}_q75']) if batch_stats[f'{feature}_q75'] is not None else None
                }
        
        # ========================================================================
        # VECTORIZED CATEGORICAL STATISTICS (optimized batch processing)
        # ========================================================================
        categorical_stats = {}
        if categorical_features:
            print(f"   📊 Computing categorical statistics for {len(categorical_features)} features...")
            
            # Batch count operation for all categorical features
            count_exprs = [count(feature).alias(f'{feature}_count') for feature in categorical_features]
            if count_exprs:
                batch_counts = data.agg(*count_exprs).collect()[0]
                
                for feature in categorical_features:
                    try:
                        # Get top categories (this could be further optimized with window functions if needed)
                        top_categories = data.groupBy(feature).count().orderBy(col('count').desc()).limit(10).collect()
                        category_counts = {row[feature]: row['count'] for row in top_categories}
                        
                        categorical_stats[feature] = {
                            'count': batch_counts[f'{feature}_count'],
                            'top_categories': category_counts
                        }
                    except:
                        categorical_stats[feature] = {
                            'count': 0,
                            'top_categories': {}
                        }
        
        # ========================================================================
        # VECTORIZED QUALITY ASSESSMENT AND RESULT ASSEMBLY
        # ========================================================================
        print(f"   🔍 Performing vectorized quality assessment...")
        
        univariate_results = {}
        quality_summary = {
            'total_rows': total_rows,
            'features_analyzed': len(features),
            'quality_issues': [],
            'feature_quality_summary': {}
        }
        
        # Process all features in batch (minimal loop for result assembly)
        for feature in features:
            # Get pre-computed statistics (reused)
            feature_stats = existing_feature_stats.get(feature, {})
            unique_count = feature_stats.get('unique_count', 0)
            missing_count = feature_stats.get('missing_count', 0)
            missing_percentage = feature_stats.get('missing_percentage', 0)
            data_type = schema_dict.get(feature, 'string')
            is_numerical = feature in numerical_features
            
            # Calculate unique count if not available or is 0 (fallback logic)
            if unique_count == 0:
                print(f"     🔍 Calculating missing unique count for {feature}...")
                try:
                    unique_count = data.select(feature).distinct().count()
                    print(f"     ✅ {feature}: {unique_count} unique values calculated")
                except Exception as e:
                    print(f"     ⚠️ Failed to calculate unique count for {feature}: {str(e)}")
                    unique_count = 0
            
            # Base statistics
            base_stats = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'unique_count': unique_count,
                'data_type': data_type
            }
            
            # Add feature-specific stats
            if is_numerical and feature in numerical_stats:
                base_stats.update(numerical_stats[feature])
            elif not is_numerical and feature in categorical_stats:
                base_stats.update(categorical_stats[feature])
                # Add None values for numerical stats to maintain consistency
                base_stats.update({
                    'mean': None, 'std': None, 'min': None, 'max': None,
                    'q25': None, 'median': None, 'q75': None
                })
            
            # Population Stability Index assessment
            psi_assessment = self._calculate_population_stability_index(
                feature, unique_count, missing_percentage, total_rows, 
                numerical_stats.get(feature, {}).get('std') if is_numerical else None
            )
            
            # Store results
            base_stats['psi_assessment'] = psi_assessment
            univariate_results[feature] = base_stats
            quality_summary['feature_quality_summary'][feature] = psi_assessment
            
            # Collect stability warnings
            if psi_assessment['stability_warnings']:
                for warning in psi_assessment['stability_warnings']:
                    quality_summary['quality_issues'].append({**warning, 'feature': feature})
            
            # Determine feature type based on data type and unique count
            data_type = base_stats.get('data_type', 'string').lower()
            if data_type in ['integer', 'long', 'float', 'double', 'decimal']:
                # Numerical features
                if unique_count > 0 and unique_count <= 10:
                    base_stats['feature_type'] = 'discrete'
                else:
                    base_stats['feature_type'] = 'continuous'
            else:
                # String/categorical features
                base_stats['feature_type'] = 'categorical'
        
        # Finalize overall stability summary
        if quality_summary['feature_quality_summary']:
            # Calculate overall stability based on preliminary grades
            grade_scores = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 50}
            avg_stability_score = np.mean([
                grade_scores.get(fq.get('preliminary_stability_grade', 'F'), 50) 
                for fq in quality_summary['feature_quality_summary'].values()
            ])
            quality_summary['overall_stability_score'] = avg_stability_score
            
            if avg_stability_score >= 90:
                quality_summary['overall_stability_grade'] = 'A'
            elif avg_stability_score >= 80:
                quality_summary['overall_stability_grade'] = 'B'
            elif avg_stability_score >= 70:
                quality_summary['overall_stability_grade'] = 'C'
            elif avg_stability_score >= 60:
                quality_summary['overall_stability_grade'] = 'D'
            else:
                quality_summary['overall_stability_grade'] = 'F'
        else:
            quality_summary['overall_stability_score'] = 0.0
            quality_summary['overall_stability_grade'] = 'F'
        
        print(f"✅ Combined univariate analysis and Population Stability Index assessment completed")
        print(f"   ♻️ Reused cardinality and missing values for ALL {len(features)} features")
        print(f"   📊 Overall stability grade: {quality_summary['overall_stability_grade']} ({quality_summary['overall_stability_score']:.1f}/100)")
        print(f"   ⚠️ Stability warnings found: {len(quality_summary['quality_issues'])}")
        
        return {
            'features': univariate_results,
            'stability_summary': quality_summary
        }
    
    def _calculate_population_stability_index(self, feature: str, unique_count: int, missing_percentage: float, 
                                             total_rows: int, std_val: float = None) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) for feature stability assessment.
        
        PSI measures the shift in the distribution of a variable between two samples.
        This is a placeholder implementation that will be enhanced when OOT data is available.
        
        Args:
            feature: Feature name
            unique_count: Number of unique values
            missing_percentage: Percentage of missing values
            total_rows: Total number of rows
            std_val: Standard deviation (for numerical features)
            
        Returns:
            Dictionary with PSI assessment results
        """
        psi_assessment = {
            'feature': feature,
            'psi_score': 0.0,  # Will be calculated when comparing with OOT data
            'stability_status': 'baseline',  # baseline, stable, unstable, very_unstable
            'distribution_shift': 'none',  # none, minimal, moderate, significant
            'stability_metrics': {
                'unique_count': unique_count,
                'missing_percentage': missing_percentage,
                'total_rows': total_rows,
                'cardinality_ratio': unique_count / total_rows if total_rows > 0 else 0
            }
        }
        
        # Add numerical-specific metrics
        if std_val is not None:
            psi_assessment['stability_metrics']['std_dev'] = float(std_val)
            psi_assessment['stability_metrics']['coefficient_of_variation'] = float(std_val) / 1.0 if std_val > 0 else 0
        
        # Baseline stability assessment (without OOT comparison)
        # This provides initial feature characteristics for PSI calculation
        
        # Flag potential stability issues based on feature characteristics
        stability_warnings = []
        
        # High missing values can indicate data collection issues
        if missing_percentage > 20:
            stability_warnings.append({
                'warning': 'high_missing_values',
                'severity': 'high' if missing_percentage > 50 else 'moderate',
                'value': missing_percentage,
                'description': f"High missing values ({missing_percentage:.1f}%) may indicate data collection instability"
            })
        
        # Constant features will have PSI = 0 but are problematic
        if unique_count == 1:
            stability_warnings.append({
                'warning': 'constant_feature',
                'severity': 'critical',
                'value': unique_count,
                'description': "Constant feature - PSI cannot be calculated meaningfully"
            })
        
        # Very high cardinality features may be unstable
        cardinality_ratio = unique_count / total_rows if total_rows > 0 else 0
        if cardinality_ratio > 0.95 and unique_count > 100:
            stability_warnings.append({
                'warning': 'high_cardinality',
                'severity': 'moderate',
                'value': unique_count,
                'description': f"Very high cardinality ({unique_count} values) may lead to unstable distributions"
            })
        
        # Low variance numerical features may be near-constant
        if std_val is not None and std_val < 0.01:
            stability_warnings.append({
                'warning': 'low_variance',
                'severity': 'moderate',
                'value': float(std_val),
                'description': f"Low variance (std={std_val:.6f}) may indicate near-constant behavior"
            })
        
        psi_assessment['stability_warnings'] = stability_warnings
        
        # Assign preliminary stability grade based on feature characteristics
        if len(stability_warnings) == 0:
            psi_assessment['preliminary_stability_grade'] = 'A'
        elif any(w['severity'] == 'critical' for w in stability_warnings):
            psi_assessment['preliminary_stability_grade'] = 'F'
        elif any(w['severity'] == 'high' for w in stability_warnings):
            psi_assessment['preliminary_stability_grade'] = 'C'
        else:
            psi_assessment['preliminary_stability_grade'] = 'B'
        
        return psi_assessment
    
    def _calculate_psi_between_datasets(self, train_data: DataFrame, oot_data: DataFrame, 
                                       feature: str, is_numerical: bool = True, bins: int = 10) -> float:
        """
        Calculate actual PSI between training and OOT datasets for a specific feature.
        
        PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        
        Args:
            train_data: Training dataset (expected distribution)
            oot_data: OOT dataset (actual distribution)
            feature: Feature name to calculate PSI for
            is_numerical: Whether the feature is numerical (for binning strategy)
            bins: Number of bins for numerical features
            
        Returns:
            PSI score (float)
        """
        try:
            from pyspark.sql.functions import col, count, when, isnan, isnull
            from pyspark.ml.feature import Bucketizer
            import numpy as np
            
            # Remove null values for PSI calculation
            train_clean = train_data.filter(col(feature).isNotNull() & ~isnan(col(feature)))
            oot_clean = oot_data.filter(col(feature).isNotNull() & ~isnan(col(feature)))
            
            if is_numerical:
                # For numerical features, create quantile-based bins from training data
                quantiles = train_clean.approxQuantile(feature, [i/bins for i in range(bins+1)], 0.01)
                
                # Ensure unique splits (handle constant values)
                unique_quantiles = sorted(list(set(quantiles)))
                if len(unique_quantiles) < 2:
                    return 0.0  # Cannot calculate PSI for constant features
                
                # Create bucketizer
                bucketizer = Bucketizer(splits=unique_quantiles, inputCol=feature, outputCol=f"{feature}_bucket")
                
                # Apply binning to both datasets
                train_binned = bucketizer.transform(train_clean)
                oot_binned = bucketizer.transform(oot_clean)
                
                # Calculate distributions
                train_dist = train_binned.groupBy(f"{feature}_bucket").count().collect()
                oot_dist = oot_binned.groupBy(f"{feature}_bucket").count().collect()
                
            else:
                # For categorical features, use actual categories
                train_dist = train_clean.groupBy(feature).count().collect()
                oot_dist = oot_clean.groupBy(feature).count().collect()
            
            # Convert to dictionaries for easier manipulation
            train_counts = {row[0]: row[1] for row in train_dist}
            oot_counts = {row[0]: row[1] for row in oot_dist}
            
            # Get all unique categories/bins
            all_categories = set(train_counts.keys()) | set(oot_counts.keys())
            
            # Calculate total counts
            train_total = sum(train_counts.values())
            oot_total = sum(oot_counts.values())
            
            if train_total == 0 or oot_total == 0:
                return 0.0
            
            # Calculate PSI
            psi = 0.0
            for category in all_categories:
                # Expected percentage (from training)
                expected_pct = train_counts.get(category, 0) / train_total
                # Actual percentage (from OOT)
                actual_pct = oot_counts.get(category, 0) / oot_total
                
                # Handle zero percentages (add small epsilon to avoid log(0))
                if expected_pct == 0:
                    expected_pct = 0.0001
                if actual_pct == 0:
                    actual_pct = 0.0001
                
                # PSI formula: (Actual% - Expected%) * ln(Actual% / Expected%)
                psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            
            return float(psi)
            
        except Exception as e:
            print(f"⚠️ Error calculating PSI for {feature}: {e}")
            return 0.0
    
    def _bivariate_analysis(self, data: DataFrame, features: List[str], target_column: str, task_type: str,
                           categorical_vars: Optional[List[str]] = None, numerical_vars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform bivariate analysis showing target distribution by feature bins.
        
        Uses existing categorical/numerical classification from initial feature analysis.
        For numerical features: Create bins and show target distribution within each bin
        For categorical features: Show target distribution by category (top 5 + grouped rest)
        """
        bivariate_results = {}
        
        # Use existing feature type classification if available
        categorical_features = categorical_vars or []
        numerical_features = numerical_vars or []
        
        for feature in features:
            print(f"   📈 Analyzing target distribution by {feature}...")
            
            # Use pre-determined feature type
            if feature in categorical_features:
                print(f"     🏷️ Processing as categorical feature...")
                bivariate_results[feature] = self._analyze_categorical_target_distribution(
                    data, feature, target_column, task_type
                )
            elif feature in numerical_features:
                print(f"     📊 Processing as numerical feature...")
                bivariate_results[feature] = self._analyze_numerical_target_distribution(
                    data, feature, target_column, task_type
                )
            else:
                # Fallback: determine based on unique count if not in either list
                print(f"     🔍 Feature type unknown, determining from unique count...")
                unique_count = data.select(feature).distinct().count()
                if unique_count <= 20:
                    bivariate_results[feature] = self._analyze_categorical_target_distribution(
                        data, feature, target_column, task_type
                    )
                else:
                    bivariate_results[feature] = self._analyze_numerical_target_distribution(
                        data, feature, target_column, task_type
                    )
        
        return bivariate_results
    
    def _analyze_categorical_target_distribution(self, data: DataFrame, feature: str, target_column: str, task_type: str) -> Dict[str, Any]:
        """Analyze target distribution by categorical feature (top 5 categories + grouped rest)."""
        
        # Get category counts to identify top 5
        category_counts = data.groupBy(feature).count().orderBy(col('count').desc()).collect()
        
        # Get top 5 categories
        top_5_categories = [row[feature] for row in category_counts[:5]]
        
        if task_type == "classification":
            # For each top category, get target class distribution
            distributions = {}
            
            for category in top_5_categories:
                category_data = data.filter(col(feature) == category)
                target_dist = category_data.groupBy(target_column).count().collect()
                total_in_category = category_data.count()
                
                distributions[str(category)] = {
                    'total_count': total_in_category,
                    'target_distribution': {
                        str(row[target_column]): {
                            'count': row['count'],
                            'percentage': (row['count'] / total_in_category * 100) if total_in_category > 0 else 0
                        } for row in target_dist
                    }
                }
            
            # Group remaining categories
            if len(category_counts) > 5:
                other_categories = [row[feature] for row in category_counts[5:]]
                other_data = data.filter(col(feature).isin(other_categories))
                other_target_dist = other_data.groupBy(target_column).count().collect()
                total_other = other_data.count()
                
                distributions['OTHER_CATEGORIES'] = {
                    'total_count': total_other,
                    'category_count': len(other_categories),
                    'target_distribution': {
                        str(row[target_column]): {
                            'count': row['count'],
                            'percentage': (row['count'] / total_other * 100) if total_other > 0 else 0
                        } for row in other_target_dist
                    }
                }
            
            return {
                'feature_type': 'categorical',
                'analysis_type': 'target_distribution_by_category',
                'distributions': distributions
            }
        
        else:  # regression
            # For regression: mean target value by category
            distributions = {}
            
            for category in top_5_categories:
                category_stats = data.filter(col(feature) == category).agg(
                    mean(target_column).alias('target_mean'),
                    stddev(target_column).alias('target_std'),
                    count(target_column).alias('count')
                ).collect()[0]
                
                distributions[str(category)] = {
                    'count': category_stats['count'],
                    'target_mean': float(category_stats['target_mean']) if category_stats['target_mean'] else None,
                    'target_std': float(category_stats['target_std']) if category_stats['target_std'] else None
                }
            
            # Group remaining categories
            if len(category_counts) > 5:
                other_categories = [row[feature] for row in category_counts[5:]]
                other_stats = data.filter(col(feature).isin(other_categories)).agg(
                    mean(target_column).alias('target_mean'),
                    stddev(target_column).alias('target_std'),
                    count(target_column).alias('count')
                ).collect()[0]
                
                distributions['OTHER_CATEGORIES'] = {
                    'count': other_stats['count'],
                    'category_count': len(other_categories),
                    'target_mean': float(other_stats['target_mean']) if other_stats['target_mean'] else None,
                    'target_std': float(other_stats['target_std']) if other_stats['target_std'] else None
                }
            
            return {
                'feature_type': 'categorical',
                'analysis_type': 'target_stats_by_category',
                'distributions': distributions
            }
    
    def _analyze_numerical_target_distribution(self, data: DataFrame, feature: str, target_column: str, task_type: str) -> Dict[str, Any]:
        """Analyze target distribution by numerical feature bins."""
        
        from pyspark.sql.functions import when, lit
        
        # Get feature statistics for binning
        feature_stats = data.select(
            spark_min(feature).alias('min_val'),
            spark_max(feature).alias('max_val'),
            percentile_approx(feature, 0.25).alias('q25'),
            percentile_approx(feature, 0.75).alias('q75')
        ).collect()[0]
        
        min_val = feature_stats['min_val']
        max_val = feature_stats['max_val']
        q25 = feature_stats['q25']
        q75 = feature_stats['q75']
        
        if min_val is None or max_val is None:
            return {'feature_type': 'numerical', 'analysis_type': 'insufficient_data'}
        
        # Create 5 bins based on quantiles
        bin_edges = [min_val, q25, (q25 + q75) / 2, q75, max_val]
        
        # Define bin labels with actual value ranges
        bin_labels = {
            "Bin_1_Low": f"[{min_val:.2f} - {q25:.2f}]",
            "Bin_2_Lower_Mid": f"({q25:.2f} - {(q25 + q75) / 2:.2f}]", 
            "Bin_3_Upper_Mid": f"({(q25 + q75) / 2:.2f} - {q75:.2f}]",
            "Bin_4_High": f"({q75:.2f} - {max_val:.2f}]",
            "Bin_5_Very_High": f"(>{max_val:.2f})"
        }
        
        # Create binned column
        binned_data = data.withColumn(
            f"{feature}_bin",
            when(col(feature) <= bin_edges[1], "Bin_1_Low")
            .when(col(feature) <= bin_edges[2], "Bin_2_Lower_Mid") 
            .when(col(feature) <= bin_edges[3], "Bin_3_Upper_Mid")
            .when(col(feature) <= bin_edges[4], "Bin_4_High")
            .otherwise("Bin_5_Very_High")
        )
        
        if task_type == "classification":
            # For classification: target class distribution by bin
            bin_distributions = {}
            
            bin_analysis = binned_data.groupBy(f"{feature}_bin", target_column).count().collect()
            
            # Organize by bin
            for row in bin_analysis:
                bin_name = row[f"{feature}_bin"]
                target_class = str(row[target_column])
                count = row['count']
                
                if bin_name not in bin_distributions:
                    bin_distributions[bin_name] = {'total_count': 0, 'target_distribution': {}}
                
                bin_distributions[bin_name]['total_count'] += count
                bin_distributions[bin_name]['target_distribution'][target_class] = count
            
            # Convert counts to percentages
            for bin_name, bin_data in bin_distributions.items():
                total = bin_data['total_count']
                for target_class in bin_data['target_distribution']:
                    count = bin_data['target_distribution'][target_class]
                    bin_data['target_distribution'][target_class] = {
                        'count': count,
                        'percentage': (count / total * 100) if total > 0 else 0
                    }
            
            return {
                'feature_type': 'numerical',
                'analysis_type': 'target_distribution_by_bins',
                'bin_edges': bin_edges,
                'bin_labels': bin_labels,
                'distributions': bin_distributions
            }
        
        else:  # regression
            # For regression: mean target value by bin
            bin_stats = binned_data.groupBy(f"{feature}_bin").agg(
                mean(target_column).alias('target_mean'),
                stddev(target_column).alias('target_std'),
                count(target_column).alias('count')
            ).collect()
            
            distributions = {}
            for row in bin_stats:
                bin_name = row[f"{feature}_bin"]
                distributions[bin_name] = {
                    'count': row['count'],
                    'target_mean': float(row['target_mean']) if row['target_mean'] else None,
                    'target_std': float(row['target_std']) if row['target_std'] else None
                }
            
            return {
                'feature_type': 'numerical', 
                'analysis_type': 'target_stats_by_bins',
                'bin_edges': bin_edges,
                'bin_labels': bin_labels,
                'distributions': distributions
            }
    
    def _correlation_analysis(self, data: DataFrame, features: List[str], target_column: str) -> Dict[str, Any]:
        """
        Compute comprehensive correlation analysis for selected features using optimized MLlib approach.
        
        Based on: https://github.com/Apress/applied-data-science-using-pyspark/blob/main/Ch04/customcorrelation.py
        
        Includes:
        - Feature-target correlations with strength interpretation
        - Feature-feature correlations for multicollinearity detection
        - Correlation strength categorization
        - Optimized computation using PySpark MLlib Statistics
        """
        print(f"   🔗 Computing correlations for {len(features)} features using optimized MLlib approach...")
        
        # Convert target column to numeric if it's categorical for MLlib compatibility
        try:
            # Check if target column is string/categorical and convert to numeric
            target_data_type = dict(data.dtypes)[target_column]
            if target_data_type == 'string':
                print(f"   🔄 Converting categorical target '{target_column}' to numeric for correlation analysis...")
                from pyspark.ml.feature import StringIndexer
                
                # Create StringIndexer to convert categorical target to numeric
                indexer = StringIndexer(inputCol=target_column, outputCol=f"{target_column}_indexed")
                indexed_data = indexer.fit(data).transform(data)
                
                # Use the indexed target column
                numeric_target_col = f"{target_column}_indexed"
                correlation_data = indexed_data
            else:
                print(f"   ✅ Target column '{target_column}' is already numeric")
                numeric_target_col = target_column
                correlation_data = data
            
            # Include both features and numeric target for comprehensive correlation matrix
            all_columns = features + [numeric_target_col]
            
            # Create feature vector using VectorAssembler (all numerical now)
            print(f"   📊 Assembling feature vectors for {len(all_columns)} columns...")
            assembler = VectorAssembler(inputCols=all_columns, outputCol="features")
            df_vector = assembler.transform(correlation_data)
            
            # Extract feature vectors as RDD for MLlib Statistics
            df_vector_rdd = df_vector.rdd.map(lambda x: x['features'].toArray())
            
            # Compute correlation matrix using MLlib Statistics (much faster!)
            correlation_type = 'pearson'  # Options: 'pearson', 'spearman'
            print(f"   🧮 Computing {correlation_type} correlation matrix...")
            matrix = Statistics.corr(df_vector_rdd, method=correlation_type)
            
            # Convert to pandas DataFrame for easier manipulation
            corr_df = pd.DataFrame(matrix, columns=all_columns, index=all_columns)
            
            # Extract target correlations using the numeric target column
            target_correlations = {}
            correlation_strengths = {}
            
            for feature in features:
                correlation_value = float(corr_df.loc[feature, numeric_target_col])
                target_correlations[feature] = correlation_value
                correlation_strengths[feature] = {
                    'correlation': correlation_value,
                    'abs_correlation': abs(correlation_value),
                    'strength': self._interpret_correlation(correlation_value),
                    'relationship_type': 'positive' if correlation_value > 0 else 'negative' if correlation_value < 0 else 'none'
                }
            
        except Exception as e:
            print(f"   ⚠️ MLlib correlation analysis failed: {e}")
            print(f"   🔄 Falling back to individual correlation computation...")
            
            # Fallback to original method if MLlib approach fails
            return self._correlation_analysis_fallback(data, features, target_column)
        
        # Feature-to-feature correlations (for multicollinearity detection)
        print(f"   🔍 Analyzing feature-feature correlations for multicollinearity...")
        
        # Create flattened correlation DataFrame excluding self-correlations
        final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort', ascending=False)).reset_index()
        final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_value'}, axis=1, inplace=True)
        
        # Filter for feature-feature correlations only (exclude target correlations)
        feature_corr_df = final_corr_df[
            (final_corr_df['col1'].isin(features)) & 
            (final_corr_df['col2'].isin(features))
        ]
        
        # Identify high correlations (potential multicollinearity)
        high_correlations = []
        processed_pairs = set()
        
        for _, row in feature_corr_df.iterrows():
            feat1, feat2, corr_val = row['col1'], row['col2'], row['correlation_value']
            
            # Avoid duplicate pairs (A-B and B-A)
            pair = tuple(sorted([feat1, feat2]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            if corr_val > 0.7:  # High correlation threshold
                # Get the actual signed correlation value
                signed_corr = float(corr_df.loc[feat1, feat2])
                high_correlations.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': signed_corr,
                    'abs_correlation': corr_val
                })
        
        # Sort correlations by absolute value for ranking
        sorted_target_correlations = sorted(
            correlation_strengths.items(), 
            key=lambda x: x[1]['abs_correlation'], 
            reverse=True
        )
            
        print(f"   ✅ Correlation analysis complete: {len(high_correlations)} high correlations found")
        
        return {
            'target_correlations': target_correlations,
            'correlation_strengths': correlation_strengths,
            'correlation_ranking': [
                {'feature': feat, 'correlation': data['correlation'], 'abs_correlation': data['abs_correlation']}
                for feat, data in sorted_target_correlations
            ],
            'high_feature_correlations': high_correlations,
            'multicollinearity_warning': len(high_correlations) > 0,
                'multicollinearity_count': len(high_correlations),
                'correlation_matrix': corr_df.to_dict(),  # Full correlation matrix for advanced analysis
                'method': correlation_type
            }
    
    def _correlation_analysis_fallback(self, data: DataFrame, features: List[str], target_column: str) -> Dict[str, Any]:
        """Fallback correlation analysis using individual computations."""
        target_correlations = {}
        correlation_strengths = {}
        
        # Correlation with target
        for feature in features:
            corr_val = data.select(corr(feature, target_column).alias('corr')).collect()[0]['corr']
            correlation_value = float(corr_val) if corr_val is not None else 0.0
            
            target_correlations[feature] = correlation_value
            correlation_strengths[feature] = {
                'correlation': correlation_value,
                'abs_correlation': abs(correlation_value),
                'strength': self._interpret_correlation(correlation_value),
                'relationship_type': 'positive' if correlation_value > 0 else 'negative' if correlation_value < 0 else 'none'
            }
        
        # Feature-to-feature correlations (simplified for fallback)
        high_correlations = []
        
        # Sort correlations by absolute value for ranking
        sorted_target_correlations = sorted(
            correlation_strengths.items(), 
            key=lambda x: x[1]['abs_correlation'], 
            reverse=True
        )
        
        return {
            'target_correlations': target_correlations,
            'correlation_strengths': correlation_strengths,
            'correlation_ranking': [
                {'feature': feat, 'correlation': data['correlation'], 'abs_correlation': data['abs_correlation']}
                for feat, data in sorted_target_correlations
            ],
            'high_feature_correlations': high_correlations,
            'multicollinearity_warning': False,
            'multicollinearity_count': 0,
            'method': 'fallback'
        }
    
    # ========================================================================
    # REMOVED: _data_quality_analysis method - integrated into _univariate_analysis_optimized
    # ========================================================================
    # Data quality assessment is now performed as part of the univariate analysis
    # to eliminate redundant computations and improve efficiency.
    # ========================================================================
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation is None:
            return "undefined"
        
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

if __name__ == "__main__":
    # Example usage
    print("🔍 AutoML Feature Profiling Module")
    print("This module provides comprehensive feature profiling for selected features.")
    print("Use FeatureProfiler.profile_features() to analyze your data.")
    print("Use create_streamlit_dashboard() to create interactive visualizations.")
