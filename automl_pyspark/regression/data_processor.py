"""
Regression Data Processor

Handles data preprocessing, feature selection, and feature engineering for regression tasks.
This is a regression-specific version that uses Random Forest Regression for feature importance.
"""

import os
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, count, variance, corr
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, RobustScaler,
    StringIndexer, OneHotEncoder, Imputer, PCA
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.stat import Correlation

# Import feature selection utilities
try:
    from ..feature_selection import ExtractFeatureImp, save_feature_importance
except ImportError:
    # For direct script execution
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from feature_selection import ExtractFeatureImp, save_feature_importance

# Import utility functions
try:
    from ..data_manipulations import (
        assembled_vectors, categorical_to_index, numerical_imputation, rename_columns,
        analyze_categorical_cardinality, remove_high_cardinality_categoricals
    )
except ImportError:
    try:
        from data_manipulations import (
            assembled_vectors, categorical_to_index, numerical_imputation, rename_columns,
            analyze_categorical_cardinality, remove_high_cardinality_categoricals
        )
    except ImportError:
        # Fallback implementation
        def assembled_vectors(data: DataFrame, feature_cols: List[str], target_column: str) -> DataFrame:
            """Create feature vector from selected columns."""
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            return assembler.transform(data)
        
        def categorical_to_index(data: DataFrame, categorical_vars: List[str]):
            """Encode categorical variables using StringIndexer."""
            if not categorical_vars:
                return data, None
            from pyspark.ml.feature import StringIndexer
            indexers = [StringIndexer(inputCol=col, outputCol=col + "_encoded", handleInvalid="keep") for col in categorical_vars]
            from pyspark.ml import Pipeline
            pipeline = Pipeline(stages=indexers)
            fitted_pipeline = pipeline.fit(data)
            return fitted_pipeline.transform(data), fitted_pipeline
        
        def numerical_imputation(data: DataFrame, numerical_vars: List[str], impute_value: float):
            """Apply numerical imputation."""
            from pyspark.sql.functions import when, isnan, isnull
            for col_name in numerical_vars:
                data = data.withColumn(col_name, when(isnan(col_name) | isnull(col_name), impute_value).otherwise(col_name))
            return data
        
        def rename_columns(data: DataFrame, categorical_vars: List[str]):
            """Rename encoded categorical columns to match expected format."""
            # Columns are already named with _encoded suffix, no renaming needed
            return data


class RegressionDataProcessor:
    """
    Data processor specifically designed for regression tasks.
    
    This class handles:
    - Data preprocessing and cleaning
    - Feature selection using Random Forest Regression
    - Data splitting and scaling
    - Feature engineering
    """
    
    def __init__(self, spark_session: SparkSession, user_id: str, model_literal: str):
        """
        Initialize the regression data processor.
        
        Args:
            spark_session: PySpark SparkSession
            user_id: User identifier for saving outputs
            model_literal: Model literal for saving outputs
        """
        self.spark = spark_session
        self.user_id = user_id
        self.model_literal = model_literal
        self.output_dir: Optional[str] = None  # Will be set by AutoMLRegressor
        
        # Data processing artifacts
        self.feature_vars = []
        self.selected_vars = []
        self.categorical_vars = []
        self.numerical_vars = []
        self.char_labels = None
        
        # Variable tracking
        self.variable_tracker = None
        self.pipeline_model = None
        
        print(f"✅ RegressionDataProcessor initialized for user: {user_id}, model: {model_literal}")
    
    def preprocess(self, data: DataFrame, target_column: str, config: Dict, dataset_info: Optional[Dict] = None) -> tuple:
        """
        Preprocess data for regression.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            dataset_info: Optional dataset information containing size info
            
        Returns:
            Tuple of (processed_data, feature_vars, selected_vars, categorical_vars, numerical_vars)
        """
        print("🔄 Starting regression data preprocessing...")
        
        # Initialize variable tracking
        from variable_tracker import VariableTracker
        self.variable_tracker = VariableTracker(task_type="regression")
        self.variable_tracker.initialize_variables(data.columns, target_column)
        
        # Check if target column is string and convert to numeric for regression
        target_data_type = dict(data.dtypes)[target_column]
        if target_data_type == 'string':
            print(f"🔄 Converting string target column '{target_column}' to numeric for regression...")
            try:
                # First, check for null/empty values and handle them
                from pyspark.sql.functions import when, isnull, length, trim, isnan
                
                # Remove null, empty, and whitespace-only values
                data = data.filter(
                    ~isnull(col(target_column)) & 
                    (length(trim(col(target_column))) > 0) &
                    (trim(col(target_column)) != '')
                )
                
                print(f"📊 Filtered out null/empty values. Remaining rows: {data.count()}")
                
                # Try to convert string to numeric using cast
                data = data.withColumn(target_column, col(target_column).cast("double"))
                
                # Check for NaN values after conversion and filter them out
                data = data.filter(~isnan(col(target_column)))
                
                print(f"📊 Filtered out NaN values. Final rows: {data.count()}")
                print(f"✅ Successfully converted '{target_column}' to numeric type")
                
            except Exception as e:
                print(f"⚠️ Failed to convert '{target_column}' to numeric: {e}")
                print("🔄 Attempting to use StringIndexer for categorical target...")
                try:
                    # Use StringIndexer as fallback for categorical targets
                    from pyspark.ml.feature import StringIndexer
                    target_indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_indexed")
                    self.target_label_indexer = target_indexer.fit(data)
                    data = self.target_label_indexer.transform(data)
                    # Replace original target column with indexed version
                    data = data.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
                    print(f"✅ Successfully encoded '{target_column}' using StringIndexer")
                except Exception as e2:
                    print(f"❌ Failed to encode target column: {e2}")
                    raise ValueError(f"Cannot convert target column '{target_column}' to numeric format for regression")
        
        # Get feature variables
        processed_data, self.feature_vars = self._get_feature_variables(data, target_column, config)
        print(f"📊 Found {len(self.feature_vars)} potential features")
        
        # Filter out date/timestamp columns automatically
        self.feature_vars = self._filter_date_columns(data, self.feature_vars)
        print(f"📊 Features after date column filtering: {len(self.feature_vars)}")
        
        # Apply preprocessing transformations
        processed_data, self.categorical_vars, self.numerical_vars = self._apply_preprocessing_transformations(
            processed_data, self.categorical_vars, self.numerical_vars, config.get('impute_value', -999)
        )
        
        # Final validation: Ensure target column has no null/NaN values for regression
        print(f"🔍 Final validation: Checking target column '{target_column}' for null/NaN values...")
        from pyspark.sql.functions import isnull, isnan
        
        # Count null/NaN values in target column
        null_count = processed_data.filter(isnull(col(target_column))).count()
        nan_count = processed_data.filter(isnan(col(target_column))).count()
        
        if null_count > 0:
            print(f"⚠️ Found {null_count} null values in target column. Filtering them out...")
            processed_data = processed_data.filter(~isnull(col(target_column)))
        
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} NaN values in target column. Filtering them out...")
            processed_data = processed_data.filter(~isnan(col(target_column)))
        
        final_count = processed_data.count()
        print(f"✅ Target column validation complete. Final dataset: {final_count} rows")
        
        if final_count == 0:
            raise ValueError(f"No valid data remaining after cleaning target column '{target_column}'")
        
        # Select features using Random Forest Regression
        # Calculate actual number of features needed (min of available and configured)
        available_features = [col for col in processed_data.columns if col != target_column]
        configured_max = config.get('max_features', 30)
        actual_max_features = min(configured_max, len(available_features))
        
        print(f"📊 Available features: {len(available_features)}, Configured max: {configured_max}")
        print(f"🎯 Will select: {actual_max_features} features")
        
        self.selected_vars = self.select_features(processed_data, target_column, actual_max_features, dataset_info)
        
        print(f"✅ Preprocessing completed. Selected {len(self.selected_vars)} features")
        
        # Generate variable tracking report
        if self.variable_tracker:
            try:
                output_dir = getattr(self, 'output_dir', '.')
                report_path = f"{output_dir}/regression_variable_tracking_report.xlsx"
                self.variable_tracker.generate_excel_report(report_path)
                self.variable_tracker.print_summary()
            except Exception as e:
                print(f"⚠️ Failed to generate variable tracking report: {e}")
        
        return processed_data, self.feature_vars, self.selected_vars, self.categorical_vars, self.numerical_vars
    
    def select_features(self, data: DataFrame, target_column: str, max_features: int = 30, dataset_info: Optional[Dict] = None) -> List[str]:
        """
        Unified feature selection with three approaches:
        1. Random Forest feature selection (for large datasets)
        2. Sequential feature selection (for 1000+ features)
        3. Standard feature selection (for moderate feature sets)
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            max_features: Maximum number of features to select
            dataset_info: Optional dataset information containing size info
            
        Returns:
            List of selected feature names
        """
        print(f"🎯 Feature selection: selecting top {max_features} features...")
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in data.columns if col != target_column]
        total_features = len(feature_cols)
        
        print(f"📊 Total features available: {total_features}")
        
        # Determine dataset size for feature selection strategy
        if dataset_info:
            dataset_size = dataset_info.get('dataset_size', 'medium')
            row_count = dataset_info.get('total_rows', 50000)
            total_columns = dataset_info.get('total_columns', total_features)
            print(f"📊 Using provided dataset info: {dataset_size} ({row_count:,} rows, {total_columns} columns)")
        else:
            # Fallback to calculation if dataset_info is not available
            print("⚠️ Dataset info not provided, calculating size...")
            try:
                row_count = data.count()
                print(f"📊 Dataset size: {row_count:,} rows")
                
                # Classify dataset size
                if row_count < 10000:
                    dataset_size = 'small'
                elif row_count < 100000:
                    dataset_size = 'medium'
                else:
                    dataset_size = 'large'
                    
                print(f"📊 Dataset category: {dataset_size}")
                
            except Exception as e:
                print(f"⚠️ Could not determine dataset size: {e}")
                row_count = 50000  # Default assumption
                dataset_size = 'medium'
                print(f"📊 Using default dataset category: {dataset_size}")
        
        # Use intelligent sampling for feature selection if dataset is large
        if row_count > 500000:  # 500K threshold
            sample_fraction = min(0.1, 100000 / row_count)  # At most 10%, at least 100K rows
            feature_selection_data = data.sample(fraction=sample_fraction, seed=42)
            sampled_size = feature_selection_data.count()
            print(f"📊 Using {sample_fraction:.3f} sample ({sampled_size:,} rows) for feature selection")
        else:
            feature_selection_data = data
            print(f"📊 Using full dataset ({row_count:,} rows) for feature selection")
        
        # Get sequential feature selection parameters
        sequential_threshold = getattr(self, 'sequential_threshold', 200)
        
        # PRIMARY APPROACH: Sequential/Standard feature selection based on feature count
        # Approach 1: Sequential feature selection for large feature sets (PRIMARY)
        if total_features > sequential_threshold:
            print(f"🔄 Large feature set detected ({total_features} > {sequential_threshold}). Using sequential feature selection...")
            selected_features = self._sequential_feature_selection(
                feature_selection_data, feature_cols, target_column, max_features
            )
        else:
            # Approach 2: Standard feature selection for moderate feature sets
            print(f"📊 Standard feature selection for moderate feature set ({total_features} <= {sequential_threshold})...")
            selected_features = self._standard_feature_selection(
                feature_selection_data, feature_cols, target_column, max_features
            )
        
        print(f"✅ Feature selection completed!")
        print(f"📊 Selected {len(selected_features)} features out of {total_features}")
        
        # Track feature selection results
        if hasattr(self, 'variable_tracker') and self.variable_tracker:
            sequential_threshold = getattr(self, 'sequential_threshold', 200)
            method_name = "sequential_feature_selection" if total_features > sequential_threshold else "standard_feature_selection"
            self.variable_tracker.feature_selection_result(selected_features, total_features, method_name)
        
        # Generate final feature importance plot (only once, after all selection is complete)
        self._generate_final_feature_importance_plot(
            feature_selection_data, selected_features, target_column
        )
        
        return selected_features
    
    def _generate_final_feature_importance_plot(self, data: DataFrame, selected_features: List[str], 
                                              target_column: str):
        """
        Generate final feature importance plot for the selected features.
        
        Args:
            data: DataFrame with features and target
            selected_features: List of selected feature names
            target_column: Name of the target column
        """
        if not selected_features or len(selected_features) == 0:
            print("⚠️ No selected features available for final importance plot")
            return
        
        print(f"\n📊 Generating final feature importance plot for {len(selected_features)} selected features...")
        
        try:
            # Run Random Forest on selected features to get importance scores
            selected_data = data.select([target_column] + selected_features)
            
            # Create feature vector
            data_with_features = assembled_vectors(selected_data, selected_features, target_column)
            
            # Use Random Forest to get feature importance
            rf = RandomForestRegressor(
                featuresCol="features", 
                labelCol=target_column, 
                numTrees=10,
                maxDepth=5,
                maxBins=256,
                seed=42
            )
            
            rf_model = rf.fit(data_with_features)
            
            # Extract feature importance
            feature_importance = ExtractFeatureImp(rf_model.featureImportances, data_with_features, "features")
            
            # Save final feature importance plot and Excel
            output_dir = self.output_dir if self.output_dir else '.'
            excel_path, plot_path = save_feature_importance(output_dir, self.user_id, self.model_literal, feature_importance)
            
            # Verify files were created
            excel_exists = os.path.exists(excel_path) if excel_path else False
            plot_exists = os.path.exists(plot_path) if plot_path else False
            
            print(f"✅ Final feature importance saved:")
            print(f"   📊 Excel file: {excel_path} (exists: {excel_exists})")
            print(f"   📈 Plot file: {plot_path} (exists: {plot_exists})")
            
        except Exception as e:
            print(f"⚠️ Failed to generate final feature importance plot: {e}")
            print("   This doesn't affect the feature selection results")
    
    def _sequential_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                    target_column: str, max_features: int) -> List[str]:
        """
        Sequential feature selection for large feature sets.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            max_features: Final number of features to select
            
        Returns:
            List of selected feature names
        """
        chunk_size = getattr(self, 'chunk_size', 100)
        features_per_chunk = getattr(self, 'features_per_chunk', 30)
        all_selected_features = []
        
        print(f"Processing {len(feature_cols)} features in chunks of {chunk_size}")
        
        # Split features into chunks
        feature_chunks = [feature_cols[i:i + chunk_size] for i in range(0, len(feature_cols), chunk_size)]
        
        print(f"Created {len(feature_chunks)} chunks")
        
        # Process each chunk
        for i, chunk_features in enumerate(feature_chunks):
            print(f"\nProcessing chunk {i+1}/{len(feature_chunks)} with {len(chunk_features)} features...")
            
            # Select subset of data for this chunk
            chunk_data = data.select([target_column] + chunk_features)
            
            # Use Random Forest for all chunks with smaller chunk size for stability
            chunk_selected = self._run_feature_importance(
                chunk_data, chunk_features, target_column, features_per_chunk
            )
            
            all_selected_features.extend(chunk_selected)
            print(f"Selected {len(chunk_selected)} features from chunk {i+1}")
            
            # Clean up and give JVM a break between chunks
            if i < len(feature_chunks) - 1:  # Not the last chunk
                try:
                    chunk_data.unpersist()
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("Cleaned up chunk data, pausing briefly...")
                    import time
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        print(f"\nCombined {len(all_selected_features)} features from all chunks")
        
        # Final feature selection on combined features
        if len(all_selected_features) > max_features:
            print(f"Running final feature selection to select top {max_features} from {len(all_selected_features)} features...")
            
            # Reuse sequential feature selection for consistency (handles chunking if needed)
            final_selected = self._sequential_feature_selection(
                data, all_selected_features, target_column, max_features
            )
        else:
            print(f"Combined features ({len(all_selected_features)}) <= max_features ({max_features}), using all")
            final_selected = all_selected_features
        
        return final_selected
    
    def _standard_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                  target_column: str, max_features: int) -> List[str]:
        """
        Standard feature selection for moderate feature sets.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            max_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        return self._run_feature_importance(data, feature_cols, target_column, max_features)
    
    def _run_feature_importance(self, data: DataFrame, feature_cols: List[str], 
                              target_column: str, num_features: int) -> List[str]:
        """
        Run Random Forest Regression feature importance selection.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # For very large feature sets, use statistical fallback first
        if len(feature_cols) > 200:
            print(f"Large feature set detected ({len(feature_cols)} features). Using statistical preprocessing...")
            try:
                # Use correlation-based pre-filtering for very large sets
                selected_features = self._statistical_feature_selection(data, feature_cols, target_column, min(100, len(feature_cols)))
                if len(selected_features) <= num_features:
                    return selected_features
                feature_cols = selected_features
                print(f"Pre-filtered to {len(feature_cols)} features using statistical methods")
            except Exception as e:
                print(f"Statistical pre-filtering failed: {e}, continuing with original approach...")
        
        # Try Random Forest Regression with improved configuration and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Running Random Forest Regression feature importance...")
                
                # Create feature vector
                data_with_features = assembled_vectors(data, feature_cols, target_column)
                
                # Cache the data to avoid recomputation
                data_with_features.cache()
                
                # Use Random Forest Regression with reduced complexity for large feature sets
                num_trees = min(10, max(5, 50 // max(1, len(feature_cols) // 50)))  # Adaptive tree count
                max_depth = min(10, max(3, 20 - len(feature_cols) // 20))  # Adaptive depth
                
                print(f"Using RandomForestRegressor with {num_trees} trees, max depth {max_depth}")
                
                rf = RandomForestRegressor(
                    featuresCol="features", 
                    labelCol=target_column, 
                    numTrees=num_trees,
                    maxDepth=max_depth,
                    maxBins=256,  # Increased to handle high cardinality categorical features
                    subsamplingRate=0.8,  # Use subsampling to reduce memory
                    featureSubsetStrategy="sqrt"  # Use sqrt of features at each node
                )
                
                # Fit with timeout handling
                rf_model = rf.fit(data_with_features)
                
                # Extract feature importance
                feature_importance = ExtractFeatureImp(rf_model.featureImportances, data_with_features, "features")
                
                # Select top features
                top_features = feature_importance['name'].head(num_features).tolist()
                
                # Note: Feature importance plots will be saved only at the final stage
                # to avoid creating redundant plots for each chunk during sequential selection
                
                # Unpersist cached data
                data_with_features.unpersist()
                
                print(f"✅ Successfully selected {len(top_features)} features using Random Forest Regression")
                return top_features
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Try to clean up any cached data
                try:
                    if 'data_with_features' in locals():
                        data_with_features.unpersist()
                except:
                    pass
                
                # Check if this is a Py4J error and try to restart session
                if "Py4JNetworkError" in str(e) or "Answer from Java side is empty" in str(e):
                    print("Detected Py4J network error, attempting session restart...")
                    if self._restart_spark_session():
                        print("Session restarted, retrying...")
                        continue
                
                # On final attempt, use fallback method
                if attempt == max_retries - 1:
                    print("Random Forest Regression failed on all attempts. Using variance-based fallback...")
                    return self._variance_based_selection(data, feature_cols, target_column, num_features)
                
                # Wait before retry to allow JVM recovery
                time.sleep(2)
        
        # Should not reach here, but fallback just in case
        print("Warning: All feature selection methods failed, returning first n features")
        return feature_cols[:num_features]
    
    def _statistical_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                     target_column: str, num_features: int) -> List[str]:
        """
        Statistical feature selection using correlation and variance filtering for regression.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            print("Running statistical feature selection for regression...")
            
            # Remove features with zero or very low variance
            high_variance_features = []
            for col_name in feature_cols:
                try:
                    var_val = data.select(variance(col(col_name))).collect()[0][0]
                    if var_val is not None and var_val > 1e-6:  # Threshold for variance
                        high_variance_features.append(col_name)
                except:
                    continue  # Skip problematic columns
            
            print(f"Filtered {len(feature_cols)} -> {len(high_variance_features)} features by variance")
            
            if len(high_variance_features) <= num_features:
                return high_variance_features
            
            # If still too many, use correlation with target (for regression)
            try:
                # For regression, use correlation with target
                correlations = []
                for col_name in high_variance_features[:min(200, len(high_variance_features))]:  # Limit to prevent timeout
                    try:
                        corr_val = data.select(corr(col(col_name), col(target_column))).collect()[0][0]
                        if corr_val is not None:
                            correlations.append((col_name, abs(corr_val)))
                    except:
                        continue
                
                # Sort by correlation and take top features
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected = [name for name, _ in correlations[:num_features]]
                
                print(f"Selected {len(selected)} features by correlation with target")
                return selected
                
            except Exception as e:
                print(f"Correlation analysis failed: {e}")
                # Return high variance features if correlation fails
                return high_variance_features[:num_features]
                
        except Exception as e:
            print(f"Statistical feature selection failed: {e}")
            return feature_cols[:num_features]
    
    def _variance_based_selection(self, data: DataFrame, feature_cols: List[str], 
                                target_column: str, num_features: int) -> List[str]:
        """
        Variance-based feature selection for regression.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            print("Running variance-based feature selection...")
            
            # Calculate variance for each feature
            variances = []
            for col_name in feature_cols:
                try:
                    var_val = data.select(variance(col(col_name))).collect()[0][0]
                    if var_val is not None:
                        variances.append((col_name, var_val))
                except:
                    continue
            
            # Sort by variance and select top features
            variances.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in variances[:num_features]]
            
            print(f"Selected {len(selected)} features by variance")
            return selected
            
        except Exception as e:
            print(f"Variance-based selection failed: {e}")
            return feature_cols[:num_features]
    
    def _get_feature_variables(self, data: DataFrame, target_column: str, config: Dict) -> Tuple[DataFrame, List[str]]:
        """
        Get feature variables from the dataset and apply necessary conversions.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            
        Returns:
            Tuple of (modified_data, feature_variable_names)
        """
        # Get all columns except target
        all_columns = data.columns
        feature_columns = [col for col in all_columns if col != target_column]
        
        # Separate categorical and numerical features
        self.categorical_vars = []
        self.numerical_vars = []
        
        for col_name in feature_columns:
            # Check if column is categorical (string type)
            if data.schema[col_name].dataType.typeName() == 'string':
                self.categorical_vars.append(col_name)
            else:
                self.numerical_vars.append(col_name)
        
        print(f"📊 Feature analysis: {len(self.categorical_vars)} categorical, {len(self.numerical_vars)} numerical")
        
        # Track variable types
        if self.variable_tracker:
            self.variable_tracker.update_variable_types(self.categorical_vars, self.numerical_vars)
        
        # Analyze categorical cardinality and convert high-cardinality variables to numeric
        if self.categorical_vars:
            max_categorical_cardinality = config.get('max_categorical_cardinality', 50)
            categorical_vars_to_keep, categorical_vars_to_drop, cardinality_stats = analyze_categorical_cardinality(
                data, self.categorical_vars, max_categorical_cardinality
            )
            
            # Store cardinality stats for reuse in feature profiling
            self.cardinality_stats = cardinality_stats
            
            # Track variables dropped due to high cardinality
            if self.variable_tracker and categorical_vars_to_drop:
                self.variable_tracker.filter_by_cardinality(categorical_vars_to_drop, max_categorical_cardinality)
            
            # Remove high-cardinality categorical variables (simple and reliable)
            if categorical_vars_to_drop:
                data = remove_high_cardinality_categoricals(data, categorical_vars_to_drop)
                # Update variable lists - only keep low-cardinality categoricals
                self.categorical_vars = categorical_vars_to_keep  # Only keep low-cardinality ones
                # self.numerical_vars stays unchanged
                print(f"📊 Updated: {len(self.categorical_vars)} categorical, {len(self.numerical_vars)} numerical")
        
        return data, feature_columns
    
    def _apply_preprocessing_transformations(self, data: DataFrame, 
                                           categorical_vars: List[str],
                                           numerical_vars: List[str],
                                           impute_value: float) -> Tuple[DataFrame, List[str], List[str]]:
        """
        Apply preprocessing transformations to the data.
        
        Args:
            data: Input DataFrame
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            impute_value: Value to use for imputation
            
        Returns:
            Tuple of (transformed DataFrame, updated categorical_vars, updated numerical_vars)
        """
        # Get current data types
        current_dtypes = dict(data.dtypes)
        
        # Create local copies to avoid modifying the original lists
        local_categorical_vars = categorical_vars.copy()
        local_numerical_vars = numerical_vars.copy()
        
        # Convert all numerical columns that are currently string to numeric
        print("🔄 Converting string columns to numeric where needed...")
        for col_name in list(local_numerical_vars):  # Use list() to avoid modification during iteration
            if col_name in current_dtypes and current_dtypes[col_name] == 'string':
                print(f"   Converting '{col_name}' from string to numeric...")
                try:
                    from pyspark.sql.functions import when, isnull, length, trim, isnan, regexp_replace
                    
                    # Clean the column: remove non-numeric characters except decimal points and negative signs
                    cleaned_col = regexp_replace(col(col_name), '[^0-9.-]', '')
                    
                    # Handle empty strings and convert to double
                    data = data.withColumn(
                        col_name, 
                        when(
                            (isnull(col(col_name))) | 
                            (length(trim(col(col_name))) == 0) |
                            (trim(col(col_name)) == '') |
                            (trim(col(col_name)) == '?') |
                            (trim(col(col_name)) == 'NA') |
                            (trim(col(col_name)) == 'NULL'),
                            None
                        ).otherwise(cleaned_col.cast("double"))
                    )
                    print(f"   ✅ Successfully converted '{col_name}' to numeric")
                except Exception as e:
                    print(f"   ⚠️ Failed to convert '{col_name}' to numeric: {e}")
                    # If conversion fails, treat as categorical
                    local_categorical_vars.append(col_name)
                    local_numerical_vars.remove(col_name)
                    print(f"   📊 Moving '{col_name}' to categorical variables")
        
        # Check if target column exists and convert to numeric if it's string
        target_column = None
        for col_name in data.columns:
            if col_name not in local_categorical_vars and col_name not in local_numerical_vars:
                target_column = col_name
                break
        
        if target_column and target_column in current_dtypes and current_dtypes[target_column] == 'string':
            print(f"🔄 Converting string target column '{target_column}' to numeric for regression...")
            try:
                # First, check for null/empty values and handle them
                from pyspark.sql.functions import when, isnull, length, trim, isnan, regexp_replace
                
                # Clean the target column
                cleaned_col = regexp_replace(col(target_column), '[^0-9.-]', '')
                
                # Handle empty strings and convert to double
                data = data.withColumn(
                    target_column, 
                    when(
                        (isnull(col(target_column))) | 
                        (length(trim(col(target_column))) == 0) |
                        (trim(col(target_column)) == '') |
                        (trim(col(target_column)) == '?') |
                        (trim(col(target_column)) == 'NA') |
                        (trim(col(target_column)) == 'NULL'),
                        None
                    ).otherwise(cleaned_col.cast("double"))
                )
                
                # Filter out null/NaN values from target
                data = data.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                
                print(f"📊 Final rows after target conversion: {data.count()}")
                print(f"✅ Successfully converted '{target_column}' to numeric type")
                
            except Exception as e:
                print(f"⚠️ Failed to convert '{target_column}' to numeric: {e}")
                print("🔄 Attempting to use StringIndexer for categorical target...")
                try:
                    # Use StringIndexer as fallback for categorical targets
                    target_indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_indexed")
                    fitted_indexer = target_indexer.fit(data)
                    data = fitted_indexer.transform(data)
                    # Replace original target column with indexed version
                    data = data.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
                    print(f"✅ Successfully encoded '{target_column}' using StringIndexer")
                except Exception as e2:
                    print(f"❌ Failed to encode target column: {e2}")
                    raise ValueError(f"Cannot convert target column '{target_column}' to numeric format for regression")
        
        # Apply categorical encoding
        if local_categorical_vars:
            # Reuse already calculated cardinality stats to avoid redundant BigQuery queries
            print(f"♻️ Reusing cardinality stats for {len(local_categorical_vars)} categorical features...")
            filtered_categorical_vars = []
            
            # Check if we have cardinality stats from the earlier analysis
            if hasattr(self, 'cardinality_stats') and self.cardinality_stats:
                print(f"✅ Using pre-calculated cardinality statistics (threshold: 200 unique values)")
                
                for cat_var in local_categorical_vars:
                    if cat_var in self.cardinality_stats:
                        # Use the already calculated unique count
                        distinct_count = self.cardinality_stats[cat_var].get('unique_count', 0)
                        print(f"   ♻️ {cat_var}: {distinct_count} unique values (reused)")
                        
                        # Keep features with reasonable cardinality (up to 200 for our maxBins=256 setting)
                        if distinct_count <= 200:
                            filtered_categorical_vars.append(cat_var)
                        else:
                            print(f"   ⚠️ Skipping high-cardinality feature '{cat_var}' ({distinct_count} unique values > 200)")
                    else:
                        # Fallback: include the variable if stats are missing (shouldn't happen normally)
                        print(f"   ⚠️ No cardinality stats for '{cat_var}', including anyway")
                        filtered_categorical_vars.append(cat_var)
            else:
                # Fallback: if no cardinality stats available, include all variables
                print(f"   ⚠️ No pre-calculated cardinality stats available, including all {len(local_categorical_vars)} variables")
                filtered_categorical_vars = local_categorical_vars.copy()
            
            print(f"📈 Categorical feature filtering: {len(local_categorical_vars)} → {len(filtered_categorical_vars)} features (optimized)")
            
            if filtered_categorical_vars:
                data, char_labels = categorical_to_index(data, filtered_categorical_vars)
                self.char_labels = char_labels
                print(f"Categorical encoding step complete.")
                
                # Remove original categorical columns (only the ones we encoded)
                data = data.select([c for c in data.columns if c not in filtered_categorical_vars])
            else:
                print("   📊 No categorical features remaining after cardinality filtering")
                self.char_labels = None
        
        # Apply numerical imputation
        if local_numerical_vars:
            data = numerical_imputation(data, local_numerical_vars, impute_value)
            print(f"Numerical impuation step complete.")
        
        # No need to rename columns since categorical_to_index now creates _encoded suffix directly
        # data = rename_columns(data, categorical_vars)
        print(f"Categorical encoding complete with _encoded suffix.")
        
        return data, local_categorical_vars, local_numerical_vars
    
    def split_and_scale(self, data: DataFrame, train_size: float = 0.7, 
                       valid_size: float = 0.2, target_column: str = 'target', 
                       seed: int = 42, config: Optional[Dict] = None) -> tuple:
        """
        Split data into train/validation/test sets and apply scaling.
        
        Args:
            data: Input DataFrame
            train_size: Proportion for training
            valid_size: Proportion for validation
            target_column: Name of the target column
            seed: Random seed
            config: Configuration dictionary
            
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        print("📊 Splitting and scaling data...")
        
        # Split data
        splits = data.randomSplit([train_size, valid_size, 1 - train_size - valid_size], seed=seed)
        train_df, valid_df, test_df = splits
        
        print(f"Data split: Train={train_df.count()}, Valid={valid_df.count()}, Test={test_df.count()}")
        
        # Apply scaling if configured
        scaling_method = config.get('scaling_method', 'standard') if config else 'standard'
        
        if scaling_method != 'none':
            train_df = self.apply_scaling(train_df, target_column)
            valid_df = self.apply_scaling(valid_df, target_column)
            test_df = self.apply_scaling(test_df, target_column)
        
        return train_df, valid_df, test_df
    
    def apply_scaling(self, data: DataFrame, target_column: str) -> DataFrame:
        """
        Apply scaling to numerical features.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Scaled DataFrame
        """
        # Check if features column already exists (data already processed)
        if 'features' in data.columns:
            print("🔧 Features column already exists, skipping scaling")
            return data
        
        print("🔧 Creating feature scaling pipeline...")
        
        # Use selected features if available, otherwise use all available columns (exclude target column)
        if self.selected_vars:
            # Use selected features (excluding target column)
            available_cols = [col for col in self.selected_vars if col != target_column and col in data.columns]
            print(f"🎯 Using selected features: {len(available_cols)} columns: {available_cols}")
        else:
            # Fallback to all available columns (exclude target column)
            available_cols = [col for col in data.columns if col != target_column]
            print(f"🔗 Using all available features: {len(available_cols)} columns: {available_cols}")
        
        # Create feature vector and scale it
        assembler = VectorAssembler(inputCols=available_cols, outputCol='assembled_features')
        scaler = StandardScaler(inputCol='assembled_features', outputCol='features')
        pipeline = Pipeline(stages=[assembler, scaler])
        
        # Fit the pipeline on the first dataset (training data) and store it
        if self.pipeline_model is None:
            print("📊 Fitting scaling pipeline on training data...")
            self.pipeline_model = pipeline.fit(data)
        
        # Transform the data using the fitted pipeline
        scaled_data = self.pipeline_model.transform(data)
        
        print("✅ Feature scaling pipeline applied successfully")
        return scaled_data

    def apply_preprocessing(self, data: DataFrame,
                            feature_vars: List[str],
                            selected_vars: List[str],
                            categorical_vars: List[str],
                            numerical_vars: List[str],
                            char_labels: Optional[PipelineModel],
                            impute_value: float,
                            target_column: Optional[str] = None,
                            target_label_indexer: Optional[Any] = None) -> DataFrame:
        """
        Apply the fitted preprocessing pipeline to a new dataset.

        This function mirrors the classification data processor's `apply_preprocessing` method.
        It starts from the raw feature columns to ensure categorical variables exist for
        encoding, applies the stored categorical encoding pipeline, performs numerical
        imputation, drops original categorical columns, and finally selects only the
        encoded feature columns present in `selected_vars` (plus the target column if present).

        Args:
            data: The raw input DataFrame to preprocess.
            feature_vars: Original list of feature names used during training (before encoding).
            selected_vars: List of selected encoded feature names from training.
            categorical_vars: List of raw categorical variable names.
            numerical_vars: List of raw numerical variable names.
            char_labels: Fitted StringIndexer pipeline used to encode categorical variables.
            impute_value: Value used for numerical imputation.
            target_column: Name of the target column (optional).
            target_label_indexer: Fitted StringIndexer for the target column (optional).

        Returns:
            Preprocessed DataFrame with columns matching `selected_vars` (and target column).
        """
        # Start by selecting the raw feature columns that exist in the new data
        raw_cols = [c for c in feature_vars if c in data.columns]

        # Filter out any obvious date/timestamp columns (simple heuristic)
        filtered_cols = []
        for col_name in raw_cols:
            try:
                dtype = dict(data.dtypes).get(col_name)
                if dtype not in ["timestamp", "date"]:
                    filtered_cols.append(col_name)
            except Exception:
                filtered_cols.append(col_name)

        columns_to_select = list(filtered_cols)
        # Include target column if provided
        if target_column and target_column in data.columns and target_column not in columns_to_select:
            columns_to_select.append(target_column)


        # Create a DataFrame with only the necessary columns
        # If some expected raw columns are missing from the new dataset (e.g., OOT datasets),
        # we will add them with null values after selection. This ensures that the
        # fitted preprocessing pipeline (categorical encoders and imputers) can operate
        # without raising "column does not exist" errors. Missing categorical columns
        # are cast to string type to enable encoding; missing numerical columns are cast
        # to double so that imputation works correctly.
        X = data.select(columns_to_select)

        # ------------------------------------------------------------------
        # Ensure all expected raw feature columns exist in the DataFrame
        # When applying the preprocessing pipeline to OOT datasets, some
        # expected columns may be missing.  Fitted stages (e.g., StringIndexer,
        # OneHotEncoder) require their input columns to exist.  Similarly,
        # numerical imputation expects all numeric columns to be present.  To
        # avoid "column does not exist" errors, add any missing raw categorical
        # or numerical columns with null values.  Categorical columns are cast
        # to string so that encoding behaves correctly; numerical columns are
        # cast to double so that imputation can later convert them appropriately.
        from pyspark.sql import functions as F
        for col_name in (categorical_vars + numerical_vars):
            if col_name not in X.columns:
                if col_name in categorical_vars:
                    X = X.withColumn(col_name, F.lit(None).cast('string'))
                else:
                    X = X.withColumn(col_name, F.lit(None).cast('double'))

        # Apply target label encoding if necessary (rare for regression)
        if target_column and target_label_indexer is not None and target_column in X.columns:
            try:
                X = target_label_indexer.transform(X)
                X = X.drop(target_column).withColumnRenamed(f"{target_column}_indexed", target_column)
            except Exception:
                pass  # If encoding fails, assume target is already numeric

        # Apply categorical encoding using fitted pipeline
        if char_labels is not None:
            X = char_labels.transform(X)

        # Impute numerical variables
        if numerical_vars:
            X = numerical_imputation(X, numerical_vars, impute_value)

        # Drop original categorical variables
        if categorical_vars:
            X = X.select([c for c in X.columns if c not in categorical_vars])

        # Select only the selected_vars (encoded feature names) plus target column
        final_columns: List[str] = []
        for col_name in selected_vars:
            if col_name in X.columns:
                final_columns.append(col_name)
        if target_column and target_column in X.columns and target_column not in final_columns:
            final_columns.append(target_column)

        if final_columns:
            X = X.select(final_columns)

        return X
    
    def _restart_spark_session(self):
        """Restart Spark session if needed."""
        try:
            # This is a simplified restart - in practice, you might need more complex logic
            print("Attempting to restart Spark session...")
            return True
        except Exception as e:
            print(f"Failed to restart Spark session: {e}")
            return False
    
    def process_data(self, train_data: Union[str, DataFrame], target_column: Optional[str] = None,
                    oot1_data: Optional[Union[str, DataFrame]] = None,
                    oot2_data: Optional[Union[str, DataFrame]] = None,
                    **kwargs) -> Dict[str, DataFrame]:
        """
        Process data for regression training.
        
        Args:
            train_data: Training data (file path or DataFrame)
            target_column: Name of the target column
            oot1_data: Optional out-of-time validation data
            oot2_data: Optional second out-of-time validation data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed datasets
        """
        print("🔄 Processing data for regression...")
        
        # Load data if string paths are provided
        if isinstance(train_data, str):
            print(f"📁 Loading training data from: {train_data}")
            train_data = self.spark.read.csv(train_data, header=True, inferSchema=True)
        
        if isinstance(oot1_data, str):
            print(f"📁 Loading OOT1 data from: {oot1_data}")
            oot1_data = self.spark.read.csv(oot1_data, header=True, inferSchema=True)
        
        if isinstance(oot2_data, str):
            print(f"📁 Loading OOT2 data from: {oot2_data}")
            oot2_data = self.spark.read.csv(oot2_data, header=True, inferSchema=True)
        
        # Preprocess training data
        processed_train, feature_vars, selected_vars, categorical_vars, numerical_vars = self.preprocess(
            train_data, target_column, kwargs, kwargs.get('dataset_info')
        )
        
        # Store preprocessing artifacts
        self.feature_vars = feature_vars
        self.selected_vars = selected_vars
        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        
        # Split training data
        train_size = kwargs.get('train_size', 0.7)
        valid_size = kwargs.get('valid_size', 0.2)
        seed = kwargs.get('seed', 42)
        
        train_df, valid_df, test_df = self.split_and_scale(
            processed_train, train_size, valid_size, target_column, seed, kwargs
        )
        
        # Process OOT data if provided
        oot1_df = None
        oot2_df = None
        
        if oot1_data is not None:
            print("🔄 Processing OOT1 data...")
            # Apply the same preprocessing pipeline used on training data
            oot1_df = self.apply_preprocessing(
                oot1_data,
                feature_vars,
                selected_vars,
                categorical_vars,
                numerical_vars,
                self.char_labels,
                kwargs.get('impute_value', -999),
                target_column,
                getattr(self, 'target_label_indexer', None)
            )
            # Clean OOT1 data: remove null/NaN values in target column
            if target_column and target_column in oot1_df.columns:
                print("🔍 Cleaning OOT1 target column...")
                from pyspark.sql.functions import isnull, isnan
                original_count = oot1_df.count()
                oot1_df = oot1_df.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                clean_count = oot1_df.count()
                if original_count != clean_count:
                    print(f"   📊 OOT1: Filtered out {original_count - clean_count} rows with null/NaN target values")
        
        if oot2_data is not None:
            print("🔄 Processing OOT2 data...")
            # Apply the same preprocessing pipeline used on training data
            oot2_df = self.apply_preprocessing(
                oot2_data,
                feature_vars,
                selected_vars,
                categorical_vars,
                numerical_vars,
                self.char_labels,
                kwargs.get('impute_value', -999),
                target_column,
                getattr(self, 'target_label_indexer', None)
            )
            # Clean OOT2 data: remove null/NaN values in target column
            if target_column and target_column in oot2_df.columns:
                print("🔍 Cleaning OOT2 target column...")
                from pyspark.sql.functions import isnull, isnan
                original_count = oot2_df.count()
                oot2_df = oot2_df.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                clean_count = oot2_df.count()
                if original_count != clean_count:
                    print(f"   📊 OOT2: Filtered out {original_count - clean_count} rows with null/NaN target values")
        
        # Create feature vectors for all datasets (only if features column doesn't exist)
        if 'features' not in train_df.columns:
            print("🔄 Creating feature vectors...")
            
            # Determine feature columns for assembling vectors. If selected_vars are available
            # (resulting from feature selection), use them; otherwise fall back to all numeric and
            # encoded categorical variables. Exclude the target column from the feature list.
            if self.selected_vars:
                all_feature_cols = [c for c in self.selected_vars if c != target_column]
            else:
                all_feature_cols = numerical_vars.copy()
                for cat_var in categorical_vars:
                    all_feature_cols.append(f"{cat_var}_encoded")
            
            # Create assembler
            assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")
            
            # Transform all datasets
            train_df = assembler.transform(train_df)
            valid_df = assembler.transform(valid_df) if valid_df is not None else None
            test_df = assembler.transform(test_df) if test_df is not None else None
            oot1_df = assembler.transform(oot1_df) if oot1_df is not None else None
            oot2_df = assembler.transform(oot2_df) if oot2_df is not None else None
        else:
            print("🔄 Features column already exists, skipping vector creation")
        
        print("✅ Data processing completed")
        print(f"📊 Final dataset sizes:")
        print(f"   Train: {train_df.count()}")
        if valid_df:
            print(f"   Valid: {valid_df.count()}")
        if test_df:
            print(f"   Test: {test_df.count()}")
        if oot1_df:
            print(f"   OOT1: {oot1_df.count()}")
        if oot2_df:
            print(f"   OOT2: {oot2_df.count()}")
        
        return {
            'train': train_df,
            'valid': valid_df,
            'test': test_df,
            'oot1': oot1_df,
            'oot2': oot2_df
        }
    
    def _filter_date_columns(self, data: DataFrame, feature_vars: List[str]) -> List[str]:
        """
        Filter out date/timestamp columns from the list of feature variables.
        This is important because these columns are often not suitable for machine learning.
        
        Args:
            data: Input DataFrame
            feature_vars: List of feature variable names
            
        Returns:
            List of feature variable names after date/timestamp filtering
        """
        from pyspark.sql.types import DateType, TimestampType
        
        date_types = ['date', 'timestamp']
        date_like_patterns = ['date', 'time', 'dt_', '_dt', 'timestamp']
        
        filtered_feature_vars = []
        filtered_date_columns = []
        
        for var in feature_vars:
            is_date_column = False
            
            try:
                # Check actual Spark data type
                field = data.schema[var]
                if isinstance(field.dataType, (DateType, TimestampType)):
                    is_date_column = True
                    filtered_date_columns.append(f"{var} (Spark {field.dataType.simpleString()})")
                
                # Also check string representation for edge cases
                field_type = field.dataType.simpleString()
                if field_type in date_types:
                    is_date_column = True
                    if var not in [col.split(' ')[0] for col in filtered_date_columns]:
                        filtered_date_columns.append(f"{var} ({field_type})")
                
                # Check for common date column naming patterns
                var_lower = var.lower()
                if any(pattern in var_lower for pattern in date_like_patterns):
                    # Additional validation: check if column contains date-like strings
                    sample_values = data.select(var).limit(5).collect()
                    if sample_values and self._looks_like_date_column(sample_values, var):
                        is_date_column = True
                        if var not in [col.split(' ')[0] for col in filtered_date_columns]:
                            filtered_date_columns.append(f"{var} (date pattern)")
                
            except (KeyError, Exception) as e:
                # If there's any issue checking the column, keep it to be safe
                pass
            
            if not is_date_column:
                filtered_feature_vars.append(var)
        
        # Log filtered columns
        if filtered_date_columns:
            print(f"🗓️ Automatically filtered out {len(filtered_date_columns)} date/timestamp columns:")
            for col_info in filtered_date_columns:
                print(f"   - {col_info}")
            print(f"💡 Date columns are excluded because they often don't provide meaningful features for ML models")
        
        return filtered_feature_vars
    
    def _looks_like_date_column(self, sample_values, column_name: str) -> bool:
        """
        Check if sample values look like dates/timestamps.
        
        Args:
            sample_values: List of Row objects with sample values
            column_name: Name of the column being checked
            
        Returns:
            bool: True if values look like dates
        """
        import re
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        ]
        
        date_like_count = 0
        total_non_null = 0
        
        for row in sample_values:
            value = row[column_name] if hasattr(row, column_name) else row[0]
            if value is not None:
                total_non_null += 1
                value_str = str(value)
                
                # Check against date patterns
                if any(re.search(pattern, value_str) for pattern in date_patterns):
                    date_like_count += 1
                # Check for long integers that might be timestamps
                elif re.match(r'^\d{10,13}$', value_str):  # Unix timestamps
                    date_like_count += 1
        
        # If more than 50% of non-null values look like dates, consider it a date column
        if total_non_null > 0:
            date_percentage = date_like_count / total_non_null
            return date_percentage > 0.5
        
        return False 