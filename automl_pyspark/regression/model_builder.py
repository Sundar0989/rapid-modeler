"""
Regression Model Builder

Class responsible for building different types of regression models.
This class encapsulates the model building functionality from the original modules.
"""

import os
from typing import Any, Dict, List, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import (
    LinearRegression, LinearRegressionModel,
    RandomForestRegressor, RandomForestRegressionModel,
    GBTRegressor, GBTRegressionModel,
    DecisionTreeRegressor, DecisionTreeRegressionModel,
    FMRegressor, FMRegressionModel
)
# Note: PySpark doesn't have MultilayerPerceptronRegressor, only MultilayerPerceptronClassifier

# Advanced ML algorithms (optional imports with fallbacks)
try:
    from xgboost.spark import SparkXGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available for regression. Install with: pip install xgboost>=1.6.0")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from pyspark.sql.types import StructType, StructField, DoubleType
    from pyspark.ml.linalg import VectorUDT, Vectors
    LIGHTGBM_AVAILABLE = True
    print(f"✅ Native LightGBM regression available and functional (version: {lgb.__version__})")
except ImportError as e:
    print(f"⚠️ Native LightGBM import failed: {e}")
    print("   Install with: pip install lightgbm>=4.0.0")
    LIGHTGBM_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Native LightGBM initialization failed: {e}")
    print("   LightGBM may be installed but not compatible with current environment")
    LIGHTGBM_AVAILABLE = False


class NativeLightGBMRegressionModel:
    """
    Wrapper for native LightGBM regression model to provide Spark-compatible interface.
    """
    
    def __init__(self, lgb_model, feature_names: List[str], label_col: str, 
                 features_col: str):
        """
        Initialize the wrapper with a trained LightGBM regression model.
        
        Args:
            lgb_model: Trained LightGBM model
            feature_names: List of feature column names
            label_col: Name of the label column
            features_col: Name of the features column
        """
        self.lgb_model = lgb_model
        self.feature_names = feature_names
        self.label_col = label_col
        self.features_col = features_col
        
    def transform(self, dataset: DataFrame, *args, **kwargs) -> DataFrame:
        """
        Transform a DataFrame using the trained LightGBM regression model.
        
        Args:
            dataset: Input DataFrame with features
            *args: Additional positional arguments (for Spark ML compatibility)
            **kwargs: Additional keyword arguments (for Spark ML compatibility)
            
        Returns:
            DataFrame with predictions
        """
        spark = dataset.sql_ctx.sparkSession
        
        # Convert Spark DataFrame to Pandas for LightGBM prediction
        pandas_df = dataset.toPandas()
        
        # Extract features from vector column
        if self.features_col in pandas_df.columns:
            # Convert vector column to numpy array
            features_array = np.array([
                np.array(row) for row in pandas_df[self.features_col]
            ])
        else:
            # Use individual feature columns - but check if feature_names exist in DataFrame
            available_cols = [col for col in self.feature_names if col in pandas_df.columns]
            if len(available_cols) == len(self.feature_names):
                # All expected feature columns are available
                features_array = pandas_df[self.feature_names].values
            else:
                # Feature names don't match - this happens when model was trained with vector features
                # but we're trying to predict with individual columns
                # Use all non-label columns as features
                feature_cols = [col for col in pandas_df.columns if col not in [self.label_col, 'prediction']]
                if len(feature_cols) >= len(self.feature_names):
                    # Take the first N columns that match the expected feature count
                    features_array = pandas_df[feature_cols[:len(self.feature_names)]].values
                else:
                    raise ValueError(f"Expected {len(self.feature_names)} features, but only found {len(feature_cols)} columns: {feature_cols}")
        
        # Make predictions
        predictions = self.lgb_model.predict(features_array)
        
        # Add predictions to pandas DataFrame
        pandas_df['prediction'] = predictions.astype(float)
        
        # Convert back to Spark DataFrame
        # Define schema for the result
        original_schema = dataset.schema
        
        # Add prediction column to schema
        result_fields = list(original_schema.fields)
        result_fields.append(StructField("prediction", DoubleType(), True))
        
        result_schema = StructType(result_fields)
        
        # Create result DataFrame
        result_df = spark.createDataFrame(pandas_df, schema=result_schema)
        
        return result_df
    
    def write(self):
        """
        Provide write interface for compatibility with Spark ML models.
        """
        return NativeLightGBMRegressionWriter(self)
    
    def save(self, path: str):
        """
        Save the LightGBM regression model to the specified path.
        
        Args:
            path: Path to save the model
        """
        import os
        import pickle
        
        os.makedirs(path, exist_ok=True)
        
        # Save LightGBM model
        self.lgb_model.save_model(os.path.join(path, "lightgbm_model.txt"))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'label_col': self.label_col,
            'features_col': self.features_col
        }
        
        with open(os.path.join(path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str):
        """
        Load a saved LightGBM regression model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded NativeLightGBMRegressionModel
        """
        import os
        import pickle
        
        # Load LightGBM model
        lgb_model = lgb.Booster(model_file=os.path.join(path, "lightgbm_model.txt"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        return cls(
            lgb_model=lgb_model,
            feature_names=metadata['feature_names'],
            label_col=metadata['label_col'],
            features_col=metadata['features_col']
        )


class NativeLightGBMRegressionWriter:
    """
    Writer class for NativeLightGBMRegressionModel to provide Spark ML compatibility.
    """
    
    def __init__(self, model: NativeLightGBMRegressionModel):
        self.model = model
        self._overwrite = False
    
    def overwrite(self):
        """Enable overwrite mode."""
        self._overwrite = True
        return self
    
    def save(self, path: str):
        """Save the model to the specified path."""
        self.model.save(path)


class NativeLightGBMRegressor:
    """
    Native LightGBM Regressor that provides Spark ML compatible interface.
    """
    
    def __init__(self, featuresCol: str = "features", labelCol: str = "label",
                 numLeaves: int = 31, numIterations: int = 100, 
                 learningRate: float = 0.1, featureFraction: float = 1.0, 
                 baggingFraction: float = 1.0, lambdaL1: float = 0.0, 
                 lambdaL2: float = 0.0, seed: int = 42):
        """
        Initialize Native LightGBM Regressor.
        
        Args:
            featuresCol: Name of the features column
            labelCol: Name of the label column
            numLeaves: Number of leaves in trees
            numIterations: Number of boosting iterations
            learningRate: Learning rate
            featureFraction: Feature fraction for bagging
            baggingFraction: Bagging fraction
            lambdaL1: L1 regularization
            lambdaL2: L2 regularization
            seed: Random seed
        """
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.numLeaves = numLeaves
        self.numIterations = numIterations
        self.learningRate = learningRate
        self.featureFraction = featureFraction
        self.baggingFraction = baggingFraction
        self.lambdaL1 = lambdaL1
        self.lambdaL2 = lambdaL2
        self.seed = seed
    
    def getFeaturesCol(self) -> str:
        """Get features column name."""
        return self.featuresCol
    
    def getLabelCol(self) -> str:
        """Get label column name."""
        return self.labelCol
    
    def copy(self, extra=None):
        """Create a copy of this estimator."""
        return NativeLightGBMRegressor(
            featuresCol=self.featuresCol,
            labelCol=self.labelCol,
            numLeaves=self.numLeaves,
            numIterations=self.numIterations,
            learningRate=self.learningRate,
            featureFraction=self.featureFraction,
            baggingFraction=self.baggingFraction,
            lambdaL1=self.lambdaL1,
            lambdaL2=self.lambdaL2,
            seed=self.seed
        )
    
    def fitMultiple(self, dataset: DataFrame, paramMaps):
        """
        Fit multiple models with different parameter sets for cross-validation.
        
        Args:
            dataset: Training DataFrame
            paramMaps: List of parameter maps (not used for native LightGBM)
            
        Returns:
            Iterator of (index, model) tuples for Spark ML compatibility
        """
        print(f"🔄 fitMultiple called for Native LightGBM cross-validation")
        
        # Spark ML expects (index, model) tuples from fitMultiple
        def fit_models():
            for i, param_map in enumerate(paramMaps):
                print(f"📊 Training model {i+1}/{len(paramMaps)} for CV fold")
                model = self.fit(dataset)
                yield (i, model)
        
        return fit_models()
    
    def fit(self, dataset: DataFrame, *args, **kwargs) -> NativeLightGBMRegressionModel:
        """
        Fit the LightGBM regression model on the training dataset.
        
        Args:
            dataset: Training DataFrame
            *args: Additional positional arguments (for Spark ML compatibility)
            **kwargs: Additional keyword arguments (for Spark ML compatibility)
            
        Returns:
            Trained NativeLightGBMRegressionModel
        """
        print(f"🚀 Training Native LightGBM Regressor")
        
        # Convert Spark DataFrame to Pandas
        pandas_df = dataset.toPandas()
        
        # Extract features and labels
        if self.featuresCol in pandas_df.columns:
            # Convert vector column to numpy array
            X = np.array([
                np.array(row) for row in pandas_df[self.featuresCol]
            ])
            # Try to get actual feature names from the dataset schema or use generic names
            # For now, use generic names but store them for consistency
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            # Use individual feature columns (fallback)
            feature_cols = [col for col in pandas_df.columns if col != self.labelCol]
            X = pandas_df[feature_cols].values
            feature_names = feature_cols
        
        y = pandas_df[self.labelCol].values
        
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📋 Target range: {y.min():.4f} to {y.max():.4f}")
        
        # Configure LightGBM parameters for regression
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': self.numLeaves,
            'learning_rate': self.learningRate,
            'feature_fraction': self.featureFraction,
            'bagging_fraction': self.baggingFraction,
            'lambda_l1': self.lambdaL1,
            'lambda_l2': self.lambdaL2,
            'seed': self.seed,
            'verbose': -1,
            'force_row_wise': True  # Avoid threading issues in Spark
        }
        
        print(f"🔧 LightGBM parameters: {params}")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        
        # Train model
        print(f"🏋️ Training LightGBM regression model...")
        lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=self.numIterations,
            callbacks=[lgb.log_evaluation(0)]  # Silent training
        )
        
        print(f"✅ Native LightGBM regression model trained successfully!")
        print(f"📊 Model info: {lgb_model.num_trees()} trees, {lgb_model.num_feature()} features")
        
        # Create wrapper model
        wrapper_model = NativeLightGBMRegressionModel(
            lgb_model=lgb_model,
            feature_names=feature_names,
            label_col=self.labelCol,
            features_col=self.featuresCol
        )
        
        return wrapper_model


def linear_regression_model(train, x, y):
    """Build Linear Regression model."""
    lr = LinearRegression(
        featuresCol=x,
        labelCol=y,
        regParam=0.01,
        elasticNetParam=0.0
    )
    lrModel = lr.fit(train)
    return lrModel


def random_forest_regression_model(train, x, y):
    """Build Random Forest Regression model."""
    rf = RandomForestRegressor(
        featuresCol=x,
        labelCol=y,
        numTrees=100,
        maxDepth=5,
        maxBins=256,  # Increased to handle very high-cardinality categorical features (up to 256 unique values)
        seed=42
    )
    rfModel = rf.fit(train)
    return rfModel


def gradient_boosting_regression_model(train, x, y):
    """
    Build a Gradient Boosting Regression model with conservative defaults.  By limiting
    the number of iterations and using fewer bins, the model trains faster and
    avoids large broadcast task binaries that can destabilise a Spark session.
    """
    # Apply gradient boosting specific optimisations to reduce broadcasting warnings
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if spark:
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from spark_optimization_config import apply_gradient_boosting_optimizations  # type: ignore
            apply_gradient_boosting_optimizations(spark)
        except Exception as e:
            # Optimisation module might be missing; log and continue
            print(f"⚠️ Could not apply gradient boosting optimisations: {e}")
    
    gbt = GBTRegressor(
        featuresCol=x,
        labelCol=y,
        maxIter=50,   # Fewer boosting iterations to reduce model complexity
        maxDepth=5,
        maxBins=64,   # Smaller number of bins to reduce broadcast size
        seed=42
    )
    gbtModel = gbt.fit(train)
    return gbtModel


def decision_tree_regression_model(train, x, y):
    """Build Decision Tree Regression model."""
    dt = DecisionTreeRegressor(
        featuresCol=x,
        labelCol=y,
        maxDepth=5,
        maxBins=256,  # Increased to handle very high-cardinality categorical features (up to 256 unique values)
        seed=42
    )
    dtModel = dt.fit(train)
    return dtModel


def fm_regression_model(train, x, y):
    """Build Factorization Machine Regression model."""
    fm = FMRegressor(
        featuresCol=x,
        labelCol=y,
        factorSize=8,
        seed=42
    )
    fmModel = fm.fit(train)
    return fmModel


def xgboost_regression_model(train, x, y):
    """Build XGBoost Regression model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost>=1.6.0")
    
    xgb = SparkXGBRegressor(
        features_col=x,  # Fixed: Use features_col instead of featuresCol
        label_col=y,     # Fixed: Use label_col instead of labelCol
        maxDepth=6,
        eta=0.3,
        numRound=100,
        seed=42
    )
    xgbModel = xgb.fit(train)
    return xgbModel


def lightgbm_regression_model(train, x, y):
    """Build Native LightGBM Regression model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("Native LightGBM is not available. Install with: pip install lightgbm>=4.0.0")
    
    print(f"🚀 Building Native LightGBM regression model")
    
    lgb = NativeLightGBMRegressor(
        featuresCol=x,
        labelCol=y,
        numLeaves=31,
        numIterations=100,
        learningRate=0.1,
        featureFraction=1.0,
        baggingFraction=1.0,
        lambdaL1=0.0,
        lambdaL2=0.0,
        seed=42
    )
    
    print(f"✅ Native LightGBM regressor created, fitting model...")
    lgbModel = lgb.fit(train)
    print(f"✅ Native LightGBM regression model fitted successfully")
    
    return lgbModel


# Neural Network regression is not supported in PySpark
# PySpark only has MultilayerPerceptronClassifier, not MultilayerPerceptronRegressor
# def neural_network_regression_model(train, x, y):
#     """Neural Network regression is not natively supported in PySpark."""
#     raise NotImplementedError("PySpark doesn't have MultilayerPerceptronRegressor. Use other regression models.")


class RegressionModelBuilder:
    """
    Model builder class that handles building different types of regression models.
    
    This class provides functionality for:
    - Building various regression models
    - Saving and loading models
    - Model parameter configuration
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the regression model builder.
        
        Args:
            spark_session: PySpark SparkSession
        """
        self.spark = spark_session
        
        # Model type mappings
        self.model_types = {
            'linear_regression': {
                'class': LinearRegression,
                'model_class': LinearRegressionModel,
                'build_func': linear_regression_model
            },
            'random_forest': {
                'class': RandomForestRegressor,
                'model_class': RandomForestRegressionModel,
                'build_func': random_forest_regression_model
            },
            'gradient_boosting': {
                'class': GBTRegressor,
                'model_class': GBTRegressionModel,
                'build_func': gradient_boosting_regression_model
            },
            'decision_tree': {
                'class': DecisionTreeRegressor,
                'model_class': DecisionTreeRegressionModel,
                'build_func': decision_tree_regression_model
            },
            'fm_regression': {
                'class': FMRegressor,
                'model_class': FMRegressionModel,
                'build_func': fm_regression_model
            },
# Neural Network not supported for regression in PySpark
            # 'neural_network': {
            #     'class': None,
            #     'model_class': None,
            #     'build_func': None
            # }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_types['xgboost'] = {
                'class': SparkXGBRegressor,
                'model_class': None,
                'build_func': xgboost_regression_model
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_types['lightgbm'] = {
                'class': NativeLightGBMRegressor,
                'model_class': NativeLightGBMRegressionModel,
                'build_func': lightgbm_regression_model
            }
            self.model_types['lightgbm'] = {
                'class': NativeLightGBMRegressor,
                'model_class': NativeLightGBMRegressionModel,
                'build_func': lightgbm_regression_model
            }
        
        print(f"✅ RegressionModelBuilder initialized with {len(self.model_types)} model types")
        if XGBOOST_AVAILABLE:
            print("   📦 XGBoost regression available")
        if LIGHTGBM_AVAILABLE:
            print("   📦 LightGBM regression available")
    
    def create_estimator(self, features_col: str, label_col: str, model_type: str, **params) -> Any:
        """
        Create an estimator (unfitted model) for the specified model type.
        
        Args:
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to create
            **params: Additional parameters for the model
            
        Returns:
            Unfitted estimator object
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_info = self.model_types[model_type]
        model_class = model_info['class']
        
        # Create estimator with default parameters
        if model_type == 'linear_regression':
            estimator = LinearRegression(
                featuresCol=features_col,
                labelCol=label_col,
                regParam=params.get('regParam', 0.01),
                elasticNetParam=params.get('elasticNetParam', 0.0),
                maxIter=params.get('maxIter', 100)
            )
        elif model_type == 'random_forest':
            estimator = RandomForestRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numTrees=params.get('numTrees', 20),
                maxDepth=params.get('maxDepth', 5),
                maxBins=params.get('maxBins', 32),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                subsamplingRate=params.get('subsamplingRate', 1.0),
                seed=42
            )
        elif model_type == 'gradient_boosting':
            estimator = GBTRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=params.get('maxIter', 20),
                maxDepth=params.get('maxDepth', 5),
                stepSize=params.get('stepSize', 0.1),
                seed=42
            )
        elif model_type == 'decision_tree':
            estimator = DecisionTreeRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxDepth=params.get('maxDepth', 5),
                maxBins=params.get('maxBins', 32),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                minInfoGain=params.get('minInfoGain', 0.0),
                seed=42
            )
        elif model_type == 'fm_regression':
            estimator = FMRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                factorSize=params.get('factorSize', 8),
                regParam=params.get('regParam', 0.01),
                miniBatchFraction=params.get('miniBatchFraction', 1.0),
                initStd=params.get('initStd', 0.01),
                maxIter=params.get('maxIter', 100),
                stepSize=params.get('stepSize', 1.0),
                tol=params.get('tol', 1e-6),
                solver=params.get('solver', 'adamW'),
                seed=42
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            estimator = SparkXGBRegressor(
                features_col=features_col,
                label_col=label_col,
                num_workers=params.get('num_workers', 1),
                use_gpu=params.get('use_gpu', False),
                **{k: v for k, v in params.items() if k not in ['num_workers', 'use_gpu']}
            )
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            estimator = NativeLightGBMRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numLeaves=params.get('numLeaves', 31),
                numIterations=params.get('numIterations', 100),
                learningRate=params.get('learningRate', 0.1),
                featureFraction=params.get('featureFraction', 1.0),
                baggingFraction=params.get('baggingFraction', 1.0),
                lambdaL1=params.get('lambdaL1', 0.0),
                lambdaL2=params.get('lambdaL2', 0.0),
                seed=42
            )
        else:
            raise ValueError(f"Model type {model_type} not supported or not available")
        
        return estimator

    def build_model(self, train_data: DataFrame, features_col: str, 
                   label_col: str, model_type: str, **params) -> Any:
        """
        Build a regression model.
        
        Args:
            train_data: Training DataFrame
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to build
            **params: Additional model parameters for hyperparameter optimization
            
        Returns:
            Trained model
        """
        if model_type not in self.model_types:
            # Provide more helpful error messages for advanced models
            if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
                raise ImportError(f"XGBoost not available for regression. Install with: pip install xgboost>=1.6.0")
            elif model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                raise ImportError(f"Native LightGBM not available for regression. Install with: pip install lightgbm>=4.0.0")
            else:
                available_types = list(self.model_types.keys())
                raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        print(f"Building {model_type} regression model...")
        
        # Extract hyperparameters from params (if any)
        hyperparams = {k: v for k, v in params.items() if k not in ['num_features']}
        
        if hyperparams:
            print(f"   🔧 Using optimized parameters: {hyperparams}")
            # Build model with hyperparameters using specific functions
            model = self._build_model_with_hyperparams(train_data, features_col, label_col, model_type, hyperparams)
        else:
            # Build model using default function
            model_config = self.model_types[model_type]
            build_func = model_config['build_func']
            model = build_func(train_data, features_col, label_col)
        
        print(f"✅ {model_type} regression model built successfully")
        return model
    
    def _build_model_with_hyperparams(self, train_data: DataFrame, features_col: str, 
                                     label_col: str, model_type: str, hyperparams: Dict[str, Any]) -> Any:
        """Build a model with specific hyperparameters."""
        
        if model_type == 'linear_regression':
            lr = LinearRegression(
                featuresCol=features_col,
                labelCol=label_col,
                regParam=hyperparams.get('regParam', 0.01),
                elasticNetParam=hyperparams.get('elasticNetParam', 0.0),
                maxIter=hyperparams.get('maxIter', 100)
            )
            return lr.fit(train_data)
        
        elif model_type == 'random_forest':
            rf = RandomForestRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numTrees=hyperparams.get('numTrees', 100),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 256),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
            return rf.fit(train_data)
        
        elif model_type == 'gradient_boosting':
            # Apply gradient boosting specific optimizations
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                try:
                    import sys
                    import os
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from spark_optimization_config import apply_gradient_boosting_optimizations
                    apply_gradient_boosting_optimizations(spark)
                except Exception as e:
                    print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            
            gbt = GBTRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 100),
                maxDepth=hyperparams.get('maxDepth', 6),
                maxBins=hyperparams.get('maxBins', 128),  # Reduced default from 256 to 128
                stepSize=hyperparams.get('stepSize', 0.1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
            return gbt.fit(train_data)
        
        elif model_type == 'decision_tree':
            dt = DecisionTreeRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 256),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                minInfoGain=hyperparams.get('minInfoGain', 0.0),
                seed=42
            )
            return dt.fit(train_data)
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            xgb = SparkXGBRegressor(
                features_col=features_col,
                label_col=label_col,
                max_depth=hyperparams.get('max_depth', 6),
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.3),
                subsample=hyperparams.get('subsample', 1.0),
                colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                min_child_weight=hyperparams.get('min_child_weight', 1),
                gamma=hyperparams.get('gamma', 0.0),
                seed=42
            )
            return xgb.fit(train_data)
        
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            lgb = NativeLightGBMRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numLeaves=hyperparams.get('numLeaves', 31),
                numIterations=hyperparams.get('numIterations', 100),
                learningRate=hyperparams.get('learningRate', 0.1),
                featureFraction=hyperparams.get('featureFraction', 1.0),
                baggingFraction=hyperparams.get('baggingFraction', 1.0),
                lambdaL1=hyperparams.get('lambdaL1', 0.0),
                lambdaL2=hyperparams.get('lambdaL2', 0.0),
                seed=42
            )
            return lgb.fit(train_data)
        
        # Neural Network regression not supported in PySpark
        # elif model_type == 'neural_network':
        #     raise NotImplementedError("PySpark doesn't have MultilayerPerceptronRegressor")
        
        else:
            # Fall back to default function if hyperparameter version not implemented
            model_config = self.model_types[model_type]
            build_func = model_config['build_func']
            return build_func(train_data, features_col, label_col)
    
    def save_model(self, model: Any, path: str):
        """
        Save a trained model.
        
        Args:
            model: Trained model to save
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        model.write().overwrite().save(path)
        print(f"✅ Regression model saved to {path}")
    
    def load_model(self, model_type: str, path: str) -> Any:
        """
        Load a saved model.
        
        Args:
            model_type: Type of model to load
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if model_type not in self.model_types:
            available_types = list(self.model_types.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        model_config = self.model_types[model_type]
        model_class = model_config['model_class']
        
        if model_class:
            model = model_class.load(path)
        else:
            # For XGBoost and LightGBM, use generic loading
            from pyspark.ml import PipelineModel
            model = PipelineModel.load(path)
        
        print(f"✅ {model_type} regression model loaded from {path}")
        return model
    
    def validate_model_type(self, model_type: str) -> bool:
        """
        Validate if a model type is supported.
        
        Args:
            model_type: Type of model to validate
            
        Returns:
            True if model type is supported, False otherwise
        """
        return model_type in self.model_types
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type names
        """
        return list(self.model_types.keys()) 