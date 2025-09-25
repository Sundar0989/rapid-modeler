"""
data_manipulations.py

Core data manipulation utilities for AutoML PySpark package.
Includes:
    1. Missing value calculation
    2. Variable type identification
    3. Categorical encoding
    4. Numerical imputation
    5. Column renaming
    6. Feature/target joining
    7. Train/valid/test splitting
    8. Vector assembly
    9. Feature scaling
"""

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.functions import col


def missing_value_calculation(X, miss_per=0.75):
    """
    Select columns with missing value percentage below threshold.
    """
    missing = X.select([
                F.count(F.when(F.col(c).isNull() | (F.col(c) == ''), c)).alias(c)
                if dict(X.dtypes)[c] == 'string'
                else F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c)
                for c in X.columns
             ])
    missing_len = X.count()
    
    # Use collect() instead of toPandas() to avoid Arrow conversion issues
    missing_row = missing.collect()[0]
    missing_dict = missing_row.asDict()
    
    # Create a list of variables that meet the missing value threshold
    vars_selected = []
    for col_name, missing_count in missing_dict.items():
        missing_percentage = missing_count / missing_len
        if missing_percentage <= miss_per:
            vars_selected.append(col_name)
    
    return vars_selected


def identify_variable_type(X):
    """
    Identify categorical (string) and numerical variables.
    """
    l = X.dtypes
    char_vars = []
    num_vars = []
    for i in l:
        if i[1] == 'string':
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars


def analyze_categorical_cardinality(X, char_vars, max_categorical_cardinality=50):
    """
    Analyze categorical variables and identify which ones should be treated as numeric
    due to high cardinality.
    
    Args:
        X: Input DataFrame
        char_vars: List of categorical variable names
        max_categorical_cardinality: Maximum number of unique values for a categorical variable
                                   before converting to numeric (default: 50)
    
    Returns:
        Tuple of (categorical_vars_to_keep, categorical_vars_to_convert_numeric)
    """
    if not char_vars:
        return [], [], {}
    
    print(f"🔍 Analyzing cardinality for {len(char_vars)} categorical variables...")
    
    # Batch cardinality and missing value analysis for all variables at once
    from pyspark.sql.functions import approx_count_distinct, col, sum as spark_sum, when, isnan
    
    print(f"📊 Computing cardinality and missing values for all {len(char_vars)} variables in batch operations...")
    
    # 1. Batch cardinality analysis with higher precision (rsd=0.01)
    cardinality_agg = X.agg(*(approx_count_distinct(col(c), rsd=0.01).alias(f"{c}_unique") for c in char_vars))
    cardinality_results = cardinality_agg.collect()[0].asDict()
    
    # 2. Batch missing value analysis
    missing_agg = X.agg(*(
        spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(f"{c}_missing")
        for c in char_vars
    ))
    missing_results = missing_agg.collect()[0].asDict()
    
    total_count = X.count()
    
    categorical_vars_to_keep = []
    categorical_vars_to_drop = []
    
    # Process results
    for var in char_vars:
        unique_count = cardinality_results[f"{var}_unique"]
        
        if unique_count > max_categorical_cardinality:
            print(f"   📊 {var}: ~{unique_count} unique values → DROPPING (exceeds threshold of {max_categorical_cardinality})")
            categorical_vars_to_drop.append(var)
        else:
            print(f"   📊 {var}: ~{unique_count} unique values → Keeping as CATEGORICAL")
            categorical_vars_to_keep.append(var)
    
    if categorical_vars_to_drop:
        print(f"🗑️ Dropping {len(categorical_vars_to_drop)} high-cardinality categorical variables")
        print(f"✅ Keeping {len(categorical_vars_to_keep)} low-cardinality categorical variables")
    else:
        print(f"✅ All categorical variables have acceptable cardinality (≤ {max_categorical_cardinality})")
    
    # Return both the variable lists and the computed statistics for reuse
    cardinality_stats = {}
    for var in char_vars:
        unique_count = cardinality_results[f"{var}_unique"]
        missing_count = missing_results[f"{var}_missing"]
        missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
        
        cardinality_stats[var] = {
            'unique_count': unique_count,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'total_count': total_count
        }
    
    return categorical_vars_to_keep, categorical_vars_to_drop, cardinality_stats


def remove_high_cardinality_categoricals(X, categorical_vars_to_remove):
    """
    Remove high-cardinality categorical variables from the dataset.
    
    High-cardinality categorical variables (>50 unique values) are removed to prevent
    curse of dimensionality and improve model performance. This keeps the approach
    simple and reliable.
    
    Args:
        X: Input DataFrame
        categorical_vars_to_remove: List of high-cardinality categorical variable names to remove
    
    Returns:
        DataFrame with high-cardinality categorical variables removed
    """
    if not categorical_vars_to_remove:
        return X
    
    print(f"🗑️ Removing {len(categorical_vars_to_remove)} high-cardinality categorical variables...")
    print(f"💡 These variables have >50 unique values and would create too many features after encoding")
    
    # Drop all high-cardinality columns at once (much more efficient)
    if categorical_vars_to_remove:
        print(f"   Dropping columns: {', '.join(categorical_vars_to_remove[:5])}{'...' if len(categorical_vars_to_remove) > 5 else ''}")
        X = X.drop(*categorical_vars_to_remove)
    
    print(f"✅ High-cardinality categorical variables removed from dataset")
    print(f"📊 This prevents curse of dimensionality and improves model performance")
    return X


def categorical_to_index(X, char_vars):
    """
    Encode categorical variables using StringIndexer.
    Returns transformed DataFrame and fitted PipelineModel.
    """
    if not char_vars:
        return X, None
    chars = X.select(char_vars)
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_encoded", handleInvalid="keep") for column in chars.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(chars)
    X = char_labels.transform(X)
    return X, char_labels


def numerical_imputation(X, num_vars, impute_with=0):
    """
    Impute missing values in numerical columns.
    """
    X = X.fillna(impute_with, subset=num_vars)
    return X


def rename_columns(X, char_vars):
    """
    Rename indexed columns to encoded format for VectorAssembler compatibility.
    """
    # Rename _index to _encoded to match VectorAssembler expectations
    for var in char_vars:
        if f"{var}_index" in X.columns:
            X = X.withColumnRenamed(f"{var}_index", f"{var}_encoded")
    return X


def join_features_and_target(X, Y):
    """
    Join features and target DataFrames on a generated row id.
    """
    X = X.withColumn('id', F.monotonically_increasing_id())
    Y = Y.withColumn('id', F.monotonically_increasing_id())
    joinedDF = X.join(Y, 'id', 'inner').drop('id')
    return joinedDF


def train_valid_test_split(df, train_size=0.4, valid_size=0.3, seed=12345):
    """
    Split DataFrame into train, valid, and test sets.
    """
    train, valid, test = df.randomSplit([train_size, valid_size, 1 - train_size - valid_size], seed=seed)
    return train, valid, test


def assembled_vectors(df, list_of_features_to_scale, target_column_name):
    """
    Assemble feature columns into a single vector column.
    """
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='features')
    pipeline = Pipeline(stages=[assembler])
    assembleModel = pipeline.fit(df)
    selectedCols = [target_column_name, 'features'] + list_of_features_to_scale
    df = assembleModel.transform(df).select(selectedCols)
    return df


def scaled_dataframes(train, valid, test, list_of_features_to_scale, target_column_name):
    """
    Scale features in train, valid, and test DataFrames.
    Returns transformed DataFrames and fitted PipelineModel.
    """
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='assembled_features')
    scaler = StandardScaler(inputCol='assembled_features', outputCol='features')
    pipeline = Pipeline(stages=[assembler, scaler])
    pipelineModel = pipeline.fit(train)
    selectedCols = [target_column_name, 'features'] + list_of_features_to_scale
    train = pipelineModel.transform(train).select(selectedCols)
    valid = pipelineModel.transform(valid).select(selectedCols)
    test = pipelineModel.transform(test).select(selectedCols)
    return train, valid, test, pipelineModel 