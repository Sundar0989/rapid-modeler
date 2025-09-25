#!/usr/bin/env python3
"""
Analyze differences between datasets from feature profiling results.
Provides focused comparison of key metrics across train, oot1, and oot2 datasets.
"""

import json
import pandas as pd
import os

def load_profiling_results(json_file):
    """Load feature profiling results from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_file}: {e}")
        return None

def analyze_dataset_differences(results_file):
    """Analyze differences between datasets."""
    print("🔍 DATASET DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    results = load_profiling_results(results_file)
    if not results:
        return
    
    # Extract dataset information
    datasets = {}
    for dataset_name, data in results.items():
        if isinstance(data, dict) and 'dataset_name' in data:
            datasets[data['dataset_name']] = data
    
    if not datasets:
        print("❌ No dataset information found in results")
        return
    
    print(f"📊 Found {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Compare basic statistics
    print("\n📈 DATASET SIZE COMPARISON")
    print("-" * 30)
    for name, data in datasets.items():
        size = data.get('dataset_size', 'Unknown')
        print(f"{name:10}: {size:,} rows" if isinstance(size, int) else f"{name:10}: {size}")
    
    # Compare feature statistics
    if 'train' in datasets:
        train_data = datasets['train']
        univariate = train_data.get('univariate_analysis', {})
        
        print(f"\n🎯 FEATURE ANALYSIS (based on training data)")
        print("-" * 40)
        
        # Numerical features comparison
        numerical_features = []
        categorical_features = []
        
        for feature, stats in univariate.items():
            data_type = stats.get('data_type', 'unknown')
            if data_type in ['integer', 'long', 'float', 'double', 'decimal']:
                numerical_features.append(feature)
            else:
                categorical_features.append(feature)
        
        print(f"📊 Numerical features: {len(numerical_features)}")
        print(f"📊 Categorical features: {len(categorical_features)}")
        
        # Show top numerical features by variance
        if numerical_features:
            print(f"\n📈 TOP NUMERICAL FEATURES (by standard deviation):")
            num_stats = []
            for feature in numerical_features:
                stats = univariate.get(feature, {})
                std = stats.get('std', 0)
                mean = stats.get('mean', 0)
                if std is not None and mean is not None:
                    cv = (std / abs(mean)) if mean != 0 else 0  # Coefficient of variation
                    num_stats.append((feature, std, mean, cv))
            
            # Sort by standard deviation
            num_stats.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
            
            for i, (feature, std, mean, cv) in enumerate(num_stats[:5]):
                print(f"   {i+1:2}. {feature:20} | std: {std:8.2f} | mean: {mean:8.2f} | cv: {cv:6.3f}")
        
        # Show categorical features by cardinality
        if categorical_features:
            print(f"\n📊 CATEGORICAL FEATURES (by cardinality):")
            cat_stats = []
            for feature in categorical_features:
                stats = univariate.get(feature, {})
                unique_count = stats.get('unique_count', 0)
                cat_stats.append((feature, unique_count))
            
            # Sort by unique count
            cat_stats.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, unique_count) in enumerate(cat_stats[:5]):
                print(f"   {i+1:2}. {feature:20} | unique values: {unique_count:4}")
    
    # Compare missing values across datasets
    print(f"\n❓ MISSING VALUES COMPARISON")
    print("-" * 30)
    
    if len(datasets) > 1:
        # Get common features
        all_features = set()
        for data in datasets.values():
            univariate = data.get('univariate_analysis', {})
            all_features.update(univariate.keys())
        
        # Compare missing percentages
        missing_comparison = []
        for feature in sorted(all_features):
            row = {'feature': feature}
            for dataset_name, data in datasets.items():
                univariate = data.get('univariate_analysis', {})
                stats = univariate.get(feature, {})
                missing_pct = stats.get('missing_percentage', 0)
                row[dataset_name] = missing_pct
            missing_comparison.append(row)
        
        # Show features with significant missing value differences
        print("Features with >5% missing value difference between datasets:")
        for row in missing_comparison:
            values = [row[name] for name in datasets.keys() if name in row]
            if len(values) > 1:
                max_diff = max(values) - min(values)
                if max_diff > 5:  # More than 5% difference
                    print(f"   {row['feature']:20} | ", end="")
                    for name in datasets.keys():
                        if name in row:
                            print(f"{name}: {row[name]:5.1f}% | ", end="")
                    print(f"diff: {max_diff:5.1f}%")
    
    # Data quality summary
    print(f"\n✅ DATA QUALITY SUMMARY")
    print("-" * 25)
    for name, data in datasets.items():
        quality = data.get('data_quality', {})
        duplicates = quality.get('duplicate_rows', 0)
        total_missing = quality.get('total_missing_values', 0)
        print(f"{name:10}: {duplicates:,} duplicates, {total_missing:,} missing values")

def main():
    """Main function to analyze dataset differences."""
    
    # Look for recent profiling results
    result_files = [
        "all_datasets_feature_profiling.json",
        "bank_all_datasets_feature_profiling.json", 
        "lightgbm_fix_test_all_datasets_feature_profiling.json"
    ]
    
    found_file = None
    for file in result_files:
        if os.path.exists(file):
            found_file = file
            break
    
    if found_file:
        print(f"📁 Using results file: {found_file}")
        analyze_dataset_differences(found_file)
    else:
        print("❌ No feature profiling results found.")
        print("Available files to analyze:")
        for file in result_files:
            print(f"   - {file}")
        print("\nRun feature profiling first to generate results.")

if __name__ == "__main__":
    main()
