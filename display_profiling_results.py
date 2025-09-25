#!/usr/bin/env python3
"""
Display feature profiling results in a table format for easy comparison.
Shows side-by-side statistics from different datasets.
"""

import json
import pandas as pd
import os
from tabulate import tabulate

def load_profiling_results(json_file):
    """Load feature profiling results from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_file}: {e}")
        return None

def display_univariate_comparison(results):
    """Display univariate analysis results in table format."""
    print("\n📊 UNIVARIATE ANALYSIS COMPARISON")
    print("=" * 80)
    
    # Extract datasets
    datasets = {}
    for dataset_name, data in results.items():
        if isinstance(data, dict) and 'dataset_name' in data:
            datasets[data['dataset_name']] = data
    
    if not datasets:
        print("❌ No dataset information found")
        return
    
    # Get all features from the first dataset
    first_dataset = list(datasets.values())[0]
    univariate = first_dataset.get('univariate_analysis', {})
    
    if not univariate:
        print("❌ No univariate analysis found")
        return
    
    # Prepare comparison table
    comparison_data = []
    
    for feature, stats in univariate.items():
        row = {'Feature': feature}
        
        # Add data type
        data_type = stats.get('data_type', 'unknown')
        row['Type'] = data_type
        
        # Add statistics for each dataset
        for dataset_name in sorted(datasets.keys()):
            dataset_data = datasets[dataset_name]
            dataset_univariate = dataset_data.get('univariate_analysis', {})
            feature_stats = dataset_univariate.get(feature, {})
            
            if data_type in ['integer', 'long', 'float', 'double', 'decimal']:
                # Numerical feature
                mean = feature_stats.get('mean')
                std = feature_stats.get('std')
                missing_pct = feature_stats.get('missing_percentage', 0)
                
                if mean is not None and std is not None:
                    row[f'{dataset_name}_Mean'] = f"{mean:.2f}"
                    row[f'{dataset_name}_Std'] = f"{std:.2f}"
                else:
                    row[f'{dataset_name}_Mean'] = "N/A"
                    row[f'{dataset_name}_Std'] = "N/A"
                
                row[f'{dataset_name}_Missing%'] = f"{missing_pct:.1f}%"
            else:
                # Categorical feature
                unique_count = feature_stats.get('unique_count', 0)
                missing_pct = feature_stats.get('missing_percentage', 0)
                
                row[f'{dataset_name}_Unique'] = unique_count
                row[f'{dataset_name}_Missing%'] = f"{missing_pct:.1f}%"
        
        comparison_data.append(row)
    
    # Create DataFrame and display
    df = pd.DataFrame(comparison_data)
    
    # Separate numerical and categorical features
    numerical_df = df[df['Type'].isin(['integer', 'long', 'float', 'double', 'decimal'])]
    categorical_df = df[~df['Type'].isin(['integer', 'long', 'float', 'double', 'decimal'])]
    
    if not numerical_df.empty:
        print("\n📈 NUMERICAL FEATURES")
        print("-" * 50)
        print(tabulate(numerical_df, headers='keys', tablefmt='grid', showindex=False))
    
    if not categorical_df.empty:
        print("\n📊 CATEGORICAL FEATURES")
        print("-" * 50)
        print(tabulate(categorical_df, headers='keys', tablefmt='grid', showindex=False))

def display_dataset_summary(results):
    """Display dataset summary information."""
    print("\n📋 DATASET SUMMARY")
    print("=" * 40)
    
    # Extract datasets
    datasets = {}
    for dataset_name, data in results.items():
        if isinstance(data, dict) and 'dataset_name' in data:
            datasets[data['dataset_name']] = data
    
    summary_data = []
    for dataset_name, data in datasets.items():
        row = {
            'Dataset': dataset_name,
            'Rows': f"{data.get('dataset_size', 'Unknown'):,}" if isinstance(data.get('dataset_size'), int) else str(data.get('dataset_size', 'Unknown')),
            'Features': len(data.get('univariate_analysis', {})),
            'Duplicates': data.get('data_quality', {}).get('duplicate_rows', 0),
            'Total_Missing': data.get('data_quality', {}).get('total_missing_values', 0)
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def display_correlation_comparison(results):
    """Display correlation analysis comparison."""
    print("\n🔗 CORRELATION WITH TARGET")
    print("=" * 50)
    
    # Extract datasets
    datasets = {}
    for dataset_name, data in results.items():
        if isinstance(data, dict) and 'dataset_name' in data:
            datasets[data['dataset_name']] = data
    
    # Get correlation data
    correlation_data = []
    
    # Use first dataset to get feature list
    first_dataset = list(datasets.values())[0]
    correlation_matrix = first_dataset.get('correlation_matrix', {})
    correlation_strengths = correlation_matrix.get('correlation_strengths', {})
    
    for feature, strength in correlation_strengths.items():
        row = {'Feature': feature}
        
        for dataset_name in sorted(datasets.keys()):
            dataset_data = datasets[dataset_name]
            dataset_correlation = dataset_data.get('correlation_matrix', {})
            dataset_strengths = dataset_correlation.get('correlation_strengths', {})
            
            feature_strength = dataset_strengths.get(feature, 0)
            row[f'{dataset_name}_Correlation'] = f"{feature_strength:.3f}"
        
        correlation_data.append(row)
    
    if correlation_data:
        # Sort by correlation strength (using first dataset)
        correlation_data.sort(key=lambda x: abs(float(x[f'{sorted(datasets.keys())[0]}_Correlation'])), reverse=True)
        
        df = pd.DataFrame(correlation_data)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("❌ No correlation data found")

def main():
    """Main function to display profiling results."""
    
    # Look for recent profiling results
    result_files = [
        "lightgbm_fix_test_all_datasets_feature_profiling.json",
        "all_datasets_feature_profiling.json",
        "bank_all_datasets_feature_profiling.json"
    ]
    
    found_file = None
    for file in result_files:
        if os.path.exists(file):
            found_file = file
            break
    
    if not found_file:
        print("❌ No feature profiling results found.")
        print("Available files to display:")
        for file in result_files:
            print(f"   - {file}")
        print("\nRun feature profiling first to generate results.")
        return
    
    print(f"📁 Loading results from: {found_file}")
    results = load_profiling_results(found_file)
    
    if not results:
        return
    
    # Display all sections
    display_dataset_summary(results)
    display_univariate_comparison(results)
    display_correlation_comparison(results)
    
    print(f"\n✅ Results displayed from: {found_file}")
    print(f"📊 Use this for side-by-side comparison of datasets")

if __name__ == "__main__":
    main()
