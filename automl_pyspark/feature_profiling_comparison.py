#!/usr/bin/env python3
"""
Feature Profiling Comparison Tool

This module provides functionality to compare feature profiling results 
from different datasets side-by-side, enabling easy comparison of:
- Feature distributions
- Data quality metrics
- Correlation patterns
- Statistical summaries

Usage:
    from feature_profiling_comparison import FeatureProfilingComparator
    
    comparator = FeatureProfilingComparator()
    comparator.add_dataset("Dataset A", profiling_results_a)
    comparator.add_dataset("Dataset B", profiling_results_b)
    comparator.generate_comparison_report("comparison_report.html")
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class FeatureProfilingComparator:
    """
    Compare feature profiling results from multiple datasets side-by-side.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.datasets = {}
        self.comparison_results = {}
    
    def add_dataset(self, name: str, profiling_results: Dict[str, Any]):
        """
        Add a dataset's profiling results for comparison.
        
        Args:
            name: Name/label for the dataset
            profiling_results: Results from FeatureProfiler.profile_features()
        """
        self.datasets[name] = profiling_results
        print(f"✅ Added dataset '{name}' with {len(profiling_results.get('selected_features', []))} features")
    
    def load_dataset_from_json(self, name: str, json_path: str):
        """
        Load dataset profiling results from JSON file.
        
        Args:
            name: Name/label for the dataset
            json_path: Path to JSON file containing profiling results
        """
        with open(json_path, 'r') as f:
            profiling_results = json.load(f)
        self.add_dataset(name, profiling_results)
    
    def compare_datasets(self) -> Dict[str, Any]:
        """
        Perform comprehensive comparison between datasets.
        
        Returns:
            Dictionary containing comparison results
        """
        if len(self.datasets) < 2:
            raise ValueError("Need at least 2 datasets for comparison")
        
        comparison = {
            'dataset_names': list(self.datasets.keys()),
            'feature_comparison': self._compare_features(),
            'correlation_comparison': self._compare_correlations(),
            'data_quality_comparison': self._compare_data_quality(),
            'statistical_comparison': self._compare_statistics()
        }
        
        self.comparison_results = comparison
        return comparison
    
    def _compare_features(self) -> Dict[str, Any]:
        """Compare features across datasets."""
        all_features = set()
        dataset_features = {}
        
        # Collect all features from all datasets
        for name, data in self.datasets.items():
            features = set(data.get('selected_features', []))
            dataset_features[name] = features
            all_features.update(features)
        
        # Find common and unique features
        common_features = set.intersection(*dataset_features.values()) if dataset_features else set()
        unique_features = {}
        
        for name, features in dataset_features.items():
            unique_features[name] = features - common_features
        
        return {
            'all_features': sorted(list(all_features)),
            'common_features': sorted(list(common_features)),
            'unique_features': {name: sorted(list(features)) for name, features in unique_features.items()},
            'feature_counts': {name: len(features) for name, features in dataset_features.items()}
        }
    
    def _compare_correlations(self) -> Dict[str, Any]:
        """Compare correlation patterns across datasets."""
        correlation_comparison = {}
        
        for name, data in self.datasets.items():
            corr_matrix = data.get('correlation_matrix', {})
            correlations = corr_matrix.get('correlation_ranking', [])
            
            correlation_comparison[name] = {
                'top_correlations': correlations[:10],
                'correlation_strengths': corr_matrix.get('correlation_strengths', {}),
                'multicollinearity_warning': corr_matrix.get('multicollinearity_warning', False)
            }
        
        # Find features with significantly different correlations
        correlation_differences = self._find_correlation_differences()
        
        return {
            'by_dataset': correlation_comparison,
            'correlation_differences': correlation_differences
        }
    
    def _compare_data_quality(self) -> Dict[str, Any]:
        """Compare data quality metrics across datasets."""
        quality_comparison = {}
        
        for name, data in self.datasets.items():
            univariate = data.get('univariate_analysis', {})
            quality = data.get('data_quality', {})
            
            # Calculate quality metrics
            total_features = len(univariate)
            features_with_missing = sum(1 for stats in univariate.values() 
                                      if stats.get('missing_percentage', 0) > 0)
            avg_missing_percentage = sum(stats.get('missing_percentage', 0) 
                                       for stats in univariate.values()) / total_features if total_features > 0 else 0
            
            quality_comparison[name] = {
                'total_features': total_features,
                'features_with_missing': features_with_missing,
                'avg_missing_percentage': avg_missing_percentage,
                'total_missing_values': quality.get('total_missing_values', 0),
                'duplicate_rows': quality.get('duplicate_rows', 0)
            }
        
        return quality_comparison
    
    def _compare_statistics(self) -> Dict[str, Any]:
        """Compare statistical summaries for common features."""
        common_features = self.comparison_results.get('feature_comparison', {}).get('common_features', [])
        if not common_features:
            common_features = self._compare_features()['common_features']
        
        statistical_comparison = {}
        
        for feature in common_features:
            feature_stats = {}
            
            for name, data in self.datasets.items():
                univariate = data.get('univariate_analysis', {})
                if feature in univariate:
                    stats = univariate[feature]
                    feature_stats[name] = {
                        'count': stats.get('count', 0),
                        'missing_percentage': stats.get('missing_percentage', 0),
                        'unique_count': stats.get('unique_count', 0),
                        'mean': stats.get('mean'),
                        'std': stats.get('std'),
                        'min': stats.get('min'),
                        'max': stats.get('max'),
                        'data_type': stats.get('data_type', 'unknown')
                    }
            
            if feature_stats:
                statistical_comparison[feature] = feature_stats
        
        return statistical_comparison
    
    def _find_correlation_differences(self) -> Dict[str, Any]:
        """Find features with significantly different correlations across datasets."""
        differences = {}
        
        # Get common features
        common_features = set.intersection(*[
            set(data.get('selected_features', [])) 
            for data in self.datasets.values()
        ])
        
        for feature in common_features:
            correlations = {}
            
            for name, data in self.datasets.items():
                corr_strengths = data.get('correlation_matrix', {}).get('correlation_strengths', {})
                if feature in corr_strengths:
                    correlations[name] = corr_strengths[feature].get('correlation', 0)
            
            if len(correlations) >= 2:
                corr_values = list(correlations.values())
                max_diff = max(corr_values) - min(corr_values)
                
                if max_diff > 0.1:  # Significant difference threshold
                    differences[feature] = {
                        'correlations': correlations,
                        'max_difference': max_diff,
                        'range': [min(corr_values), max(corr_values)]
                    }
        
        return differences
    
    def generate_comparison_report(self, output_path: str = "feature_profiling_comparison.json"):
        """
        Generate comprehensive JSON comparison report for Streamlit integration.
        
        Args:
            output_path: Path for the output JSON file
        """
        if not self.datasets:
            print("❌ No datasets added for comparison")
            return
        
        if len(self.datasets) < 2:
            print("❌ Need at least 2 datasets for comparison")
            return
        
        # Perform comparison
        comparison = self.compare_datasets()
        
        # Save comparison results as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✅ Feature profiling comparison results saved to: {output_path}")
        print(f"📊 Ready for Streamlit integration")
        
        return comparison
    
    def _generate_comparison_html(self, comparison: Dict[str, Any]) -> str:
        """Generate HTML content for comparison report."""
        dataset_names = comparison['dataset_names']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feature Profiling Comparison Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1600px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                
                .dataset-badges {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }}
                
                .dataset-badge {{
                    background: rgba(255,255,255,0.2);
                    padding: 10px 20px;
                    border-radius: 25px;
                    font-weight: 600;
                    backdrop-filter: blur(10px);
                }}
                
                .comparison-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 30px;
                    padding: 30px;
                }}
                
                .comparison-card {{
                    background: white;
                    border-radius: 12px;
                    padding: 25px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                    border: 1px solid #e9ecef;
                }}
                
                .card-title {{
                    font-size: 1.5em;
                    color: #2c3e50;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #3498db;
                }}
                
                .side-by-side {{
                    display: grid;
                    grid-template-columns: repeat({len(dataset_names)}, 1fr);
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .dataset-column {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                
                .dataset-name {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                    font-size: 1.1em;
                }}
                
                .metric-row {{
                    display: flex;
                    justify-content: space-between;
                    margin: 8px 0;
                    padding: 5px 0;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .metric-label {{
                    font-weight: 600;
                    color: #495057;
                }}
                
                .metric-value {{
                    color: #2c3e50;
                    font-weight: 500;
                }}
                
                .difference-highlight {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                }}
                
                .chart-container {{
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                }}
                
                .feature-list {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin: 10px 0;
                }}
                
                .feature-tag {{
                    background: #e3f2fd;
                    color: #1976d2;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 500;
                }}
                
                .feature-tag.unique {{
                    background: #fff3e0;
                    color: #f57c00;
                }}
                
                .correlation-comparison {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }}
                
                .correlation-item {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                }}
                
                .correlation-item.negative {{
                    border-left-color: #dc3545;
                }}
                
                @media (max-width: 768px) {{
                    .comparison-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .side-by-side {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Feature Profiling Comparison</h1>
                    <div class="dataset-badges">
                        {self._generate_dataset_badges(dataset_names)}
                    </div>
                </div>
                
                <div class="comparison-grid">
                    {self._generate_feature_comparison_card(comparison['feature_comparison'])}
                    {self._generate_data_quality_comparison_card(comparison['data_quality_comparison'])}
                    {self._generate_correlation_comparison_card(comparison['correlation_comparison'])}
                    {self._generate_statistical_comparison_card(comparison['statistical_comparison'])}
                </div>
            </div>
            
            <script>
                // Add any interactive charts here
                console.log('Comparison report loaded');
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_dataset_badges(self, dataset_names: List[str]) -> str:
        """Generate HTML for dataset badges."""
        return ''.join([f'<div class="dataset-badge">{name}</div>' for name in dataset_names])
    
    def _generate_feature_comparison_card(self, feature_comparison: Dict[str, Any]) -> str:
        """Generate feature comparison card HTML."""
        common_features = feature_comparison['common_features']
        unique_features = feature_comparison['unique_features']
        feature_counts = feature_comparison['feature_counts']
        
        html = f"""
        <div class="comparison-card">
            <h3 class="card-title">🔍 Feature Comparison</h3>
            
            <div class="side-by-side">
        """
        
        for dataset_name in feature_counts.keys():
            html += f"""
                <div class="dataset-column">
                    <div class="dataset-name">{dataset_name}</div>
                    <div class="metric-row">
                        <span class="metric-label">Total Features:</span>
                        <span class="metric-value">{feature_counts[dataset_name]}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Common Features:</span>
                        <span class="metric-value">{len(common_features)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Unique Features:</span>
                        <span class="metric-value">{len(unique_features.get(dataset_name, []))}</span>
                    </div>
                </div>
            """
        
        html += """
            </div>
            
            <h4>Common Features ({} total)</h4>
            <div class="feature-list">
        """.format(len(common_features))
        
        for feature in common_features[:20]:  # Show first 20
            html += f'<span class="feature-tag">{feature}</span>'
        
        if len(common_features) > 20:
            html += f'<span class="feature-tag">... and {len(common_features) - 20} more</span>'
        
        html += '</div>'
        
        # Show unique features for each dataset
        for dataset_name, features in unique_features.items():
            if features:
                html += f"""
                <h4>Unique to {dataset_name} ({len(features)} total)</h4>
                <div class="feature-list">
                """
                
                for feature in features[:10]:  # Show first 10
                    html += f'<span class="feature-tag unique">{feature}</span>'
                
                if len(features) > 10:
                    html += f'<span class="feature-tag unique">... and {len(features) - 10} more</span>'
                
                html += '</div>'
        
        html += '</div>'
        return html
    
    def _generate_data_quality_comparison_card(self, quality_comparison: Dict[str, Any]) -> str:
        """Generate data quality comparison card HTML."""
        html = f"""
        <div class="comparison-card">
            <h3 class="card-title">🔍 Data Quality Comparison</h3>
            
            <div class="side-by-side">
        """
        
        for dataset_name, quality in quality_comparison.items():
            html += f"""
                <div class="dataset-column">
                    <div class="dataset-name">{dataset_name}</div>
                    <div class="metric-row">
                        <span class="metric-label">Total Features:</span>
                        <span class="metric-value">{quality['total_features']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Features with Missing:</span>
                        <span class="metric-value">{quality['features_with_missing']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Avg Missing %:</span>
                        <span class="metric-value">{quality['avg_missing_percentage']:.2f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Missing Values:</span>
                        <span class="metric-value">{quality['total_missing_values']:,}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Duplicate Rows:</span>
                        <span class="metric-value">{quality['duplicate_rows']:,}</span>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_correlation_comparison_card(self, correlation_comparison: Dict[str, Any]) -> str:
        """Generate correlation comparison card HTML."""
        differences = correlation_comparison.get('correlation_differences', {})
        
        html = f"""
        <div class="comparison-card">
            <h3 class="card-title">📊 Correlation Comparison</h3>
            
            <h4>Features with Significant Correlation Differences</h4>
        """
        
        if differences:
            for feature, diff_info in list(differences.items())[:10]:  # Show top 10
                correlations = diff_info['correlations']
                max_diff = diff_info['max_difference']
                
                html += f"""
                <div class="difference-highlight">
                    <strong>{feature}</strong> (Max difference: {max_diff:.3f})
                    <div class="correlation-comparison">
                """
                
                for dataset_name, correlation in correlations.items():
                    color_class = 'negative' if correlation < 0 else ''
                    html += f"""
                        <div class="correlation-item {color_class}">
                            <strong>{dataset_name}:</strong> {correlation:.3f}
                        </div>
                    """
                
                html += """
                    </div>
                </div>
                """
        else:
            html += '<p>No significant correlation differences found between datasets.</p>'
        
        html += '</div>'
        return html
    
    def _generate_statistical_comparison_card(self, statistical_comparison: Dict[str, Any]) -> str:
        """Generate statistical comparison card HTML."""
        html = f"""
        <div class="comparison-card">
            <h3 class="card-title">📈 Statistical Summary Comparison</h3>
        """
        
        # Show comparison for first few common features
        features_to_show = list(statistical_comparison.keys())[:5]
        
        for feature in features_to_show:
            feature_stats = statistical_comparison[feature]
            
            html += f"""
            <h4>{feature}</h4>
            <div class="side-by-side">
            """
            
            for dataset_name, stats in feature_stats.items():
                html += f"""
                <div class="dataset-column">
                    <div class="dataset-name">{dataset_name}</div>
                    <div class="metric-row">
                        <span class="metric-label">Count:</span>
                        <span class="metric-value">{stats['count']:,}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Missing %:</span>
                        <span class="metric-value">{stats['missing_percentage']:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Unique:</span>
                        <span class="metric-value">{stats['unique_count']:,}</span>
                    </div>
                """
                
                if stats['mean'] is not None:
                    html += f"""
                    <div class="metric-row">
                        <span class="metric-label">Mean:</span>
                        <span class="metric-value">{stats['mean']:.2f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Std:</span>
                        <span class="metric-value">{f"{stats['std']:.2f}" if stats['std'] else 'N/A'}</span>
                    </div>
                    """
                
                html += '</div>'
            
            html += '</div>'
        
        if len(statistical_comparison) > 5:
            html += f'<p><em>... and {len(statistical_comparison) - 5} more features</em></p>'
        
        html += '</div>'
        return html


if __name__ == "__main__":
    print("🔍 Feature Profiling Comparison Tool")
    print("This module provides side-by-side comparison of feature profiling results.")
    print("Use FeatureProfilingComparator to compare multiple datasets.")
