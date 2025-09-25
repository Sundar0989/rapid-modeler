#!/usr/bin/env python3
"""
Variable Tracker for AutoML Pipeline
Tracks the lifecycle of variables through each processing step.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

class VariableTracker:
    """
    Tracks variables through the AutoML pipeline and generates detailed reports.
    """
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize the variable tracker.
        
        Args:
            task_type: Type of ML task ("classification" or "regression")
        """
        self.task_type = task_type
        self.variables = {}  # Dict to store variable information
        self.steps = []      # List to track processing steps
        self.current_step = 0
        
        # Initialize tracking
        self.reset()
    
    def reset(self):
        """Reset the tracker for a new run."""
        self.variables = {}
        self.steps = []
        self.current_step = 0
        
    def initialize_variables(self, all_columns: List[str], target_column: str):
        """
        Initialize tracking for all variables in the dataset.
        
        Args:
            all_columns: List of all column names in the dataset
            target_column: Name of the target column
        """
        self.target_column = target_column
        
        for col in all_columns:
            self.variables[col] = {
                'original_name': col,
                'current_status': 'active' if col != target_column else 'target',
                'data_type': 'unknown',
                'variable_type': 'unknown',  # categorical, numerical, target, excluded
                'drop_step': None,
                'drop_reason': None,
                'final_selected': False,
                'steps_history': []
            }
            
        # Add initial step
        self.add_step("initialization", f"Dataset loaded with {len(all_columns)} columns")
        
        print(f"📊 Variable Tracker initialized: {len(all_columns)} variables (including target: {target_column})")
    
    def add_step(self, step_name: str, description: str):
        """
        Add a processing step to the tracker.
        
        Args:
            step_name: Name of the processing step
            description: Description of what happened in this step
        """
        self.current_step += 1
        step_info = {
            'step_number': self.current_step,
            'step_name': step_name,
            'description': description,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.steps.append(step_info)
        
        print(f"📝 Step {self.current_step}: {step_name} - {description}")
    
    def update_variable_types(self, categorical_vars: List[str], numerical_vars: List[str]):
        """
        Update variable types after data type identification.
        
        Args:
            categorical_vars: List of categorical variable names
            numerical_vars: List of numerical variable names
        """
        self.add_step("type_identification", f"Identified {len(categorical_vars)} categorical and {len(numerical_vars)} numerical variables")
        
        for var_name in self.variables:
            if var_name == self.target_column:
                self.variables[var_name]['variable_type'] = 'target'
                self.variables[var_name]['data_type'] = 'target'
            elif var_name in categorical_vars:
                self.variables[var_name]['variable_type'] = 'categorical'
                self.variables[var_name]['data_type'] = 'categorical'
            elif var_name in numerical_vars:
                self.variables[var_name]['variable_type'] = 'numerical'
                self.variables[var_name]['data_type'] = 'numerical'
            else:
                # Variable not in either list - might be excluded
                if self.variables[var_name]['current_status'] == 'active':
                    self.drop_variable(var_name, "type_identification", "Not identified as categorical or numerical")
    
    def drop_variable(self, var_name: str, step_name: str, reason: str):
        """
        Mark a variable as dropped.
        
        Args:
            var_name: Name of the variable to drop
            step_name: Name of the step where variable was dropped
            reason: Reason for dropping the variable
        """
        if var_name in self.variables and self.variables[var_name]['current_status'] == 'active':
            self.variables[var_name]['current_status'] = 'dropped'
            self.variables[var_name]['drop_step'] = step_name
            self.variables[var_name]['drop_reason'] = reason
            self.variables[var_name]['steps_history'].append({
                'step': self.current_step,
                'step_name': step_name,
                'action': 'dropped',
                'reason': reason
            })
            
            print(f"❌ Variable '{var_name}' dropped at {step_name}: {reason}")
    
    def drop_variables_batch(self, var_names: List[str], step_name: str, reason: str):
        """
        Drop multiple variables at once.
        
        Args:
            var_names: List of variable names to drop
            step_name: Name of the step where variables were dropped
            reason: Reason for dropping the variables
        """
        dropped_count = 0
        for var_name in var_names:
            if var_name in self.variables and self.variables[var_name]['current_status'] == 'active':
                self.drop_variable(var_name, step_name, reason)
                dropped_count += 1
        
        if dropped_count > 0:
            self.add_step(step_name, f"Dropped {dropped_count} variables: {reason}")
    
    def filter_by_cardinality(self, high_cardinality_vars: List[str], threshold: int = 200):
        """
        Track variables dropped due to high cardinality.
        
        Args:
            high_cardinality_vars: List of high cardinality variable names
            threshold: Cardinality threshold used
        """
        self.drop_variables_batch(
            high_cardinality_vars, 
            "cardinality_filtering", 
            f"High cardinality (>{threshold} unique values)"
        )
    
    def filter_by_missing_values(self, high_missing_vars: List[str], threshold: float = 0.8):
        """
        Track variables dropped due to high missing values.
        
        Args:
            high_missing_vars: List of variable names with high missing values
            threshold: Missing value threshold used (as percentage)
        """
        self.drop_variables_batch(
            high_missing_vars, 
            "missing_value_filtering", 
            f"High missing values (>{threshold*100}%)"
        )
    
    def filter_by_date_columns(self, date_vars: List[str]):
        """
        Track variables dropped because they are date/timestamp columns.
        
        Args:
            date_vars: List of date/timestamp variable names
        """
        self.drop_variables_batch(
            date_vars, 
            "date_column_filtering", 
            "Date/timestamp column (automatically excluded)"
        )
    
    def feature_selection_result(self, selected_vars: List[str], total_available: int, method: str = "feature_selection"):
        """
        Track the result of feature selection.
        
        Args:
            selected_vars: List of variables selected by feature selection
            total_available: Total number of variables available for selection
            method: Feature selection method used
        """
        # Mark selected variables
        for var_name in selected_vars:
            if var_name in self.variables and self.variables[var_name]['current_status'] == 'active':
                self.variables[var_name]['final_selected'] = True
                self.variables[var_name]['steps_history'].append({
                    'step': self.current_step,
                    'step_name': method,
                    'action': 'selected',
                    'reason': f'Selected by {method}'
                })
        
        # Mark non-selected variables as dropped
        active_vars = [name for name, info in self.variables.items() 
                      if info['current_status'] == 'active' and name != self.target_column]
        
        not_selected = [var for var in active_vars if var not in selected_vars]
        self.drop_variables_batch(
            not_selected, 
            method, 
            f"Not selected by {method} (ranked lower than top {len(selected_vars)})"
        )
        
        self.add_step(method, f"Selected {len(selected_vars)} out of {total_available} available features")
    
    def get_active_variables(self) -> List[str]:
        """Get list of currently active variables (excluding target)."""
        return [name for name, info in self.variables.items() 
                if info['current_status'] == 'active' and name != self.target_column]
    
    def get_dropped_variables(self) -> List[str]:
        """Get list of dropped variables."""
        return [name for name, info in self.variables.items() 
                if info['current_status'] == 'dropped']
    
    def get_selected_variables(self) -> List[str]:
        """Get list of finally selected variables."""
        return [name for name, info in self.variables.items() 
                if info['final_selected']]
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the variable tracking."""
        total_vars = len(self.variables) - 1  # Exclude target
        active_vars = len(self.get_active_variables())
        dropped_vars = len(self.get_dropped_variables())
        selected_vars = len(self.get_selected_variables())
        
        # Count by drop reasons
        drop_reasons = {}
        for var_info in self.variables.values():
            if var_info['current_status'] == 'dropped' and var_info['drop_reason']:
                reason = var_info['drop_reason']
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
        
        return {
            'total_variables': total_vars,
            'active_variables': active_vars,
            'dropped_variables': dropped_vars,
            'selected_variables': selected_vars,
            'drop_reasons': drop_reasons,
            'steps_completed': len(self.steps)
        }
    
    def generate_excel_report(self, output_path: str = "variable_tracking_report.xlsx"):
        """
        Generate a detailed Excel report of variable tracking.
        
        Args:
            output_path: Path where to save the Excel report
        """
        try:
            # Prepare data for Excel
            report_data = []
            
            for var_name, var_info in self.variables.items():
                if var_name == self.target_column:
                    continue  # Skip target column
                
                report_data.append({
                    'Variable_Name': var_name,
                    'Data_Type': var_info['data_type'],
                    'Variable_Type': var_info['variable_type'],
                    'Final_Status': 'Selected' if var_info['final_selected'] else 'Dropped',
                    'Current_Status': var_info['current_status'],
                    'Drop_Step': var_info['drop_step'] if var_info['drop_step'] else 'N/A',
                    'Drop_Reason': var_info['drop_reason'] if var_info['drop_reason'] else 'N/A',
                    'Steps_History': '; '.join([f"Step {h['step']}: {h['action']} - {h['reason']}" 
                                              for h in var_info['steps_history']])
                })
            
            # Create DataFrame
            df_variables = pd.DataFrame(report_data)
            
            # Create steps summary
            steps_data = []
            for step in self.steps:
                steps_data.append({
                    'Step_Number': step['step_number'],
                    'Step_Name': step['step_name'],
                    'Description': step['description'],
                    'Timestamp': step['timestamp']
                })
            
            df_steps = pd.DataFrame(steps_data)
            
            # Create summary
            summary = self.generate_summary()
            summary_data = [
                {'Metric': 'Total Variables (excluding target)', 'Value': summary['total_variables']},
                {'Metric': 'Finally Selected Variables', 'Value': summary['selected_variables']},
                {'Metric': 'Dropped Variables', 'Value': summary['dropped_variables']},
                {'Metric': 'Selection Rate', 'Value': f"{(summary['selected_variables']/summary['total_variables']*100):.1f}%" if summary['total_variables'] > 0 else "0%"},
                {'Metric': 'Processing Steps Completed', 'Value': summary['steps_completed']},
                {'Metric': 'Task Type', 'Value': self.task_type.title()},
                {'Metric': 'Target Column', 'Value': self.target_column}
            ]
            
            # Add drop reasons to summary
            for reason, count in summary['drop_reasons'].items():
                summary_data.append({
                    'Metric': f'Dropped - {reason}',
                    'Value': count
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Variables detail sheet
                df_variables.to_excel(writer, sheet_name='Variables_Detail', index=False)
                
                # Steps sheet
                df_steps.to_excel(writer, sheet_name='Processing_Steps', index=False)
                
                # Selected variables sheet
                selected_vars = df_variables[df_variables['Final_Status'] == 'Selected']
                selected_vars.to_excel(writer, sheet_name='Selected_Variables', index=False)
                
                # Dropped variables sheet
                dropped_vars = df_variables[df_variables['Final_Status'] == 'Dropped']
                dropped_vars.to_excel(writer, sheet_name='Dropped_Variables', index=False)
            
            print(f"✅ Variable tracking report saved to: {output_path}")
            print(f"📊 Report includes {len(report_data)} variables across {len(self.steps)} processing steps")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to generate Excel report: {e}")
            return None
    
    def print_summary(self):
        """Print a summary of the variable tracking to console."""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("📊 VARIABLE TRACKING SUMMARY")
        print("="*60)
        print(f"Task Type: {self.task_type.title()}")
        print(f"Target Column: {self.target_column}")
        print(f"Total Variables: {summary['total_variables']} (excluding target)")
        print(f"Finally Selected: {summary['selected_variables']}")
        print(f"Dropped: {summary['dropped_variables']}")
        print(f"Selection Rate: {(summary['selected_variables']/summary['total_variables']*100):.1f}%" if summary['total_variables'] > 0 else "0%")
        print(f"Processing Steps: {summary['steps_completed']}")
        
        if summary['drop_reasons']:
            print(f"\n📋 Drop Reasons:")
            for reason, count in summary['drop_reasons'].items():
                print(f"   • {reason}: {count} variables")
        
        print("="*60)
