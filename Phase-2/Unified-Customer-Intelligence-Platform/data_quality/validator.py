import pandas as pd
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class DataQualityValidator:
    def __init__(self):
        self.rules = {}
        self.results = {}
    
    def add_rule(self, column: str, rule_type: str, **kwargs):
        """Add validation rule"""
        if column not in self.rules:
            self.rules[column] = []
        self.rules[column].append({"type": rule_type, "params": kwargs})
    
    def validate_completeness(self, df: pd.DataFrame, column: str, threshold: float = 0.95):
        """Check data completeness"""
        completeness = 1 - (df[column].isnull().sum() / len(df))
        return {
            "passed": completeness >= threshold,
            "score": completeness,
            "threshold": threshold,
            "message": f"Completeness: {completeness:.2%}"
        }
    
    def validate_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float):
        """Check value range"""
        valid_count = df[(df[column] >= min_val) & (df[column] <= max_val)].shape[0]
        total_count = df[column].dropna().shape[0]
        score = valid_count / total_count if total_count > 0 else 0
        
        return {
            "passed": score >= 0.95,
            "score": score,
            "message": f"Range validation: {score:.2%} within [{min_val}, {max_val}]"
        }
    
    def validate_uniqueness(self, df: pd.DataFrame, column: str):
        """Check uniqueness"""
        unique_count = df[column].nunique()
        total_count = df[column].dropna().shape[0]
        score = unique_count / total_count if total_count > 0 else 0
        
        return {
            "passed": score >= 0.95,
            "score": score,
            "message": f"Uniqueness: {score:.2%}"
        }
    
    def run_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation rules"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "validations": {}
        }
        
        for column, rules in self.rules.items():
            if column not in df.columns:
                results["validations"][column] = {
                    "passed": False,
                    "message": f"Column {column} not found"
                }
                continue
            
            column_results = []
            for rule in rules:
                rule_type = rule["type"]
                params = rule["params"]
                
                if rule_type == "completeness":
                    result = self.validate_completeness(df, column, **params)
                elif rule_type == "range":
                    result = self.validate_range(df, column, **params)
                elif rule_type == "uniqueness":
                    result = self.validate_uniqueness(df, column)
                
                column_results.append(result)
            
            results["validations"][column] = column_results
        
        return results

validator = DataQualityValidator()
validator.add_rule("customer_id", "uniqueness")
validator.add_rule("tenure_months", "range", min_val=0, max_val=100)
validator.add_rule("monthly_charges", "completeness", threshold=0.95)