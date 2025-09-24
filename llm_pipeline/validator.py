"""
Validator for Quality Checks and Validation
Ensures extracted statistics are valid and compares with regex results
"""

import re
import json
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Validator:
    """Validate extracted statistics and ensure quality"""
    
    def __init__(self):
        """Initialize the validator"""
        self.validation_rules = self._define_validation_rules()
        self.validation_stats = defaultdict(int)
        self.error_log = []
        
    def _define_validation_rules(self) -> Dict[str, Any]:
        """Define validation rules for different statistic types"""
        return {
            'confidence_interval': {
                'lower_bound_check': lambda lower, upper: lower < upper,
                'reasonable_range': lambda lower, upper: -1000 < lower < 1000 and -1000 < upper < 1000,
                'confidence_level': lambda level: level in [90, 95, 99] if level else True,
                'width_check': lambda lower, upper: 0 < (upper - lower) < 100
            },
            'p_value': {
                'range_check': lambda p: 0 <= p <= 1,
                'precision_check': lambda p: len(str(p).split('.')[-1]) <= 10 if '.' in str(p) else True,
                'common_values': lambda p: p not in [0.0000000000, 1.0000000000]  # Avoid extreme precision
            },
            'sample_size': {
                'positive_check': lambda n: n > 0,
                'reasonable_range': lambda n: n < 10000000,  # Less than 10 million
                'integer_check': lambda n: float(n).is_integer() if isinstance(n, (int, float)) else True
            },
            'effect_size': {
                'cohens_d_range': lambda d: -5 <= d <= 5,
                'correlation_range': lambda r: -1 <= r <= 1,
                'odds_ratio_positive': lambda odds: odds > 0,
                'r_squared_range': lambda r2: 0 <= r2 <= 1
            }
        }
        
    def validate_extraction(self, extraction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single extraction result
        
        Args:
            extraction: Extracted statistics for an article
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if extraction has required fields
        if 'pmc_id' not in extraction:
            issues.append("Missing PMC ID")
            
        # Validate confidence intervals
        if 'confidence_intervals' in extraction:
            ci_issues = self._validate_confidence_intervals(extraction['confidence_intervals'])
            issues.extend(ci_issues)
            
        # Validate p-values
        if 'p_values' in extraction:
            p_issues = self._validate_p_values(extraction['p_values'])
            issues.extend(p_issues)
            
        # Validate sample sizes
        if 'sample_sizes' in extraction:
            n_issues = self._validate_sample_sizes(extraction['sample_sizes'])
            issues.extend(n_issues)
            
        # Validate effect sizes
        if 'effect_sizes' in extraction:
            effect_issues = self._validate_effect_sizes(extraction['effect_sizes'])
            issues.extend(effect_issues)
            
        # Check for impossible combinations
        combo_issues = self._validate_combinations(extraction)
        issues.extend(combo_issues)
        
        # Update stats
        self.validation_stats['total_validated'] += 1
        if not issues:
            self.validation_stats['valid'] += 1
        else:
            self.validation_stats['invalid'] += 1
            self.error_log.append({
                'pmc_id': extraction.get('pmc_id', 'unknown'),
                'issues': issues,
                'timestamp': datetime.now().isoformat()
            })
            
        return len(issues) == 0, issues
        
    def _validate_confidence_intervals(self, cis: List[Any]) -> List[str]:
        """Validate confidence intervals"""
        issues = []
        rules = self.validation_rules['confidence_interval']
        
        for i, ci in enumerate(cis):
            try:
                if isinstance(ci, dict):
                    lower = float(ci.get('lower', 0))
                    upper = float(ci.get('upper', 0))
                    level = ci.get('level')
                elif isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    lower = float(ci[0])
                    upper = float(ci[1])
                    level = ci[2] if len(ci) > 2 else None
                else:
                    issues.append(f"CI {i}: Invalid format")
                    continue
                    
                # Apply validation rules
                if not rules['lower_bound_check'](lower, upper):
                    issues.append(f"CI {i}: Lower bound ({lower}) >= upper bound ({upper})")
                    
                if not rules['reasonable_range'](lower, upper):
                    issues.append(f"CI {i}: Unreasonable range [{lower}, {upper}]")
                    
                if level and not rules['confidence_level'](level):
                    issues.append(f"CI {i}: Unusual confidence level {level}%")
                    
                if not rules['width_check'](lower, upper):
                    issues.append(f"CI {i}: Unusual width {upper - lower}")
                    
            except (ValueError, TypeError) as e:
                issues.append(f"CI {i}: Parsing error - {str(e)}")
                
        return issues
        
    def _validate_p_values(self, p_values: List[Any]) -> List[str]:
        """Validate p-values"""
        issues = []
        rules = self.validation_rules['p_value']
        
        for i, p_val in enumerate(p_values):
            try:
                if isinstance(p_val, dict):
                    p = float(p_val.get('value', 0))
                else:
                    p = float(p_val)
                    
                # Apply validation rules
                if not rules['range_check'](p):
                    issues.append(f"P-value {i}: Out of range ({p})")
                    
                if not rules['precision_check'](p):
                    issues.append(f"P-value {i}: Excessive precision")
                    
                if not rules['common_values'](p):
                    issues.append(f"P-value {i}: Suspicious value ({p})")
                    
            except (ValueError, TypeError) as e:
                issues.append(f"P-value {i}: Parsing error - {str(e)}")
                
        return issues
        
    def _validate_sample_sizes(self, sample_sizes: List[Any]) -> List[str]:
        """Validate sample sizes"""
        issues = []
        rules = self.validation_rules['sample_size']
        
        for i, n in enumerate(sample_sizes):
            try:
                if isinstance(n, dict):
                    size = float(n.get('value', 0))
                else:
                    size = float(n)
                    
                # Apply validation rules
                if not rules['positive_check'](size):
                    issues.append(f"Sample size {i}: Non-positive value ({size})")
                    
                if not rules['reasonable_range'](size):
                    issues.append(f"Sample size {i}: Unreasonable value ({size})")
                    
                if not rules['integer_check'](size):
                    issues.append(f"Sample size {i}: Non-integer value ({size})")
                    
            except (ValueError, TypeError) as e:
                issues.append(f"Sample size {i}: Parsing error - {str(e)}")
                
        return issues
        
    def _validate_effect_sizes(self, effect_sizes: List[Any]) -> List[str]:
        """Validate effect sizes"""
        issues = []
        rules = self.validation_rules['effect_size']
        
        for i, effect in enumerate(effect_sizes):
            try:
                if isinstance(effect, dict):
                    effect_type = effect.get('type', 'unknown')
                    value = float(effect.get('value', 0))
                    
                    if effect_type == 'cohens_d' and not rules['cohens_d_range'](value):
                        issues.append(f"Cohen's d {i}: Out of typical range ({value})")
                    elif effect_type == 'correlation' and not rules['correlation_range'](value):
                        issues.append(f"Correlation {i}: Out of range ({value})")
                    elif effect_type == 'odds_ratio' and not rules['odds_ratio_positive'](value):
                        issues.append(f"Odds ratio {i}: Non-positive value ({value})")
                    elif effect_type == 'r_squared' and not rules['r_squared_range'](value):
                        issues.append(f"R-squared {i}: Out of range ({value})")
                        
            except (ValueError, TypeError) as e:
                issues.append(f"Effect size {i}: Parsing error - {str(e)}")
                
        return issues
        
    def _validate_combinations(self, extraction: Dict[str, Any]) -> List[str]:
        """Check for impossible statistical combinations"""
        issues = []
        
        # Check if significant p-value with CI including null
        if 'p_values' in extraction and 'confidence_intervals' in extraction:
            for p_val in extraction['p_values']:
                try:
                    p = float(p_val) if not isinstance(p_val, dict) else float(p_val.get('value', 1))
                    if p < 0.05:  # Significant
                        for ci in extraction['confidence_intervals']:
                            if isinstance(ci, dict):
                                lower = float(ci.get('lower', 0))
                                upper = float(ci.get('upper', 0))
                                # Check if CI includes null (0 for differences, 1 for ratios)
                                if lower < 0 < upper:
                                    issues.append("Significant p-value with CI including null")
                                    break
                except:
                    pass
                    
        # Check sample size consistency
        if 'sample_sizes' in extraction:
            sizes = []
            for n in extraction['sample_sizes']:
                try:
                    size = float(n) if not isinstance(n, dict) else float(n.get('value', 0))
                    sizes.append(size)
                except:
                    pass
                    
            if len(sizes) > 1:
                # Check for unrealistic variation
                if max(sizes) / min(sizes) > 1000:
                    issues.append("Extreme variation in sample sizes within article")
                    
        return issues
        
    def compare_with_regex(self, 
                          llm_extraction: Dict[str, Any],
                          regex_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare LLM extraction with regex baseline
        
        Args:
            llm_extraction: Results from LLM extraction
            regex_extraction: Results from regex extraction
            
        Returns:
            Comparison metrics
        """
        comparison = {
            'pmc_id': llm_extraction.get('pmc_id'),
            'llm_only': {},
            'regex_only': {},
            'both': {},
            'metrics': {}
        }
        
        # Compare confidence intervals
        llm_cis = set(self._normalize_cis(llm_extraction.get('confidence_intervals', [])))
        regex_cis = set(self._normalize_cis(regex_extraction.get('confidence_intervals', [])))
        
        comparison['both']['confidence_intervals'] = len(llm_cis & regex_cis)
        comparison['llm_only']['confidence_intervals'] = len(llm_cis - regex_cis)
        comparison['regex_only']['confidence_intervals'] = len(regex_cis - llm_cis)
        
        # Compare p-values
        llm_ps = set(self._normalize_p_values(llm_extraction.get('p_values', [])))
        regex_ps = set(self._normalize_p_values(regex_extraction.get('p_values', [])))
        
        comparison['both']['p_values'] = len(llm_ps & regex_ps)
        comparison['llm_only']['p_values'] = len(llm_ps - regex_ps)
        comparison['regex_only']['p_values'] = len(regex_ps - llm_ps)
        
        # Calculate metrics
        comparison['metrics']['llm_total'] = sum(
            len(llm_extraction.get(key, [])) 
            for key in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']
        )
        comparison['metrics']['regex_total'] = sum(
            len(regex_extraction.get(key, [])) 
            for key in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']
        )
        comparison['metrics']['improvement_rate'] = (
            (comparison['metrics']['llm_total'] - comparison['metrics']['regex_total']) / 
            max(comparison['metrics']['regex_total'], 1) * 100
        )
        
        return comparison
        
    def _normalize_cis(self, cis: List[Any]) -> List[Tuple[float, float]]:
        """Normalize confidence intervals for comparison"""
        normalized = []
        for ci in cis:
            try:
                if isinstance(ci, dict):
                    lower = round(float(ci.get('lower', 0)), 3)
                    upper = round(float(ci.get('upper', 0)), 3)
                elif isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    lower = round(float(ci[0]), 3)
                    upper = round(float(ci[1]), 3)
                else:
                    continue
                normalized.append((lower, upper))
            except:
                continue
        return normalized
        
    def _normalize_p_values(self, p_values: List[Any]) -> List[float]:
        """Normalize p-values for comparison"""
        normalized = []
        for p in p_values:
            try:
                if isinstance(p, dict):
                    value = float(p.get('value', 0))
                else:
                    value = float(p)
                # Round to avoid floating point comparison issues
                normalized.append(round(value, 6))
            except:
                continue
        return normalized
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validated': self.validation_stats['total_validated'],
                'valid': self.validation_stats['valid'],
                'invalid': self.validation_stats['invalid'],
                'validity_rate': (
                    self.validation_stats['valid'] / 
                    max(self.validation_stats['total_validated'], 1) * 100
                )
            },
            'common_issues': self._analyze_common_issues(),
            'recent_errors': self.error_log[-10:],  # Last 10 errors
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _analyze_common_issues(self) -> List[Dict[str, Any]]:
        """Analyze most common validation issues"""
        issue_counter = Counter()
        
        for error in self.error_log:
            for issue in error['issues']:
                # Extract issue type
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counter[issue_type] += 1
                
        common_issues = [
            {'issue': issue, 'count': count}
            for issue, count in issue_counter.most_common(10)
        ]
        
        return common_issues
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        validity_rate = (
            self.validation_stats['valid'] / 
            max(self.validation_stats['total_validated'], 1) * 100
        )
        
        if validity_rate < 80:
            recommendations.append("High error rate detected. Consider adjusting LLM prompts.")
            
        # Check for specific issues
        common_issues = self._analyze_common_issues()
        if common_issues:
            top_issue = common_issues[0]['issue']
            if 'CI' in top_issue:
                recommendations.append("Many CI validation errors. Review CI extraction logic.")
            if 'P-value' in top_issue:
                recommendations.append("P-value extraction issues. Ensure proper decimal handling.")
            if 'Sample size' in top_issue:
                recommendations.append("Sample size errors. Check for non-integer values.")
                
        if len(self.error_log) > 100:
            recommendations.append("Large number of errors. Consider manual review of error patterns.")
            
        return recommendations
        
    def validate_batch(self, batch_results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
        """
        Validate a batch of extraction results
        
        Args:
            batch_results: List of extraction results
            
        Returns:
            Tuple of (validated_results, batch_issues)
        """
        validated_results = []
        batch_issues = []
        
        for result in batch_results:
            is_valid, issues = self.validate_extraction(result)
            if is_valid:
                validated_results.append(result)
            else:
                batch_issues.append(f"{result.get('pmc_id', 'unknown')}: {'; '.join(issues[:3])}")
                # Still include with flag
                result['validation_issues'] = issues
                validated_results.append(result)
                
        return validated_results, batch_issues
        
    def calculate_confidence_score(self, extraction: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for an extraction
        
        Args:
            extraction: Extraction results
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = []
        
        # Base score on validation
        is_valid, issues = self.validate_extraction(extraction)
        base_score = 1.0 if is_valid else max(0.3, 1.0 - len(issues) * 0.1)
        scores.append(base_score)
        
        # Check completeness
        expected_fields = ['confidence_intervals', 'p_values', 'sample_sizes']
        present_fields = sum(1 for field in expected_fields if field in extraction and extraction[field])
        completeness_score = present_fields / len(expected_fields)
        scores.append(completeness_score)
        
        # Check for reasonable counts
        total_findings = sum(len(extraction.get(key, [])) for key in expected_fields)
        if total_findings == 0:
            count_score = 0.2
        elif total_findings > 100:
            count_score = 0.7  # Might be over-extraction
        else:
            count_score = min(1.0, total_findings / 20)  # Expect ~20 findings
        scores.append(count_score)
        
        return statistics.mean(scores)


def test_validator():
    """Test the validator"""
    validator = Validator()
    
    # Test valid extraction
    valid_extraction = {
        'pmc_id': 'PMC123456',
        'confidence_intervals': [
            {'lower': 1.2, 'upper': 3.4, 'level': 95},
            [0.5, 0.8]
        ],
        'p_values': [0.023, 0.001, 0.456],
        'sample_sizes': [100, 250, 175],
        'effect_sizes': [
            {'type': 'cohens_d', 'value': 0.85},
            {'type': 'correlation', 'value': 0.65}
        ]
    }
    
    is_valid, issues = validator.validate_extraction(valid_extraction)
    print(f"Valid extraction: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
        
    # Test invalid extraction
    invalid_extraction = {
        'pmc_id': 'PMC789',
        'confidence_intervals': [
            {'lower': 5.0, 'upper': 2.0},  # Invalid: lower > upper
        ],
        'p_values': [1.5, -0.1],  # Invalid: out of range
        'sample_sizes': [-100, 0],  # Invalid: non-positive
    }
    
    is_valid, issues = validator.validate_extraction(invalid_extraction)
    print(f"\nInvalid extraction: {is_valid}")
    print(f"Issues: {issues}")
    
    # Test comparison
    llm_result = valid_extraction.copy()
    llm_result['confidence_intervals'].append({'lower': 2.1, 'upper': 4.5})
    
    regex_result = {
        'pmc_id': 'PMC123456',
        'confidence_intervals': [
            {'lower': 1.2, 'upper': 3.4, 'level': 95}
        ],
        'p_values': [0.023, 0.001]
    }
    
    comparison = validator.compare_with_regex(llm_result, regex_result)
    print(f"\nComparison:")
    print(f"  Both found: {comparison['both']}")
    print(f"  LLM only: {comparison['llm_only']}")
    print(f"  Regex only: {comparison['regex_only']}")
    print(f"  Improvement rate: {comparison['metrics']['improvement_rate']:.1f}%")
    
    # Generate report
    report = validator.generate_validation_report()
    print(f"\nValidation Report:")
    print(f"  Summary: {report['summary']}")
    print(f"  Recommendations: {report['recommendations']}")


if __name__ == "__main__":
    test_validator()
