"""
Core Extraction Engine for Statistical Information
Handles the actual extraction logic and prompting
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalFinding:
    """Represents a single statistical finding"""
    type: str  # 'ci', 'p_value', 'sample_size', 'effect_size', 'other'
    value: Any
    context: str  # Surrounding text
    location: str  # 'table', 'figure', 'text', 'abstract'
    confidence: float  # 0-1 confidence score
    raw_text: str  # Original text
    
    def to_dict(self):
        return asdict(self)


class ExtractionEngine:
    """Core engine for extracting statistical information from articles"""
    
    # Enhanced extraction prompt focusing on tables and non-standard formats
    EXTRACTION_PROMPT_TEMPLATE = """
You are a statistical extraction expert. Your task is to find ALL statistical information in the provided articles, with special attention to tables, figures, and non-standard formats that regex patterns typically miss.

CRITICAL FOCUS AREAS:
1. **Tables**: Extract ALL values from statistical tables, including:
   - Column headers and row labels
   - All numeric values with their units
   - Footnotes and table captions
   - Values in parentheses or brackets

2. **Confidence Intervals** in ANY format:
   - Standard: (X to Y), (X-Y), [X, Y], X (95% CI: Y-Z)
   - In tables as separate columns (Lower CI, Upper CI)
   - Parenthetical: value (lower, upper)
   - Narrative: "between X and Y"
   - With percentage: X% (Y%-Z%)

3. **P-values** in non-standard formats:
   - Text descriptions: "significant at 0.05 level", "p less than 0.05"
   - Symbols: * = p<0.05, ** = p<0.01, *** = p<0.001
   - Table footnotes with significance indicators
   - Exact values: p=0.023, P < 0.001

4. **Sample Sizes**:
   - n=X format anywhere in text
   - "X participants", "X patients", "X subjects"
   - In table headers: (n=X)
   - Total and subgroup sizes

5. **Effect Sizes**:
   - Cohen's d, eta squared (η²), R squared (R²)
   - Odds ratios (OR), hazard ratios (HR), risk ratios (RR)
   - Beta coefficients (β), correlation coefficients (r)
   - Standardized mean differences (SMD)

6. **Statistical Tests**:
   - Test names and their statistics (t, F, χ², z values)
   - Degrees of freedom (df)
   - Test assumptions and corrections applied

For each statistical value found, provide:
- The exact value as it appears
- The type of statistic
- The surrounding context (20-30 words)
- Where it was found (table/figure/text/abstract)
- Your confidence level (0-1)

IMPORTANT: 
- Extract values even if you're uncertain - include them with lower confidence
- Preserve the exact format and precision of all numbers
- Include units when present
- Note if values are from supplementary materials references

Articles to process:
{articles_text}

Return your findings in this JSON structure:
{{
    "articles": [
        {{
            "pmc_id": "PMC_ID_HERE",
            "findings": [
                {{
                    "type": "confidence_interval",
                    "value": {{"lower": X, "upper": Y, "level": 95}},
                    "context": "surrounding text here",
                    "location": "table",
                    "confidence": 0.95,
                    "raw_text": "exact text from article"
                }},
                {{
                    "type": "p_value",
                    "value": 0.023,
                    "context": "surrounding text",
                    "location": "text",
                    "confidence": 0.99,
                    "raw_text": "p=0.023"
                }}
            ],
            "table_count": X,
            "figure_count": Y,
            "has_supplementary": true/false
        }}
    ],
    "extraction_metadata": {{
        "total_findings": X,
        "confidence_intervals": Y,
        "p_values": Z,
        "sample_sizes": A,
        "effect_sizes": B
    }}
}}
"""

    TABLE_FOCUSED_PROMPT = """
Focus specifically on extracting data from TABLES in these articles. Tables often contain the majority of statistical information that regex patterns miss.

For each table found:
1. Identify all column headers
2. Extract every numeric value with its row/column context
3. Look for CI columns (often labeled as "95% CI", "CI", "Lower-Upper", etc.)
4. Find p-value columns or significance indicators
5. Extract sample sizes from table headers or first columns
6. Capture footnotes with statistical information

Pay special attention to:
- Values in parentheses: often contain CIs or ranges
- Asterisks and symbols: usually indicate p-values
- Formatted ranges: X-Y, X to Y, X (Y, Z)
- Percentages with CIs: X% (Y%-Z%)

Articles:
{articles_text}

Extract and structure all table data as JSON.
"""

    def __init__(self):
        """Initialize the extraction engine"""
        self.extraction_patterns = self._compile_patterns()
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for validation and initial extraction"""
        patterns = {
            # Confidence intervals - comprehensive patterns
            'ci_parentheses': re.compile(
                r'(\d+\.?\d*)\s*\((\d+\.?\d*)\s*[,-]\s*(\d+\.?\d*)\)',
                re.IGNORECASE
            ),
            'ci_brackets': re.compile(
                r'\[(\d+\.?\d*)\s*[,-]\s*(\d+\.?\d*)\]',
                re.IGNORECASE
            ),
            'ci_to': re.compile(
                r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'ci_between': re.compile(
                r'between\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'ci_explicit': re.compile(
                r'(CI|confidence\s+interval)[:\s]+(\d+\.?\d*)\s*[,-]\s*(\d+\.?\d*)',
                re.IGNORECASE
            ),
            
            # P-values
            'p_value_exact': re.compile(
                r'[Pp]\s*[=<>≤≥]\s*(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'p_value_text': re.compile(
                r'(significant|significance)\s+at\s+(\d+\.?\d*)\s+level',
                re.IGNORECASE
            ),
            
            # Sample sizes
            'sample_size': re.compile(
                r'[nN]\s*=\s*(\d+)',
                re.IGNORECASE
            ),
            'participants': re.compile(
                r'(\d+)\s+(participants?|patients?|subjects?|individuals?|cases?)',
                re.IGNORECASE
            ),
            
            # Effect sizes
            'cohens_d': re.compile(
                r"Cohen's\s+d\s*=\s*(\d+\.?\d*)",
                re.IGNORECASE
            ),
            'odds_ratio': re.compile(
                r'(OR|odds\s+ratio)\s*[=:]\s*(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'correlation': re.compile(
                r'[rR]\s*=\s*[+-]?(\d+\.?\d*)',
                re.IGNORECASE
            ),
            
            # Statistical tests
            't_test': re.compile(
                r't\s*\(\s*(\d+)\s*\)\s*=\s*(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'f_test': re.compile(
                r'F\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'chi_square': re.compile(
                r'χ²\s*\(\s*(\d+)\s*\)\s*=\s*(\d+\.?\d*)',
                re.IGNORECASE
            )
        }
        return patterns
        
    def extract_statistics(self, 
                          article: Dict[str, Any],
                          use_llm: bool = True) -> List[StatisticalFinding]:
        """
        Extract statistics from a single article
        
        Args:
            article: Article dictionary with text content
            use_llm: Whether to use LLM extraction (vs regex only)
            
        Returns:
            List of StatisticalFinding objects
        """
        findings = []
        
        # First pass: Quick regex extraction for validation
        regex_findings = self._regex_extraction(article)
        findings.extend(regex_findings)
        
        # Identify potential table/figure sections
        table_sections = self._identify_table_sections(article)
        figure_sections = self._identify_figure_sections(article)
        
        if use_llm:
            # Prepare article for LLM extraction
            llm_input = self._prepare_article_for_llm(
                article, 
                table_sections, 
                figure_sections
            )
            return llm_input  # This will be processed by the LLM
        
        return findings
        
    def _regex_extraction(self, article: Dict[str, Any]) -> List[StatisticalFinding]:
        """Perform regex-based extraction as baseline"""
        findings = []
        
        # Combine all text content
        full_text = f"{article.get('title', '')} {article.get('abstract', '')} {article.get('body', '')}"
        
        # Extract using patterns
        for pattern_name, pattern in self.extraction_patterns.items():
            matches = pattern.finditer(full_text)
            for match in matches:
                # Determine statistic type from pattern name
                stat_type = self._get_stat_type_from_pattern(pattern_name)
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(full_text), match.end() + 50)
                context = full_text[start:end]
                
                finding = StatisticalFinding(
                    type=stat_type,
                    value=match.groups(),
                    context=context,
                    location='text',
                    confidence=0.9,  # High confidence for regex matches
                    raw_text=match.group(0)
                )
                findings.append(finding)
                
        return findings
        
    def _get_stat_type_from_pattern(self, pattern_name: str) -> str:
        """Map pattern name to statistic type"""
        if 'ci' in pattern_name.lower() or 'confidence' in pattern_name.lower():
            return 'confidence_interval'
        elif 'p_value' in pattern_name:
            return 'p_value'
        elif 'sample' in pattern_name or 'participants' in pattern_name:
            return 'sample_size'
        elif any(x in pattern_name for x in ['cohens', 'odds', 'correlation']):
            return 'effect_size'
        elif any(x in pattern_name for x in ['t_test', 'f_test', 'chi']):
            return 'statistical_test'
        else:
            return 'other'
            
    def _identify_table_sections(self, article: Dict[str, Any]) -> List[Tuple[int, int, str]]:
        """Identify sections of text that likely contain tables"""
        body = article.get('body', '')
        if not body:
            return []
            
        table_sections = []
        
        # Look for table markers
        table_patterns = [
            r'Table\s+\d+[:\.]',
            r'TABLE\s+\d+',
            r'\|.*\|.*\|',  # Pipe-delimited tables
            r'┌─.*─┐',  # Box-drawing tables
            r'<table.*?>.*?</table>',  # HTML tables
        ]
        
        for pattern_str in table_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            for match in pattern.finditer(body):
                # Extract table and surrounding context
                start = max(0, match.start() - 100)
                end = min(len(body), match.end() + 500)
                table_text = body[start:end]
                table_sections.append((start, end, table_text))
                
        return table_sections
        
    def _identify_figure_sections(self, article: Dict[str, Any]) -> List[Tuple[int, int, str]]:
        """Identify sections of text that likely contain figure captions"""
        body = article.get('body', '')
        if not body:
            return []
            
        figure_sections = []
        
        # Look for figure markers
        figure_patterns = [
            r'Figure\s+\d+[:\.].*?(?=Figure\s+\d+|Table\s+\d+|$)',
            r'FIGURE\s+\d+.*?(?=FIGURE\s+\d+|TABLE\s+\d+|$)',
            r'Fig\.\s+\d+.*?(?=Fig\.\s+\d+|Table\s+\d+|$)',
        ]
        
        for pattern_str in figure_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            for match in pattern.finditer(body):
                figure_sections.append((match.start(), match.end(), match.group(0)))
                
        return figure_sections
        
    def _prepare_article_for_llm(self, 
                                article: Dict[str, Any],
                                table_sections: List[Tuple[int, int, str]],
                                figure_sections: List[Tuple[int, int, str]]) -> Dict[str, Any]:
        """Prepare article content for LLM processing"""
        
        # Truncate body if too long, but keep tables and figures
        body = article.get('body', '')
        if len(body) > 50000:  # Truncate very long articles
            # Keep abstract and beginning
            truncated_body = body[:10000]
            
            # Add all table sections
            for start, end, text in table_sections:
                truncated_body += f"\n\n[TABLE SECTION]\n{text}\n"
                
            # Add figure captions
            for start, end, text in figure_sections[:10]:  # Limit figures
                truncated_body += f"\n\n[FIGURE CAPTION]\n{text[:500]}\n"
                
            body = truncated_body
            
        return {
            'pmc_id': article.get('pmc_id'),
            'title': article.get('title', ''),
            'abstract': article.get('abstract', ''),
            'body': body,
            'has_tables': len(table_sections) > 0,
            'has_figures': len(figure_sections) > 0,
            'table_count': len(table_sections),
            'figure_count': len(figure_sections)
        }
        
    def parse_llm_response(self, 
                          response: Dict[str, Any],
                          article: Dict[str, Any]) -> List[StatisticalFinding]:
        """Parse LLM response into StatisticalFinding objects"""
        findings = []
        
        try:
            # Handle different response formats
            if 'findings' in response:
                raw_findings = response['findings']
            elif 'articles' in response and len(response['articles']) > 0:
                raw_findings = response['articles'][0].get('findings', [])
            else:
                raw_findings = []
                
            for finding_dict in raw_findings:
                finding = StatisticalFinding(
                    type=finding_dict.get('type', 'other'),
                    value=finding_dict.get('value'),
                    context=finding_dict.get('context', ''),
                    location=finding_dict.get('location', 'unknown'),
                    confidence=float(finding_dict.get('confidence', 0.5)),
                    raw_text=finding_dict.get('raw_text', '')
                )
                findings.append(finding)
                
        except Exception as e:
            logger.error(f"Error parsing LLM response for {article.get('pmc_id')}: {e}")
            
        return findings
        
    def validate_findings(self, findings: List[StatisticalFinding]) -> List[StatisticalFinding]:
        """Validate and filter findings for quality"""
        validated = []
        
        for finding in findings:
            # Skip findings with very low confidence
            if finding.confidence < self.confidence_thresholds['low']:
                continue
                
            # Validate based on type
            if finding.type == 'confidence_interval':
                if self._validate_ci(finding):
                    validated.append(finding)
            elif finding.type == 'p_value':
                if self._validate_p_value(finding):
                    validated.append(finding)
            elif finding.type == 'sample_size':
                if self._validate_sample_size(finding):
                    validated.append(finding)
            else:
                validated.append(finding)
                
        return validated
        
    def _validate_ci(self, finding: StatisticalFinding) -> bool:
        """Validate confidence interval"""
        try:
            if isinstance(finding.value, dict):
                lower = float(finding.value.get('lower', 0))
                upper = float(finding.value.get('upper', 0))
                return lower < upper
            elif isinstance(finding.value, (list, tuple)) and len(finding.value) >= 2:
                lower = float(finding.value[0])
                upper = float(finding.value[1])
                return lower < upper
        except:
            return False
        return True
        
    def _validate_p_value(self, finding: StatisticalFinding) -> bool:
        """Validate p-value"""
        try:
            if isinstance(finding.value, (int, float)):
                p = float(finding.value)
                return 0 <= p <= 1
            elif isinstance(finding.value, str):
                # Try to extract numeric value
                match = re.search(r'(\d+\.?\d*)', finding.value)
                if match:
                    p = float(match.group(1))
                    return 0 <= p <= 1
        except:
            return False
        return True
        
    def _validate_sample_size(self, finding: StatisticalFinding) -> bool:
        """Validate sample size"""
        try:
            if isinstance(finding.value, (int, float)):
                n = int(finding.value)
                return n > 0
            elif isinstance(finding.value, str):
                match = re.search(r'(\d+)', finding.value)
                if match:
                    n = int(match.group(1))
                    return n > 0
        except:
            return False
        return True
        
    def merge_findings(self, 
                      regex_findings: List[StatisticalFinding],
                      llm_findings: List[StatisticalFinding]) -> List[StatisticalFinding]:
        """Merge and deduplicate findings from regex and LLM"""
        all_findings = []
        seen = set()
        
        # Process LLM findings first (usually more complete)
        for finding in llm_findings:
            finding_key = f"{finding.type}_{finding.raw_text[:50]}"
            if finding_key not in seen:
                all_findings.append(finding)
                seen.add(finding_key)
                
        # Add regex findings not found by LLM
        for finding in regex_findings:
            finding_key = f"{finding.type}_{finding.raw_text[:50]}"
            if finding_key not in seen:
                # Lower confidence for regex-only findings
                finding.confidence *= 0.8
                all_findings.append(finding)
                seen.add(finding_key)
                
        return all_findings


def test_extraction_engine():
    """Test the extraction engine"""
    engine = ExtractionEngine()
    
    # Test article
    test_article = {
        'pmc_id': 'PMC123456',
        'title': 'Test Article with Statistics',
        'abstract': 'This study found significant results (p<0.05) with a sample of n=100 participants.',
        'body': '''
        Table 1. Results
        | Variable | Mean (95% CI) | P-value |
        |----------|---------------|---------|
        | Age      | 45.2 (42.1-48.3) | 0.023 |
        | Weight   | 72.5 (68.9-76.1) | <0.001 |
        
        The effect size was large (Cohen's d = 0.85) with an odds ratio of 2.3 (1.5-3.1).
        '''
    }
    
    # Test regex extraction
    regex_findings = engine._regex_extraction(test_article)
    print(f"Found {len(regex_findings)} statistics with regex")
    for finding in regex_findings[:3]:
        print(f"  - {finding.type}: {finding.raw_text}")
        
    # Test table identification
    table_sections = engine._identify_table_sections(test_article)
    print(f"Found {len(table_sections)} table sections")
    
    # Prepare for LLM
    llm_input = engine._prepare_article_for_llm(test_article, table_sections, [])
    print(f"Prepared article for LLM: {llm_input['pmc_id']}, has_tables: {llm_input['has_tables']}")


if __name__ == "__main__":
    test_extraction_engine()
