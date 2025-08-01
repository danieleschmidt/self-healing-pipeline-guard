#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Repository: self-healing-pipeline-guard
Maturity Level: Advanced (90%)

This module implements continuous value discovery with intelligent scoring
for autonomous SDLC enhancement.
"""

import json
import yaml
import subprocess
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort: float  # hours
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    risk_level: float
    discovered_at: str
    metadata: Dict


class ValueDiscoveryEngine:
    """Autonomous value discovery and scoring engine."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config = self._load_config()
        self.metrics_file = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_file = self.repo_path / ".terragon" / "backlog.json"
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def discover_opportunities(self) -> List[ValueItem]:
        """Execute comprehensive value discovery across all sources."""
        logger.info("Starting autonomous value discovery...")
        
        opportunities = []
        
        # 1. Git history analysis for technical debt
        opportunities.extend(self._analyze_git_history())
        
        # 2. Static analysis for code quality issues
        opportunities.extend(self._analyze_code_quality())
        
        # 3. Security vulnerability scanning
        opportunities.extend(self._analyze_security())
        
        # 4. Dependency update opportunities
        opportunities.extend(self._analyze_dependencies())
        
        # 5. Missing SDLC components
        opportunities.extend(self._analyze_missing_components())
        
        # 6. Performance optimization opportunities
        opportunities.extend(self._analyze_performance())
        
        # Score and prioritize all opportunities
        scored_opportunities = [self._calculate_scores(item) for item in opportunities]
        
        # Sort by composite score descending
        scored_opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"Discovered {len(scored_opportunities)} value opportunities")
        return scored_opportunities
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Analyze git history for technical debt markers."""
        opportunities = []
        
        try:
            # Search for TODO, FIXME, HACK markers in recent commits
            result = subprocess.run([
                "git", "log", "--grep=TODO\\|FIXME\\|HACK\\|TEMP", 
                "--oneline", "-10"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                opportunities.append(ValueItem(
                    id="git-001",
                    title="Address technical debt markers in git history",
                    description="Found TODO/FIXME markers in recent commits",
                    category="technical-debt",
                    source="git-history",
                    files_affected=["multiple"],
                    estimated_effort=2.0,
                    wsjf_score=0, ice_score=0, technical_debt_score=0,
                    composite_score=0, risk_level=0.2,
                    discovered_at=datetime.datetime.now().isoformat(),
                    metadata={"commit_count": len(result.stdout.split('\n'))}
                ))
        except Exception as e:
            logger.warning(f"Git history analysis failed: {e}")
            
        return opportunities
    
    def _analyze_code_quality(self) -> List[ValueItem]:
        """Analyze code quality using static analysis tools."""
        opportunities = []
        
        # Check if ruff is available and run analysis
        try:
            result = subprocess.run([
                "python", "-m", "ruff", "check", "--output-format=json", "."
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                if issues:
                    opportunities.append(ValueItem(
                        id="quality-001",
                        title="Fix code quality issues detected by Ruff",
                        description=f"Found {len(issues)} code quality issues",
                        category="code-quality",
                        source="static-analysis",
                        files_affected=list(set(issue.get('filename', 'unknown') for issue in issues)),
                        estimated_effort=len(issues) * 0.1,  # 6 minutes per issue
                        wsjf_score=0, ice_score=0, technical_debt_score=0,
                        composite_score=0, risk_level=0.1,
                        discovered_at=datetime.datetime.now().isoformat(),
                        metadata={"issue_count": len(issues), "tool": "ruff"}
                    ))
        except Exception as e:
            logger.warning(f"Code quality analysis failed: {e}")
            
        return opportunities
    
    def _analyze_security(self) -> List[ValueItem]:
        """Analyze security vulnerabilities."""
        opportunities = []
        
        # Check for obvious security issues
        try:
            result = subprocess.run([
                "python", "-m", "bandit", "-r", ".", "-f", "json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    issues = bandit_results.get('results', [])
                    if issues:
                        high_severity = [i for i in issues if i.get('issue_severity') == 'HIGH']
                        if high_severity:
                            opportunities.append(ValueItem(
                                id="security-001",
                                title="Fix high-severity security issues",
                                description=f"Found {len(high_severity)} high-severity security issues",
                                category="security",
                                source="security-scan",
                                files_affected=list(set(issue.get('filename', 'unknown') for issue in high_severity)),
                                estimated_effort=len(high_severity) * 0.5,  # 30 min per issue
                                wsjf_score=0, ice_score=0, technical_debt_score=0,
                                composite_score=0, risk_level=0.8,
                                discovered_at=datetime.datetime.now().isoformat(),
                                metadata={"high_severity_count": len(high_severity), "total_issues": len(issues)}
                            ))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Security analysis failed: {e}")
            
        return opportunities
    
    def _analyze_dependencies(self) -> List[ValueItem]:
        """Analyze dependency update opportunities."""
        opportunities = []
        
        # Check if poetry.lock is missing
        if not (self.repo_path / "poetry.lock").exists():
            opportunities.append(ValueItem(
                id="deps-001",
                title="Create poetry.lock for reproducible builds",
                description="Missing poetry.lock file prevents reproducible builds",
                category="dependency-management",
                source="file-analysis",
                files_affected=["poetry.lock"],
                estimated_effort=0.5,
                wsjf_score=0, ice_score=0, technical_debt_score=0,
                composite_score=0, risk_level=0.3,
                discovered_at=datetime.datetime.now().isoformat(),
                metadata={"missing_file": "poetry.lock"}
            ))
            
        return opportunities
    
    def _analyze_missing_components(self) -> List[ValueItem]:
        """Analyze missing SDLC components for Advanced repository."""
        opportunities = []
        
        # Check for missing GitHub Actions workflows
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists() or not list(workflows_dir.glob("*.yml")):
            opportunities.append(ValueItem(
                id="cicd-001",
                title="Implement GitHub Actions workflows",
                description="Missing GitHub Actions workflows despite documentation",
                category="automation",
                source="file-analysis",
                files_affected=[".github/workflows/"],
                estimated_effort=4.0,
                wsjf_score=0, ice_score=0, technical_debt_score=0,
                composite_score=0, risk_level=0.4,
                discovered_at=datetime.datetime.now().isoformat(),
                metadata={"component": "github-actions"}
            ))
            
        # Check for missing pre-commit configuration
        if not (self.repo_path / ".pre-commit-config.yaml").exists():
            opportunities.append(ValueItem(
                id="quality-002",
                title="Setup pre-commit hooks configuration",
                description="Missing pre-commit hooks for automated quality checks",
                category="code-quality",
                source="file-analysis",
                files_affected=[".pre-commit-config.yaml"],
                estimated_effort=1.0,
                wsjf_score=0, ice_score=0, technical_debt_score=0,
                composite_score=0, risk_level=0.2,
                discovered_at=datetime.datetime.now().isoformat(),
                metadata={"component": "pre-commit"}
            ))
            
        return opportunities
    
    def _analyze_performance(self) -> List[ValueItem]:
        """Analyze performance optimization opportunities."""
        opportunities = []
        
        # Check if actual Python implementation exists
        healing_guard_dir = self.repo_path / "healing_guard"
        python_files = list(healing_guard_dir.rglob("*.py")) if healing_guard_dir.exists() else []
        
        # Filter out __pycache__ and get actual implementation files
        impl_files = [f for f in python_files if "__pycache__" not in str(f)]
        
        if len(impl_files) < 5:  # Minimal implementation
            opportunities.append(ValueItem(
                id="impl-001",
                title="Implement core healing_guard Python modules",
                description="Core Python implementation is minimal, only monitoring modules exist",
                category="implementation",
                source="file-analysis",
                files_affected=["healing_guard/"],
                estimated_effort=16.0,  # Substantial implementation effort
                wsjf_score=0, ice_score=0, technical_debt_score=0,
                composite_score=0, risk_level=0.5,
                discovered_at=datetime.datetime.now().isoformat(),
                metadata={"current_files": len(impl_files), "needed": "core-implementation"}
            ))
            
        return opportunities
    
    def _calculate_scores(self, item: ValueItem) -> ValueItem:
        """Calculate WSJF, ICE, and composite scores for value item."""
        # WSJF Calculation (Weighted Shortest Job First)
        user_business_value = self._score_business_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        item.wsjf_score = cost_of_delay / max(item.estimated_effort, 0.1)
        
        # ICE Calculation (Impact, Confidence, Ease)
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = 10 - min(item.estimated_effort, 10)  # Easier = higher score
        item.ice_score = impact * confidence * ease
        
        # Technical Debt Score
        item.technical_debt_score = self._score_technical_debt(item)
        
        # Composite Score with adaptive weighting
        weights = self.config['scoring']['weights']['advanced']
        item.composite_score = (
            weights['wsjf'] * self._normalize_score(item.wsjf_score, 0, 100) +
            weights['ice'] * self._normalize_score(item.ice_score, 0, 1000) +
            weights['technicalDebt'] * self._normalize_score(item.technical_debt_score, 0, 100) +
            weights['security'] * (50 if item.category == 'security' else 0)
        )
        
        # Apply category-specific boosts
        if item.category == 'security':
            item.composite_score *= self.config['scoring']['thresholds']['securityBoost']
        
        return item
    
    def _score_business_value(self, item: ValueItem) -> float:
        """Score user/business value impact (1-10)."""
        category_scores = {
            'security': 9,
            'automation': 8,
            'implementation': 7,
            'dependency-management': 6,
            'code-quality': 5,
            'technical-debt': 4,
            'performance': 7,
            'documentation': 3
        }
        return category_scores.get(item.category, 5)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time criticality (1-10)."""
        if item.category == 'security':
            return 9
        elif item.category == 'automation':
            return 7
        elif item.category == 'implementation':
            return 6
        return 4
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk reduction value (1-10)."""
        return 10 - (item.risk_level * 10)
    
    def _score_opportunity_enablement(self, item: ValueItem) -> float:
        """Score how much this enables other opportunities (1-10)."""
        enabler_categories = {'automation', 'implementation', 'dependency-management'}
        return 8 if item.category in enabler_categories else 3
    
    def _score_impact(self, item: ValueItem) -> float:
        """Score overall impact (1-10)."""
        return self._score_business_value(item)
    
    def _score_confidence(self, item: ValueItem) -> float:
        """Score execution confidence (1-10)."""
        if item.estimated_effort < 2:
            return 9  # High confidence for small tasks
        elif item.estimated_effort < 8:
            return 7  # Medium confidence for medium tasks
        else:
            return 5  # Lower confidence for large tasks
    
    def _score_technical_debt(self, item: ValueItem) -> float:
        """Score technical debt reduction value (1-100)."""
        debt_categories = {'technical-debt', 'code-quality', 'security'}
        if item.category in debt_categories:
            return min(item.estimated_effort * 10, 100)
        return 10
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 50
        return max(0, min(100, ((score - min_val) / (max_val - min_val)) * 100))
    
    def save_backlog(self, opportunities: List[ValueItem]) -> None:
        """Save value backlog to file."""
        backlog_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_items": len(opportunities),
            "next_best_item": asdict(opportunities[0]) if opportunities else None,
            "top_10_items": [asdict(item) for item in opportunities[:10]],
            "all_items": [asdict(item) for item in opportunities]
        }
        
        self.backlog_file.parent.mkdir(exist_ok=True)
        with open(self.backlog_file, 'w') as f:
            json.dump(backlog_data, f, indent=2)
    
    def get_next_best_item(self) -> Optional[ValueItem]:
        """Get the next highest-value item for execution."""
        opportunities = self.discover_opportunities()
        
        # Apply minimum score threshold
        min_score = self.config['scoring']['thresholds']['minScore']
        qualified_items = [item for item in opportunities if item.composite_score >= min_score]
        
        # Apply risk threshold
        max_risk = self.config['scoring']['thresholds']['maxRisk']
        safe_items = [item for item in qualified_items if item.risk_level <= max_risk]
        
        return safe_items[0] if safe_items else None


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    opportunities = engine.discover_opportunities()
    engine.save_backlog(opportunities)
    
    next_item = engine.get_next_best_item()
    if next_item:
        print(f"Next best value item: {next_item.title} (Score: {next_item.composite_score:.1f})")
    else:
        print("No qualifying value items found.")