#!/usr/bin/env python3
"""
Self-Healing Pipeline Guard - Simple Demonstration
A standalone demo showcasing autonomous failure detection and healing capabilities.
"""

import json
import time
import random
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


class FailureType(Enum):
    """Types of pipeline failures."""
    FLAKY_TEST = "flaky_test"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    NETWORK_TIMEOUT = "network_timeout"
    COMPILATION_ERROR = "compilation_error"


class SeverityLevel(Enum):
    """Failure severity levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class HealingStatus(Enum):
    """Healing operation status."""
    SUCCESSFUL = "successful"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class FailureEvent:
    """Represents a detected failure event."""
    id: str
    timestamp: str
    job_id: str
    repository: str
    branch: str
    failure_type: str
    severity: str
    confidence: float
    raw_logs: str
    remediation_suggestions: List[str]


@dataclass
class HealingAction:
    """Represents a healing action."""
    id: str
    strategy: str
    description: str
    estimated_duration: float
    success_probability: float


@dataclass
class HealingResult:
    """Results of a healing operation."""
    healing_id: str
    status: str
    actions_executed: List[str]
    actions_successful: List[str] 
    actions_failed: List[str]
    total_duration: float


class SimpleFailureDetector:
    """Simplified failure detection for demo."""
    
    def __init__(self):
        self.patterns = {
            "timeout": {
                "type": FailureType.NETWORK_TIMEOUT,
                "keywords": ["timeout", "connection refused", "network"],
                "suggestions": ["retry_with_backoff", "check_connectivity"]
            },
            "memory": {
                "type": FailureType.RESOURCE_EXHAUSTION,
                "keywords": ["OutOfMemoryError", "OOMKilled", "memory"],
                "suggestions": ["increase_memory", "optimize_resources"]
            },
            "dependency": {
                "type": FailureType.DEPENDENCY_FAILURE,
                "keywords": ["dependency", "package not found", "npm ERR"],
                "suggestions": ["clear_cache", "update_dependencies"]
            },
            "test": {
                "type": FailureType.FLAKY_TEST,
                "keywords": ["test failed", "assertion", "flaky"],
                "suggestions": ["retry_with_isolation", "increase_timeout"]
            },
            "compile": {
                "type": FailureType.COMPILATION_ERROR,
                "keywords": ["compilation failed", "SyntaxError", "build error"],
                "suggestions": ["lint_check", "syntax_review"]
            }
        }
    
    def detect_failure(self, job_id: str, repository: str, branch: str, logs: str) -> FailureEvent:
        """Detect failure type from logs."""
        # Simple pattern matching
        best_match = None
        best_score = 0
        
        logs_lower = logs.lower()
        for pattern_name, pattern in self.patterns.items():
            score = sum(1 for keyword in pattern["keywords"] if keyword in logs_lower)
            if score > best_score:
                best_score = score
                best_match = pattern
        
        if best_match:
            failure_type = best_match["type"]
            confidence = min(0.95, best_score * 0.3)
            suggestions = best_match["suggestions"]
        else:
            failure_type = FailureType.COMPILATION_ERROR
            confidence = 0.5
            suggestions = ["manual_review"]
        
        # Determine severity
        severity = SeverityLevel.HIGH if confidence > 0.8 else SeverityLevel.MEDIUM
        
        return FailureEvent(
            id=f"{job_id}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            job_id=job_id,
            repository=repository,
            branch=branch,
            failure_type=failure_type.value,
            severity=severity.value,
            confidence=confidence,
            raw_logs=logs,
            remediation_suggestions=suggestions
        )


class SimpleHealingEngine:
    """Simplified healing engine for demo."""
    
    def __init__(self):
        self.strategies = {
            "retry_with_backoff": {
                "description": "Retry failed operation with exponential backoff",
                "duration": 2.0,
                "success_rate": 0.8
            },
            "increase_memory": {
                "description": "Increase memory allocation for job",
                "duration": 1.5,
                "success_rate": 0.9
            },
            "clear_cache": {
                "description": "Clear dependency and build caches",
                "duration": 1.0,
                "success_rate": 0.7
            },
            "update_dependencies": {
                "description": "Update project dependencies",
                "duration": 5.0,
                "success_rate": 0.6
            },
            "retry_with_isolation": {
                "description": "Retry test with environment isolation",
                "duration": 2.5,
                "success_rate": 0.85
            },
            "check_connectivity": {
                "description": "Verify network connectivity and DNS",
                "duration": 1.0,
                "success_rate": 0.75
            },
            "lint_check": {
                "description": "Run linting and syntax validation",
                "duration": 0.8,
                "success_rate": 0.9
            }
        }
    
    def create_healing_actions(self, failure_event: FailureEvent) -> List[HealingAction]:
        """Create healing actions for failure."""
        actions = []
        
        for suggestion in failure_event.remediation_suggestions:
            if suggestion in self.strategies:
                strategy = self.strategies[suggestion]
                action = HealingAction(
                    id=f"{suggestion}_{int(time.time())}",
                    strategy=suggestion,
                    description=strategy["description"],
                    estimated_duration=strategy["duration"],
                    success_probability=strategy["success_rate"]
                )
                actions.append(action)
        
        return actions
    
    def execute_healing_action(self, action: HealingAction) -> bool:
        """Execute a healing action (simulated)."""
        print(f"  ⚡ Executing: {action.description}")
        
        # Simulate execution time
        time.sleep(min(0.5, action.estimated_duration * 0.1))
        
        # Simulate success/failure based on probability
        success = random.random() < action.success_probability
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"    {status} - {action.strategy}")
        
        return success
    
    def heal_failure(self, failure_event: FailureEvent) -> HealingResult:
        """Execute complete healing process."""
        print(f"\n🏥 HEALING FAILURE: {failure_event.failure_type}")
        print(f"   Job: {failure_event.job_id}")
        print(f"   Confidence: {failure_event.confidence:.2f}")
        
        actions = self.create_healing_actions(failure_event)
        
        result = HealingResult(
            healing_id=f"heal_{int(time.time())}",
            status=HealingStatus.FAILED.value,
            actions_executed=[],
            actions_successful=[],
            actions_failed=[],
            total_duration=0.0
        )
        
        start_time = time.time()
        
        print(f"\n📋 Executing {len(actions)} healing actions:")
        
        for action in actions:
            result.actions_executed.append(action.id)
            
            success = self.execute_healing_action(action)
            
            if success:
                result.actions_successful.append(action.id)
            else:
                result.actions_failed.append(action.id)
        
        result.total_duration = time.time() - start_time
        
        # Determine overall status
        if len(result.actions_successful) == len(actions):
            result.status = HealingStatus.SUCCESSFUL.value
        elif len(result.actions_successful) > 0:
            result.status = HealingStatus.PARTIAL.value
        
        success_rate = len(result.actions_successful) / len(actions) * 100 if actions else 0
        print(f"\n🎯 HEALING COMPLETE: {result.status}")
        print(f"   Success Rate: {success_rate:.0f}%")
        print(f"   Duration: {result.total_duration:.1f}s")
        
        return result


def simulate_failure_scenarios():
    """Simulate various pipeline failure scenarios."""
    
    failure_scenarios = [
        {
            "job_id": "build_123",
            "repository": "app-frontend",
            "branch": "feature/auth",
            "logs": """
            npm ERR! code ENETUNREACH
            npm ERR! network connection timeout
            npm ERR! network This is most likely not a problem with npm itself
            npm ERR! network and is related to network connectivity.
            """
        },
        {
            "job_id": "test_456",
            "repository": "api-service",
            "branch": "main",
            "logs": """
            java.lang.OutOfMemoryError: Java heap space
            	at com.example.service.DataProcessor.process(DataProcessor.java:45)
            	at com.example.service.BatchJob.execute(BatchJob.java:123)
            Container killed due to OOMKilled
            """
        },
        {
            "job_id": "unit_789",
            "repository": "web-ui",
            "branch": "develop",
            "logs": """
            Test failed: should login user successfully
            AssertionError: expected element to be visible but was hidden
            This test is flaky and fails intermittently
            Timeout exceeded waiting for element
            """
        },
        {
            "job_id": "compile_101",
            "repository": "backend-api",
            "branch": "hotfix/security",
            "logs": """
            SyntaxError: unexpected token at line 45
            compilation failed with 3 errors
            build error in src/auth/validator.js
            Unexpected character '{'
            """
        },
        {
            "job_id": "deploy_202",
            "repository": "data-pipeline",
            "branch": "release/v2.1",
            "logs": """
            dependency 'pandas==1.5.0' not found
            package installation failed
            npm ERR! dependency tree resolution failed
            Version conflict detected
            """
        }
    ]
    
    return failure_scenarios


def main():
    """Main demonstration function."""
    print("🚀 SELF-HEALING PIPELINE GUARD - AUTONOMOUS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize components
    detector = SimpleFailureDetector()
    healing_engine = SimpleHealingEngine()
    
    # Get failure scenarios
    scenarios = simulate_failure_scenarios()
    
    healing_results = []
    total_failures = len(scenarios)
    successful_healings = 0
    
    print(f"\n🔍 ANALYZING {total_failures} PIPELINE FAILURES")
    print("-" * 40)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{total_failures}] PROCESSING FAILURE")
        
        # Detect failure
        failure_event = detector.detect_failure(
            scenario["job_id"],
            scenario["repository"], 
            scenario["branch"],
            scenario["logs"]
        )
        
        print(f"🚨 DETECTED: {failure_event.failure_type}")
        print(f"   Repository: {failure_event.repository}")
        print(f"   Branch: {failure_event.branch}")
        print(f"   Severity: {failure_event.severity}")
        
        # Execute healing
        healing_result = healing_engine.heal_failure(failure_event)
        healing_results.append(healing_result)
        
        if healing_result.status == HealingStatus.SUCCESSFUL.value:
            successful_healings += 1
        
        print("-" * 40)
    
    # Generate summary report
    print(f"\n📊 AUTONOMOUS HEALING SUMMARY")
    print("=" * 40)
    print(f"Total Failures Processed: {total_failures}")
    print(f"Successful Healings: {successful_healings}")
    print(f"Success Rate: {successful_healings/total_failures*100:.1f}%")
    
    # Calculate metrics
    total_duration = sum(result.total_duration for result in healing_results)
    avg_duration = total_duration / total_failures
    
    print(f"Total Processing Time: {total_duration:.1f}s")
    print(f"Average Healing Time: {avg_duration:.1f}s")
    
    # Strategy effectiveness
    strategy_stats = {}
    for result in healing_results:
        for action_id in result.actions_successful:
            strategy = action_id.split('_')[0]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0}
            strategy_stats[strategy]["total"] += 1
            strategy_stats[strategy]["successful"] += 1
        
        for action_id in result.actions_failed:
            strategy = action_id.split('_')[0]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0}
            strategy_stats[strategy]["total"] += 1
    
    print(f"\n🎯 STRATEGY EFFECTIVENESS")
    print("-" * 30)
    for strategy, stats in strategy_stats.items():
        success_rate = stats["successful"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{strategy:20}: {success_rate:5.1f}% ({stats['successful']}/{stats['total']})")
    
    print(f"\n✨ AUTONOMOUS SDLC EXECUTION COMPLETE")
    print("   Self-healing capabilities demonstrated successfully!")
    
    return {
        "total_failures": total_failures,
        "successful_healings": successful_healings,
        "success_rate": successful_healings/total_failures,
        "total_duration": total_duration,
        "strategy_effectiveness": strategy_stats
    }


if __name__ == "__main__":
    results = main()
    
    # Save results for analysis
    with open("/root/repo/demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: demo_results.json")