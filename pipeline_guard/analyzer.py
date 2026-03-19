"""
FailureClassifier — scans error_output with regex patterns to categorise CI/CD failures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .models import PipelineStep, StepStatus


class FailureCategory(str, Enum):
    TEST_FAILURE = "TEST_FAILURE"
    BUILD_ERROR = "BUILD_ERROR"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    TIMEOUT = "TIMEOUT"
    FLAKY_TEST = "FLAKY_TEST"
    OOM = "OOM"
    LINT_ERROR = "LINT_ERROR"
    UNKNOWN = "UNKNOWN"


# (category, [regex patterns]) — first match wins
_PATTERNS: list[tuple[FailureCategory, list[str]]] = [
    (
        FailureCategory.OOM,
        [
            r"out of memory",
            r"oom.?kill",
            r"memory limit exceeded",
            r"killed.*memory",
            r"java\.lang\.OutOfMemoryError",
            r"malloc.*failed",
        ],
    ),
    (
        FailureCategory.TIMEOUT,
        [
            r"timed? ?out",
            r"deadline exceeded",
            r"connection timed? ?out",
            r"process timed? ?out",
            r"exceeded.*timeout",
            r"context deadline exceeded",
        ],
    ),
    (
        FailureCategory.DEPENDENCY_MISSING,
        [
            r"module not found",
            r"cannot find module",
            r"no module named",
            r"package.*not found",
            r"could not resolve",
            r"dependency.*missing",
            r"ImportError",
            r"ModuleNotFoundError",
            r"pip.*no matching distribution",
            r"npm ERR! 404",
            r"unable to locate package",
        ],
    ),
    (
        FailureCategory.FLAKY_TEST,
        [
            r"flaky",
            r"intermittent",
            r"race condition",
            r"non-deterministic",
            r"test.*randomly fail",
            r"sometimes fail",
            r"flakytest",
        ],
    ),
    (
        FailureCategory.LINT_ERROR,
        [
            r"lint.*error",
            r"linting failed",
            r"eslint",
            r"flake8",
            r"pylint",
            r"ruff.*error",
            r"mypy.*error",
            r"type error",
            r"style.*violation",
        ],
    ),
    (
        FailureCategory.TEST_FAILURE,
        [
            r"test.*fail",
            r"assertion.*error",
            r"AssertionError",
            r"FAILED.*test",
            r"pytest.*fail",
            r"jest.*fail",
            r"spec.*fail",
            r"expected.*got",
            r"unittest.*fail",
        ],
    ),
    (
        FailureCategory.BUILD_ERROR,
        [
            r"build.*fail",
            r"compilation.*error",
            r"compile.*error",
            r"syntax.*error",
            r"SyntaxError",
            r"make.*error",
            r"gradle.*fail",
            r"maven.*fail",
            r"cargo.*error",
            r"linker.*error",
        ],
    ),
]


@dataclass
class ClassificationResult:
    category: FailureCategory
    matched_pattern: Optional[str] = None
    confidence: float = 1.0


class FailureClassifier:
    """Classifies a failed PipelineStep into a FailureCategory.

    Uses a priority-ordered set of regex patterns.  The step's ``status``
    field is used as a tie-breaker / override for TIMEOUT when no pattern
    matches the error output.
    """

    def __init__(self) -> None:
        # pre-compile all patterns for speed
        self._compiled: list[tuple[FailureCategory, list[re.Pattern[str]]]] = [
            (cat, [re.compile(p, re.IGNORECASE) for p in patterns])
            for cat, patterns in _PATTERNS
        ]

    def classify(self, step: PipelineStep) -> ClassificationResult:
        """Return the best-matching FailureCategory for *step*."""
        text = step.error_output or ""

        for category, compiled_patterns in self._compiled:
            for pattern in compiled_patterns:
                m = pattern.search(text)
                if m:
                    return ClassificationResult(
                        category=category,
                        matched_pattern=pattern.pattern,
                    )

        # Fall back on status field
        if step.status == StepStatus.TIMEOUT:
            return ClassificationResult(
                category=FailureCategory.TIMEOUT,
                matched_pattern=None,
                confidence=0.8,
            )

        return ClassificationResult(category=FailureCategory.UNKNOWN, confidence=0.0)

    def classify_all(self, steps: list[PipelineStep]) -> list[ClassificationResult]:
        """Classify every *failed* step in *steps* (PASS/SKIPPED are ignored)."""
        results = []
        for step in steps:
            if step.failed():
                results.append(self.classify(step))
        return results


# Convenience alias used by the public __init__ export
FailureAnalyzer = FailureClassifier
