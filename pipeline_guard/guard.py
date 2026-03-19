"""
PipelineGuard — orchestrates detect → classify → remediate.

Loop-prevention: tracks (step_name, category) pairs in a history set.
If the same step + category has already been attempted, further retries
are aborted to prevent infinite loops.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .analyzer import FailureCategory, FailureClassifier
from .models import PipelineStep, StepStatus
from .remediation import RemediationAction, RemediationStrategy
from .report import HealingReport, StepHealingRecord

# Signature: (step, action) -> new StepStatus
RemediationHook = Callable[[PipelineStep, RemediationAction], StepStatus]


def _default_hook(step: PipelineStep, action: RemediationAction) -> StepStatus:
    """Default no-op hook: always returns the original status (healing not simulated)."""
    return step.status


@dataclass
class PipelineGuard:
    """Orchestrates failure detection, classification, and remediation.

    Parameters
    ----------
    pipeline_name:
        Human-readable name for the pipeline (used in reports).
    classifier:
        A FailureClassifier instance.  Created automatically if not supplied.
    strategy:
        A RemediationStrategy instance.  Created automatically if not supplied.
    remediation_hook:
        Callable that *actually performs* a remediation action and returns the
        new step status.  In production this would call your CI API / runner.
        Defaults to a no-op that always returns the original status.
    max_retries_per_step:
        Maximum number of action attempts per step before loop-abort.
    """

    pipeline_name: str = "pipeline"
    classifier: FailureClassifier = field(default_factory=FailureClassifier)
    strategy: RemediationStrategy = field(default_factory=RemediationStrategy)
    remediation_hook: RemediationHook = field(default=_default_hook)
    max_retries_per_step: int = 3

    # Internal history: maps (step_name, category) -> number of attempts
    _history: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)

    def _attempt_key(self, step: PipelineStep, category: FailureCategory) -> tuple[str, str]:
        return (step.name, category.value)

    def _can_attempt(self, step: PipelineStep, category: FailureCategory) -> bool:
        key = self._attempt_key(step, category)
        return self._history.get(key, 0) < self.max_retries_per_step

    def _record_attempt(self, step: PipelineStep, category: FailureCategory) -> None:
        key = self._attempt_key(step, category)
        self._history[key] = self._history.get(key, 0) + 1

    def reset_history(self) -> None:
        """Clear the retry-prevention history (useful between pipeline runs)."""
        self._history.clear()

    def run(self, steps: list[PipelineStep]) -> HealingReport:
        """Detect failures, classify, remediate, return a HealingReport."""
        failed = [s for s in steps if s.failed()]
        records: list[StepHealingRecord] = []

        for step in failed:
            result = self.classifier.classify(step)
            category = result.category
            actions = self.strategy.actions_for(category)

            actions_tried: list[RemediationAction] = []
            final_status = step.status
            loop_aborted = False

            if not self._can_attempt(step, category):
                loop_aborted = True
            else:
                for action in actions:
                    self._record_attempt(step, category)
                    actions_tried.append(action)
                    new_status = self.remediation_hook(step, action)
                    if new_status == StepStatus.PASS:
                        final_status = StepStatus.PASS
                        break
                    # If we've exhausted our retry budget mid-loop, stop
                    if not self._can_attempt(step, category):
                        loop_aborted = True
                        break

            records.append(
                StepHealingRecord(
                    step=step,
                    category=category,
                    matched_pattern=result.matched_pattern,
                    actions_tried=actions_tried,
                    final_status=final_status,
                    loop_aborted=loop_aborted,
                )
            )

        healed = sum(1 for r in records if r.healed)

        return HealingReport(
            pipeline_name=self.pipeline_name,
            total_steps=len(steps),
            failed_steps=len(failed),
            healed_steps=healed,
            records=records,
        )
