"""
HealingReport — structured output of a PipelineGuard healing cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from .analyzer import FailureCategory
from .models import PipelineStep, StepStatus
from .remediation import RemediationAction


@dataclass
class StepHealingRecord:
    """Record of classify + remediation for a single failed step."""

    step: PipelineStep
    category: FailureCategory
    matched_pattern: Optional[str]
    actions_tried: list[RemediationAction]
    final_status: StepStatus
    loop_aborted: bool = False  # True when retry loop was halted

    @property
    def healed(self) -> bool:
        return self.final_status == StepStatus.PASS

    def to_dict(self) -> dict:
        return {
            "step": self.step.name,
            "original_status": self.step.status.value,
            "category": self.category.value,
            "matched_pattern": self.matched_pattern,
            "actions_tried": [a.value for a in self.actions_tried],
            "final_status": self.final_status.value,
            "healed": self.healed,
            "loop_aborted": self.loop_aborted,
        }


@dataclass
class HealingReport:
    """Top-level report returned by PipelineGuard.run()."""

    pipeline_name: str
    total_steps: int
    failed_steps: int
    healed_steps: int
    records: list[StepHealingRecord] = field(default_factory=list)

    @property
    def fully_healed(self) -> bool:
        return self.healed_steps == self.failed_steps and self.failed_steps > 0

    @property
    def partial_heal(self) -> bool:
        return 0 < self.healed_steps < self.failed_steps

    def summary(self) -> str:
        status = (
            "✅ FULLY HEALED"
            if self.fully_healed
            else ("⚠️  PARTIALLY HEALED" if self.partial_heal else "❌ UNHEALED")
        )
        lines = [
            f"=== Healing Report: {self.pipeline_name} ===",
            f"Status      : {status}",
            f"Total steps : {self.total_steps}",
            f"Failed      : {self.failed_steps}",
            f"Healed      : {self.healed_steps}",
            "",
            "--- Step Details ---",
        ]
        for rec in self.records:
            heal_flag = "✅" if rec.healed else ("🔁 (loop aborted)" if rec.loop_aborted else "❌")
            lines.append(
                f"  {heal_flag} [{rec.step.name}]  category={rec.category.value}"
                f"  actions={[a.value for a in rec.actions_tried]}"
                f"  final={rec.final_status.value}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "total_steps": self.total_steps,
            "failed_steps": self.failed_steps,
            "healed_steps": self.healed_steps,
            "fully_healed": self.fully_healed,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
