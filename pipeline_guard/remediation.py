"""
RemediationStrategy — maps FailureCategory → ordered list of RemediationActions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .analyzer import FailureCategory


class RemediationAction(str, Enum):
    RETRY = "retry"
    CLEAR_CACHE = "clear_cache"
    INSTALL_DEP = "install_dep"
    INCREASE_TIMEOUT = "increase_timeout"
    RERUN_FLAKY = "rerun_flaky"
    INCREASE_MEMORY = "increase_memory"
    FIX_LINT = "fix_lint"
    ROLLBACK = "rollback"
    NOTIFY_TEAM = "notify_team"
    SKIP = "skip"


# Default remediation playbooks
_DEFAULT_PLAYBOOKS: dict[FailureCategory, list[RemediationAction]] = {
    FailureCategory.TEST_FAILURE: [
        RemediationAction.RETRY,
        RemediationAction.CLEAR_CACHE,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.BUILD_ERROR: [
        RemediationAction.CLEAR_CACHE,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.DEPENDENCY_MISSING: [
        RemediationAction.INSTALL_DEP,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.TIMEOUT: [
        RemediationAction.INCREASE_TIMEOUT,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.FLAKY_TEST: [
        RemediationAction.RERUN_FLAKY,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.OOM: [
        RemediationAction.INCREASE_MEMORY,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.LINT_ERROR: [
        RemediationAction.FIX_LINT,
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
    FailureCategory.UNKNOWN: [
        RemediationAction.RETRY,
        RemediationAction.NOTIFY_TEAM,
    ],
}


@dataclass
class RemediationStrategy:
    """Maps failure categories to ordered action lists.

    Custom playbooks can be supplied at construction time; missing categories
    fall back to built-in defaults.
    """

    custom_playbooks: dict[FailureCategory, list[RemediationAction]] = field(
        default_factory=dict
    )

    def actions_for(self, category: FailureCategory) -> list[RemediationAction]:
        """Return the ordered list of actions to try for *category*."""
        if category in self.custom_playbooks:
            return list(self.custom_playbooks[category])
        return list(_DEFAULT_PLAYBOOKS.get(category, _DEFAULT_PLAYBOOKS[FailureCategory.UNKNOWN]))

    def all_categories(self) -> list[FailureCategory]:
        """Return all categories that have playbooks (custom + default)."""
        seen = set(_DEFAULT_PLAYBOOKS.keys()) | set(self.custom_playbooks.keys())
        return sorted(seen, key=lambda c: c.value)
