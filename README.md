# self-healing-pipeline-guard

A lightweight CI/CD failure detector and auto-remediation engine — **stdlib only, zero dependencies**.

## What it does

When your pipeline breaks, this tool:

1. **Detects** which steps failed (FAIL or TIMEOUT status)
2. **Classifies** the failure by scanning `error_output` with regex patterns
3. **Selects** an ordered remediation playbook for that failure category
4. **Attempts** each action in order (via a pluggable hook) until the step heals
5. **Prevents infinite loops** by tracking (step, category) retry history
6. **Reports** a structured `HealingReport` with what happened to every step

---

## Failure categories

| Category | Example triggers |
|---|---|
| `TEST_FAILURE` | `pytest: 2 failed`, `AssertionError` |
| `BUILD_ERROR` | `compilation error`, `SyntaxError`, `make error` |
| `DEPENDENCY_MISSING` | `ModuleNotFoundError`, `npm ERR! 404`, `ImportError` |
| `TIMEOUT` | `timed out`, `deadline exceeded`, `TIMEOUT` status |
| `FLAKY_TEST` | `flaky`, `race condition`, `intermittent` |
| `OOM` | `out of memory`, `OutOfMemoryError`, `malloc failed` |
| `LINT_ERROR` | `eslint`, `flake8`, `mypy error` |
| `UNKNOWN` | No pattern matched |

## Remediation playbooks (defaults)

| Category | Actions (in order) |
|---|---|
| TEST_FAILURE | retry → clear_cache → notify_team |
| BUILD_ERROR | clear_cache → retry → notify_team |
| DEPENDENCY_MISSING | install_dep → retry → notify_team |
| TIMEOUT | increase_timeout → retry → notify_team |
| FLAKY_TEST | rerun_flaky → retry → notify_team |
| OOM | increase_memory → retry → notify_team |
| LINT_ERROR | fix_lint → retry → notify_team |
| UNKNOWN | retry → notify_team |

---

## Quick start

```python
from pipeline_guard import (
    PipelineStep, StepStatus,
    PipelineGuard, RemediationAction,
)

# Define your pipeline steps (populate from CI API / webhook payload)
steps = [
    PipelineStep("lint",  StepStatus.PASS, duration_s=4.1),
    PipelineStep("build", StepStatus.PASS, duration_s=92.3),
    PipelineStep("test",  StepStatus.FAIL, duration_s=120.0,
                 error_output="pytest: 3 failed, 47 passed\nAssertionError: expected 200 got 500"),
    PipelineStep("deps",  StepStatus.FAIL, duration_s=8.0,
                 error_output="ModuleNotFoundError: No module named 'httpx'"),
]

# Plug in your actual remediation logic
def my_hook(step, action):
    if action == RemediationAction.RETRY:
        success = rerun_step_in_ci(step.name)   # your CI call here
        return StepStatus.PASS if success else StepStatus.FAIL
    if action == RemediationAction.INSTALL_DEP:
        run_cmd("pip install -r requirements.txt")
    if action == RemediationAction.CLEAR_CACHE:
        run_cmd("rm -rf .cache __pycache__")
    return step.status  # not yet healed

guard = PipelineGuard(
    pipeline_name="my-service-ci",
    remediation_hook=my_hook,
    max_retries_per_step=3,
)
report = guard.run(steps)
print(report.summary())
print(report.to_json())
```

### Example healing report output

```
=== Healing Report: my-service-ci ===
Status      : ✅ FULLY HEALED
Total steps : 4
Failed      : 2
Healed      : 2

--- Step Details ---
  ✅ [test]  category=TEST_FAILURE  actions=['retry', 'clear_cache']  final=PASS
  ✅ [deps]  category=DEPENDENCY_MISSING  actions=['install_dep', 'retry']  final=PASS
```

```json
{
  "pipeline_name": "my-service-ci",
  "total_steps": 4,
  "failed_steps": 2,
  "healed_steps": 2,
  "fully_healed": true,
  "records": [
    {
      "step": "test",
      "original_status": "FAIL",
      "category": "TEST_FAILURE",
      "matched_pattern": "pytest.*fail",
      "actions_tried": ["retry", "clear_cache"],
      "final_status": "PASS",
      "healed": true,
      "loop_aborted": false
    },
    {
      "step": "deps",
      "original_status": "FAIL",
      "category": "DEPENDENCY_MISSING",
      "matched_pattern": "ModuleNotFoundError",
      "actions_tried": ["install_dep", "retry"],
      "final_status": "PASS",
      "healed": true,
      "loop_aborted": false
    }
  ]
}
```

---

## Loop prevention

`PipelineGuard` tracks `(step_name, category)` → attempt count.  
Once a step+category pair hits `max_retries_per_step` (default: 3), further retries are **aborted** — the `HealingReport` record will have `loop_aborted=True`.

Call `guard.reset_history()` between pipeline runs to reset the counter.

---

## Custom playbooks

```python
from pipeline_guard import RemediationStrategy, RemediationAction
from pipeline_guard.analyzer import FailureCategory

strategy = RemediationStrategy(
    custom_playbooks={
        FailureCategory.FLAKY_TEST: [
            RemediationAction.RERUN_FLAKY,
            RemediationAction.NOTIFY_TEAM,
        ]
    }
)
guard = PipelineGuard(strategy=strategy, ...)
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

43 tests, stdlib only, no external dependencies.

---

## Project structure

```
pipeline_guard/
  models.py        # PipelineStep, StepStatus
  analyzer.py      # FailureClassifier (pattern matching → FailureCategory)
  remediation.py   # RemediationStrategy (category → action list)
  guard.py         # PipelineGuard (orchestration + loop prevention)
  report.py        # HealingReport, StepHealingRecord
tests/
  test_pipeline_guard.py   # 43 tests
```

## License

MIT
