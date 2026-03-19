"""
Comprehensive tests for self-healing-pipeline-guard.
Covers: pattern matching, action selection, loop prevention, report structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from pipeline_guard.models import PipelineStep, StepStatus
from pipeline_guard.analyzer import FailureClassifier, FailureCategory
from pipeline_guard.remediation import RemediationAction, RemediationStrategy
from pipeline_guard.guard import PipelineGuard
from pipeline_guard.report import HealingReport, StepHealingRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(name: str, status: StepStatus = StepStatus.FAIL, error: str = "", duration: float = 0.0) -> PipelineStep:
    return PipelineStep(name=name, status=status, duration_s=duration, error_output=error)


def _always_heal(step, action):
    """Hook that always heals on first action."""
    return StepStatus.PASS


def _never_heal(step, action):
    """Hook that never heals."""
    return step.status


# ---------------------------------------------------------------------------
# PipelineStep tests
# ---------------------------------------------------------------------------

class TestPipelineStep(unittest.TestCase):

    def test_fail_status_is_failed(self):
        s = _step("build", StepStatus.FAIL)
        self.assertTrue(s.failed())

    def test_timeout_status_is_failed(self):
        s = _step("test", StepStatus.TIMEOUT)
        self.assertTrue(s.failed())

    def test_pass_is_not_failed(self):
        s = _step("lint", StepStatus.PASS)
        self.assertFalse(s.failed())

    def test_skipped_is_not_failed(self):
        s = _step("deploy", StepStatus.SKIPPED)
        self.assertFalse(s.failed())

    def test_repr_contains_name_and_status(self):
        s = _step("foo", StepStatus.FAIL)
        r = repr(s)
        self.assertIn("foo", r)
        self.assertIn("FAIL", r)

    def test_duration_stored(self):
        s = _step("x", duration=42.5)
        self.assertAlmostEqual(s.duration_s, 42.5)


# ---------------------------------------------------------------------------
# FailureClassifier — pattern matching tests
# ---------------------------------------------------------------------------

class TestFailureClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = FailureClassifier()

    def _classify(self, error: str, status: StepStatus = StepStatus.FAIL) -> FailureCategory:
        return self.clf.classify(_step("s", status, error)).category

    # OOM
    def test_oom_out_of_memory(self):
        self.assertEqual(self._classify("Process out of memory"), FailureCategory.OOM)

    def test_oom_java(self):
        self.assertEqual(self._classify("java.lang.OutOfMemoryError: GC overhead"), FailureCategory.OOM)

    def test_oom_malloc(self):
        self.assertEqual(self._classify("malloc: failed to allocate 4GB"), FailureCategory.OOM)

    # TIMEOUT
    def test_timeout_timed_out(self):
        self.assertEqual(self._classify("Job timed out after 30 minutes"), FailureCategory.TIMEOUT)

    def test_timeout_deadline(self):
        self.assertEqual(self._classify("context deadline exceeded"), FailureCategory.TIMEOUT)

    def test_timeout_via_status(self):
        # No matching text but status=TIMEOUT
        result = self.clf.classify(_step("s", StepStatus.TIMEOUT, ""))
        self.assertEqual(result.category, FailureCategory.TIMEOUT)
        self.assertAlmostEqual(result.confidence, 0.8)

    # DEPENDENCY_MISSING
    def test_dep_module_not_found(self):
        self.assertEqual(self._classify("ModuleNotFoundError: No module named 'requests'"), FailureCategory.DEPENDENCY_MISSING)

    def test_dep_npm_404(self):
        self.assertEqual(self._classify("npm ERR! 404 Not Found - GET https://registry.npmjs.org/left-pad"), FailureCategory.DEPENDENCY_MISSING)

    def test_dep_import_error(self):
        self.assertEqual(self._classify("ImportError: cannot import name 'foo'"), FailureCategory.DEPENDENCY_MISSING)

    # FLAKY_TEST
    def test_flaky_keyword(self):
        self.assertEqual(self._classify("flaky test detected: test_login"), FailureCategory.FLAKY_TEST)

    def test_flaky_race_condition(self):
        self.assertEqual(self._classify("race condition in database cleanup"), FailureCategory.FLAKY_TEST)

    # LINT_ERROR
    def test_lint_eslint(self):
        self.assertEqual(self._classify("eslint: 3 errors found"), FailureCategory.LINT_ERROR)

    def test_lint_flake8(self):
        self.assertEqual(self._classify("flake8: E501 line too long"), FailureCategory.LINT_ERROR)

    def test_lint_mypy(self):
        self.assertEqual(self._classify("mypy error: incompatible types"), FailureCategory.LINT_ERROR)

    # TEST_FAILURE
    def test_test_failure_pytest(self):
        self.assertEqual(self._classify("pytest: 2 failed, 10 passed"), FailureCategory.TEST_FAILURE)

    def test_test_failure_assertion(self):
        self.assertEqual(self._classify("AssertionError: expected 42 got 0"), FailureCategory.TEST_FAILURE)

    # BUILD_ERROR
    def test_build_error_compilation(self):
        self.assertEqual(self._classify("compilation error: undefined symbol"), FailureCategory.BUILD_ERROR)

    def test_build_error_syntax(self):
        self.assertEqual(self._classify("SyntaxError: invalid syntax"), FailureCategory.BUILD_ERROR)

    # UNKNOWN
    def test_unknown_empty_output(self):
        self.assertEqual(self._classify(""), FailureCategory.UNKNOWN)

    def test_unknown_garbage(self):
        self.assertEqual(self._classify("something completely random xyz123"), FailureCategory.UNKNOWN)

    # classify_all only includes failed steps
    def test_classify_all_skips_passing(self):
        steps = [
            _step("a", StepStatus.PASS),
            _step("b", StepStatus.FAIL, "pytest failed"),
            _step("c", StepStatus.SKIPPED),
        ]
        results = self.clf.classify_all(steps)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, FailureCategory.TEST_FAILURE)


# ---------------------------------------------------------------------------
# RemediationStrategy tests
# ---------------------------------------------------------------------------

class TestRemediationStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = RemediationStrategy()

    def test_test_failure_actions(self):
        actions = self.strategy.actions_for(FailureCategory.TEST_FAILURE)
        self.assertIn(RemediationAction.RETRY, actions)
        self.assertIn(RemediationAction.CLEAR_CACHE, actions)

    def test_dependency_missing_actions(self):
        actions = self.strategy.actions_for(FailureCategory.DEPENDENCY_MISSING)
        self.assertEqual(actions[0], RemediationAction.INSTALL_DEP)

    def test_oom_actions(self):
        actions = self.strategy.actions_for(FailureCategory.OOM)
        self.assertEqual(actions[0], RemediationAction.INCREASE_MEMORY)

    def test_custom_playbook_overrides_default(self):
        custom = RemediationStrategy(
            custom_playbooks={FailureCategory.TEST_FAILURE: [RemediationAction.SKIP]}
        )
        actions = custom.actions_for(FailureCategory.TEST_FAILURE)
        self.assertEqual(actions, [RemediationAction.SKIP])

    def test_unknown_falls_back_gracefully(self):
        actions = self.strategy.actions_for(FailureCategory.UNKNOWN)
        self.assertIn(RemediationAction.RETRY, actions)


# ---------------------------------------------------------------------------
# PipelineGuard — orchestration + loop prevention
# ---------------------------------------------------------------------------

class TestPipelineGuard(unittest.TestCase):

    def _guard(self, hook=None, max_retries=3):
        return PipelineGuard(
            pipeline_name="test-pipeline",
            remediation_hook=hook or _never_heal,
            max_retries_per_step=max_retries,
        )

    def test_no_failures_produces_empty_report(self):
        steps = [_step("a", StepStatus.PASS), _step("b", StepStatus.SKIPPED)]
        report = self._guard().run(steps)
        self.assertEqual(report.failed_steps, 0)
        self.assertEqual(report.healed_steps, 0)
        self.assertEqual(len(report.records), 0)

    def test_healing_hook_called(self):
        called = []

        def hook(step, action):
            called.append((step.name, action))
            return StepStatus.PASS  # heal on first try

        steps = [_step("build", StepStatus.FAIL, "pytest failed")]
        guard = self._guard(hook=hook)
        report = guard.run(steps)
        self.assertEqual(report.healed_steps, 1)
        self.assertTrue(len(called) >= 1)

    def test_fully_healed_flag(self):
        steps = [_step("x", StepStatus.FAIL, "pytest failed")]
        report = PipelineGuard(remediation_hook=_always_heal).run(steps)
        self.assertTrue(report.fully_healed)

    def test_loop_prevention_halts_retry(self):
        """Same (step, category) pair cannot exceed max_retries_per_step."""
        call_count = [0]

        def counting_hook(step, action):
            call_count[0] += 1
            return StepStatus.FAIL  # never heals

        guard = self._guard(hook=counting_hook, max_retries=2)
        steps = [_step("t", StepStatus.FAIL, "pytest failed")]

        # Run twice — second run should be aborted
        guard.run(steps)
        report2 = guard.run(steps)

        record = report2.records[0]
        self.assertTrue(record.loop_aborted)
        self.assertFalse(record.healed)
        # call_count should not have grown on second run
        calls_after_first = call_count[0]
        guard.run(steps)  # third run
        # still aborted — no new calls
        self.assertEqual(call_count[0], calls_after_first)

    def test_reset_history_re_enables_retry(self):
        call_count = [0]

        def hook(step, action):
            call_count[0] += 1
            return StepStatus.FAIL

        guard = self._guard(hook=hook, max_retries=1)
        steps = [_step("t", StepStatus.FAIL, "pytest failed")]
        guard.run(steps)  # uses up 1 retry
        guard.reset_history()
        guard.run(steps)  # should retry again
        self.assertGreater(call_count[0], 1)

    def test_report_total_steps_count(self):
        steps = [_step("a", StepStatus.PASS), _step("b", StepStatus.FAIL, "build failed")]
        report = self._guard().run(steps)
        self.assertEqual(report.total_steps, 2)
        self.assertEqual(report.failed_steps, 1)


# ---------------------------------------------------------------------------
# HealingReport tests
# ---------------------------------------------------------------------------

class TestHealingReport(unittest.TestCase):

    def _make_report(self, healed=1, failed=1):
        step = _step("test", StepStatus.FAIL, "pytest failed")
        record = StepHealingRecord(
            step=step,
            category=FailureCategory.TEST_FAILURE,
            matched_pattern=r"pytest.*fail",
            actions_tried=[RemediationAction.RETRY],
            final_status=StepStatus.PASS if healed else StepStatus.FAIL,
        )
        return HealingReport(
            pipeline_name="ci",
            total_steps=5,
            failed_steps=failed,
            healed_steps=healed,
            records=[record],
        )

    def test_fully_healed(self):
        r = self._make_report(healed=1, failed=1)
        self.assertTrue(r.fully_healed)

    def test_partial_heal(self):
        step = _step("test", StepStatus.FAIL)
        rec2 = StepHealingRecord(
            step=step,
            category=FailureCategory.BUILD_ERROR,
            matched_pattern=None,
            actions_tried=[],
            final_status=StepStatus.FAIL,
        )
        r = self._make_report(healed=1, failed=2)
        r.records.append(rec2)
        self.assertTrue(r.partial_heal)
        self.assertFalse(r.fully_healed)

    def test_summary_contains_pipeline_name(self):
        r = self._make_report()
        self.assertIn("ci", r.summary())

    def test_to_dict_structure(self):
        r = self._make_report()
        d = r.to_dict()
        self.assertIn("pipeline_name", d)
        self.assertIn("records", d)
        self.assertIsInstance(d["records"], list)

    def test_to_json_valid(self):
        import json
        r = self._make_report()
        parsed = json.loads(r.to_json())
        self.assertEqual(parsed["pipeline_name"], "ci")


if __name__ == "__main__":
    unittest.main(verbosity=2)
