import os
import sys
import unittest
import tempfile
import textwrap
import time

from mlx_core import MLXCore, VerificationResult


class TestRunIntegration(unittest.TestCase):
    def setUp(self):
        # Isolate MLXCore state in a temp .mlx directory
        self.tmp_mlx_dir = tempfile.TemporaryDirectory()
        self.core = MLXCore(self.tmp_mlx_dir.name)

    def tearDown(self):
        self.tmp_mlx_dir.cleanup()

    def _write_experiment(self, src_text):
        tmpdir = tempfile.TemporaryDirectory()
        path = os.path.join(tmpdir.name, "exp.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(src_text))
        return tmpdir, path

    def test_run_experiment_validated(self):
        src = """
        from decorators import trial, verify, VerificationResult

        @trial(1, {"x": [1, 2]})
        def t(x):
            return {"x": x, "val": x * 2}

        @verify(["t"])
        def check(results):
            # All runs should be success with a result
            runs = results.get("t", [])
            if not runs:
                return VerificationResult.INCONCLUSIVE
            ok = all(r.get("success") and ("result" in r) for r in runs)
            return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
        """
        tmpdir, exp_path = self._write_experiment(src)
        try:
            h = self.core.create_new_hypothesis("H1")
            exp = self.core.add_experiment("E1", exp_path)

            run_id, run = self.core.run_experiment(str(exp.id)[:8])
            self.assertIsNotNone(run_id)
            self.assertEqual(run.status, "validated")
            # Experiment should now be validated
            self.assertEqual(self.core.find_experiment_by_id(str(exp.id)[:8]).status, "validated")
            # Ensure results captured for trial 't'
            self.assertIn("t", run.results)
            self.assertGreaterEqual(len(run.results["t"]), 1)
        finally:
            tmpdir.cleanup()

    def test_verification_mapping_multi(self):
        src = """
        from decorators import trial, verify, VerificationResult

        @trial(1, {"x": [1]})
        def t(x):
            return x

        @verify(["t"])  # -> SUPPORTS
        def v_support(results):
            return "supports"

        @verify(["t"])  # -> REFUTES
        def v_refute(results):
            return False

        @verify(["t"])  # -> INCONCLUSIVE
        def v_none(results):
            return None

        @verify(["t"])  # -> INVALID (unknown string)
        def v_weird(results):
            return "something-else"
        """
        tmpdir, exp_path = self._write_experiment(src)
        try:
            h = self.core.create_new_hypothesis("H2")
            exp = self.core.add_experiment("E2", exp_path)

            run_id, run = self.core.run_experiment(str(exp.id)[:8])
            # Mapping expectations
            vr = run.verification_results
            self.assertEqual(vr.get("v_support"), VerificationResult.SUPPORTS)
            self.assertEqual(vr.get("v_refute"), VerificationResult.REFUTES)
            self.assertEqual(vr.get("v_none"), VerificationResult.INCONCLUSIVE)
            self.assertEqual(vr.get("v_weird"), VerificationResult.INVALID)
            # Any REFUTES -> overall invalidated (takes precedence over INVALID)
            self.assertEqual(run.status, "invalidated")
        finally:
            tmpdir.cleanup()

    def test_orphan_cleanup_removes_old_dirs(self):
        # Ensure temp_root_dir exists
        os.makedirs(self.core.temp_root_dir, exist_ok=True)
        old_dir = os.path.join(self.core.temp_root_dir, "run_old")
        os.makedirs(old_dir, exist_ok=True)
        # Set mtime to old
        past = time.time() - 3600
        os.utime(old_dir, (past, past))
        # With max_age_hours=0, any dir should be removed
        self.core.cleanup_orphan_temp_dirs(max_age_hours=0)
        self.assertFalse(os.path.isdir(old_dir))

    def test_persistence_store_and_load(self):
        # Create and persist a simple tree
        h = self.core.create_new_hypothesis("Root Hypothesis")
        # Head should be this new hypothesis
        head_id_prefix = str(self.core.head_hypothesis.id)[:8]
        # Add an experiment (no need to run it)
        tmpdir, exp_path = self._write_experiment("from decorators import trial\n@trial(1, {\"x\": [1]})\ndef t(x):\n    return x\n")
        try:
            self.core.add_experiment("Persist Exp", exp_path)
            # Persist
            self.core.store_hypothesis_tree()

            # Reload in new core instance
            core2 = MLXCore(self.tmp_mlx_dir.name)
            self.assertIsNotNone(core2.root_hypothesis)
            self.assertIsNotNone(core2.head_hypothesis)
            self.assertTrue(str(core2.head_hypothesis.id).startswith(head_id_prefix))
            # Ensure experiment exists under root
            self.assertGreaterEqual(len(core2.root_hypothesis.experiments), 1)
        finally:
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
