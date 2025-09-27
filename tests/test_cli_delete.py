import os
import sys
import re
import unittest
import tempfile
import textwrap
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLX_CLI = os.path.join(REPO_ROOT, "mlx_init.py")


def run_cli(args, cwd):
    env = os.environ.copy()
    # Ensure repo root is on PYTHONPATH for decorators import in subprocess runs
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + existing if existing else "")
    return subprocess.run([sys.executable, MLX_CLI] + args, cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class TestDeleteCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cwd = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_experiment_file(self, content):
        path = os.path.join(self.cwd, "exp.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(content))
        return path

    def _create_simple_experiment(self):
        exp_src = """
        from decorators import trial, verify, VerificationResult
        @trial(1, {"x": [1]})
        def t(x):
            return x
        @verify(["t"])
        def check(results):
            runs = results.get("t", [])
            return VerificationResult.SUPPORTS if runs else VerificationResult.INCONCLUSIVE
        """
        return self._create_experiment_file(exp_src)

    def test_delete_hypothesis_empty(self):
        # init
        r = run_cli(["init"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # create hypothesis
        body = "My Hypo\nSome description"
        r = run_cli(["hypothesis", "create", "-m", body], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        m = re.search(r"ID ([0-9a-fA-F]{8})", r.stdout)
        self.assertIsNotNone(m, msg=r.stdout)
        hypo_id = m.group(1)

        # delete hypothesis
        r = run_cli(["hypothesis", "delete", hypo_id], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # show should fail after deletion
        r = run_cli(["hypothesis", "show", hypo_id], self.cwd)
        self.assertNotEqual(r.returncode, 0, msg=r.stdout)

    def test_delete_hypothesis_nonempty_requires_force(self):
        # init
        r = run_cli(["init"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # create hypothesis
        r = run_cli(["hypothesis", "create", "-m", "H with exp\nBody"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        m = re.search(r"ID ([0-9a-fA-F]{8})", r.stdout)
        self.assertIsNotNone(m, msg=r.stdout)
        hypo_id = m.group(1)

        # add experiment
        exp_path = self._create_simple_experiment()
        r = run_cli(["exp", "add", exp_path, "Delete Test Exp"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # deletion without --force should fail
        r = run_cli(["hypothesis", "delete", hypo_id], self.cwd)
        self.assertNotEqual(r.returncode, 0, msg=r.stdout)
        self.assertIn("Cannot delete hypothesis", r.stdout)

        # deletion with --force should succeed
        r = run_cli(["hypothesis", "delete", hypo_id, "--force"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

    def test_delete_experiment(self):
        # init
        r = run_cli(["init"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # create hypothesis
        r = run_cli(["hypothesis", "create", "-m", "For exp delete\nBody"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # add experiment
        exp_path = self._create_simple_experiment()
        r = run_cli(["exp", "add", exp_path, "Experiment To Delete"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        m = re.search(r"ID ([0-9a-fA-F]{8})", r.stdout)
        self.assertIsNotNone(m, msg=r.stdout)
        exp_id = m.group(1)

        # delete experiment
        r = run_cli(["exp", "delete", exp_id], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # show should fail after deletion
        r = run_cli(["exp", exp_id], self.cwd)  # shorthand show
        self.assertNotEqual(r.returncode, 0, msg=r.stdout)


if __name__ == "__main__":
    unittest.main()
