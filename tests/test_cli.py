import os
import sys
import unittest
import tempfile
import textwrap
import re
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


class TestCLI(unittest.TestCase):
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

    def test_cli_end_to_end(self):
        # init
        r = run_cli(["init"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        self.assertTrue(os.path.isdir(os.path.join(self.cwd, ".mlx")))

        # hypothesis create
        title = "CLI Hypothesis"
        body = "CLI Hypothesis\nThis is a multiline description."
        r = run_cli(["hypothesis", "create", "-m", body], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        m = re.search(r"ID ([0-9a-fA-F]{8})", r.stdout)
        self.assertIsNotNone(m, msg=r.stdout)
        hypo_id = m.group(1)

        # hypothesis show (basic check)
        r = run_cli(["hypothesis", "show", hypo_id], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        self.assertIn("Name:", r.stdout)
        self.assertIn("Created:", r.stdout)

        # set-desc
        new_desc = "Updated description via CLI."
        r = run_cli(["hypothesis", "set-desc", hypo_id, "-m", new_desc], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # conclude
        r = run_cli(["hypothesis", "conclude", hypo_id, "--status", "supports", "-m", "Looks good."], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)

        # add experiment
        exp_src = """
        from decorators import trial, verify, VerificationResult
        @trial(1, {"x": [1, 2]})
        def t(x):
            return x * 2
        @verify(["t"])
        def check(results):
            runs = results.get("t", [])
            if not runs:
                return VerificationResult.INCONCLUSIVE
            ok = all(r.get("success") and ("result" in r) for r in runs)
            return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
        """
        exp_path = self._create_experiment_file(exp_src)
        r = run_cli(["exp", "add", exp_path, "CLI Experiment"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        m = re.search(r"ID ([0-9a-fA-F]{8})", r.stdout)
        self.assertIsNotNone(m, msg=r.stdout)
        exp_id = m.group(1)

        # run experiment
        r = run_cli(["run", exp_id], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        # Don't assert colored status text; success exit code is sufficient here

        # status
        r = run_cli(["status", exp_id], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        self.assertIn("Experiment", r.stdout)
        self.assertIn("status:", r.stdout.lower())

        # log
        r = run_cli(["log"], self.cwd)
        self.assertEqual(r.returncode, 0, msg=r.stdout)
        # Should include some tree header text
        self.assertIn("HYPOTHESIS TREE", r.stdout)


if __name__ == "__main__":
    unittest.main()