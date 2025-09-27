import os
import sys
import unittest
import tempfile
import textwrap
import ast

from mlx_core import MLXCore


class TestParserAndExecution(unittest.TestCase):
    def setUp(self):
        # Isolate MLXCore state in a temp .mlx directory
        self.tmp_mlx_dir = tempfile.TemporaryDirectory()
        self.core = MLXCore(self.tmp_mlx_dir.name)

    def tearDown(self):
        self.tmp_mlx_dir.cleanup()

    def test_extract_file_components_parses_decorators(self):
        src = textwrap.dedent(
            """
            import os
            from decorators import trial, verify, VerificationResult

            CONST = 42

            def helper(z):
                return z + 1

            @trial(1, {"x": [1, 2], "y": [3]})
            def t1(x, y):
                return x + y

            @trial(order=2, values={"a": [10]})
            def t2(a):
                return a * 2

            @verify(["t1"])  # positional
            def v1(results):
                return "supports"

            @verify(trial_names=["t2"])  # keyword
            def v2(results):
                return "supports"
            """
        )

        trials, verifs, imports, other = self.core.extract_file_components(src)

        # Imports and other code captured
        self.assertGreaterEqual(len(imports), 1)
        self.assertGreaterEqual(len(other), 1)

        # Trials parsed and sorted by order
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0][0], 1)
        self.assertEqual(trials[0][1], "t1")
        self.assertEqual(trials[1][0], 2)
        self.assertEqual(trials[1][1], "t2")

        # Ensure values mapping is preserved (light check)
        order1, name1, values1, body1 = trials[0]
        self.assertIn("x", values1)
        self.assertIn("y", values1)
        self.assertIn("return x + y", body1)

        order2, name2, values2, body2 = trials[1]
        self.assertIn("a", values2)
        self.assertIn("return a * 2", body2)

        # Verifications parsed with trial name lists
        self.assertEqual(len(verifs), 2)
        v1_name, v1_trials, _ = verifs[0]
        v2_name, v2_trials, _ = verifs[1]
        self.assertEqual(v1_name, "v1")
        self.assertEqual(v1_trials, ["t1"])
        self.assertEqual(v2_name, "v2")
        self.assertEqual(v2_trials, ["t2"])

    def test_create_trial_temp_file_and_execute_success(self):
        func_name = "add"
        func_body = "return x + y"
        arg_names = ["x", "y"]
        arg_values = [1, 2]
        import_lines = []
        other_code = []

        with tempfile.TemporaryDirectory() as base_dir:
            with tempfile.TemporaryDirectory() as run_dir:
                temp_path = self.core.create_trial_temp_file(
                    func_name,
                    func_body,
                    arg_names,
                    arg_values,
                    import_lines,
                    other_code,
                    base_dir,
                    temp_dir=run_dir,
                )
                out = self.core.execute_temp_file(temp_path, sys.executable, base_dir, True)
                self.assertTrue(out.get("success"))
                self.assertEqual(out.get("result"), 3)

    def test_create_trial_temp_file_and_execute_error(self):
        func_name = "boom"
        func_body = "raise ValueError(\"boom\")"
        arg_names = []
        arg_values = []
        import_lines = []
        other_code = []

        with tempfile.TemporaryDirectory() as base_dir:
            with tempfile.TemporaryDirectory() as run_dir:
                temp_path = self.core.create_trial_temp_file(
                    func_name,
                    func_body,
                    arg_names,
                    arg_values,
                    import_lines,
                    other_code,
                    base_dir,
                    temp_dir=run_dir,
                )
                out = self.core.execute_temp_file(temp_path, sys.executable, base_dir, True)
                self.assertFalse(out.get("success"))
                # Should include error info produced by the wrapper
                self.assertIn("error", out)
                self.assertEqual(out.get("type"), "ValueError")

    def test_create_verify_temp_file_and_execute_enum_supports(self):
        # Build a simple verify function AST: returns VerificationResult.SUPPORTS
        verify_src = "def check(results):\n    return VerificationResult.SUPPORTS\n"
        verify_node = ast.parse(verify_src).body[0]

        with tempfile.TemporaryDirectory() as base_dir:
            with tempfile.TemporaryDirectory() as run_dir:
                temp_path = self.core.create_verify_temp_file(
                    "check",
                    verify_node,
                    {"dummy": []},
                    import_lines=[],
                    other_code=[],
                    base_dir=base_dir,
                    temp_dir=run_dir,
                )
                out = self.core.execute_temp_file(temp_path, sys.executable, base_dir, True)
                self.assertTrue(out.get("success"))
                # Wrapper maps Enum to its value string
                self.assertEqual(out.get("result"), "supports")


if __name__ == "__main__":
    unittest.main()
