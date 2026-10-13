"""Check extraction distinguishes definitions from executed wrapper launches."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/rmsnorm_compiled_inspect.py"
spec = importlib.util.spec_from_file_location("rmsnorm_compiled_inspect", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeneratedCodeTest(unittest.TestCase):
    def test_extract_definitions_and_only_main_wrapper_launches(self):
        source = '''
kernel = async_compile.triton("norm", "def norm(): pass", device_str="cuda")
def call(args):
    kernel.run(args[0], grid=grid(128))
def benchmark():
    kernel.run(None, grid=grid(128))
'''
        result = module.extract_kernels(source)
        self.assertEqual(result["kernel_definitions"], [{"name": "norm", "source": "def norm(): pass"}])
        self.assertEqual(result["wrapper_launches"], ["kernel.run(args[0], grid=grid(128))"])

    def test_selected_launcher_reports_blocks_and_resources(self):
        from types import SimpleNamespace
        config = SimpleNamespace(kwargs={"XBLOCK": 4, "RBLOCK": 256}, num_warps=4, num_stages=1)
        launcher = SimpleNamespace(config=config, n_regs=40, n_spills=0, shared=32)
        records = module.selected_launchers({"norm": SimpleNamespace(launchers=[launcher])}, 128)
        self.assertEqual(records[0]["programs_from_xblock"], 32)
        self.assertEqual(records[0]["block_parameters"]["RBLOCK"], 256)
        self.assertEqual(records[0]["status"], "selected_after_execution")

    def test_multiple_candidates_are_not_reported_as_selected(self):
        from types import SimpleNamespace
        records = module.selected_launchers({"norm": SimpleNamespace(launchers=[1, 2])}, 128)
        self.assertEqual(records[0]["status"], "no_unique_selected_launcher")
        self.assertNotIn("block_parameters", records[0])

    def test_empty_source_does_not_invent_kernel_details(self):
        self.assertEqual(module.extract_kernels("x = 1"),
                         {"kernel_definitions": [], "wrapper_launches": []})


if __name__ == "__main__":
    unittest.main()
