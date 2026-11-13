import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SimpleWorkflowStaticTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_files_compile(self):
        for name in ["config.py", "data.py", "model.py", "train.py", "run_experiment.py"]:
            ast.parse(self.read(name), filename=name)

    def test_data_uses_approved_provider_and_checks_lock(self):
        source = self.read("data.py")
        self.assertIn("from bloodmnist_baselines import build_provider_datasets", source)
        self.assertIn('metadata.get("test_policy") != "locked_not_loaded"', source)
        self.assertNotIn("include_test=True", source)
        self.assertNotIn('split="test"', source)

    def test_validation_selection_and_outputs_are_explicit(self):
        source = self.read("run_experiment.py")
        self.assertIn('if validation_metrics["macro_f1"] > best_macro_f1:', source)
        self.assertIn('"best_model.pt"', source)
        self.assertIn('"history.csv"', source)
        self.assertIn('"test_evaluated": False', source)

    def test_configurable_image_size_reaches_transforms(self):
        source = self.read("data.py")
        self.assertIn("def make_transforms(image_size):", source)
        self.assertIn('make_transforms(config["image_size"])', source)
        self.assertIn("Resize((image_size, image_size))", source)


if __name__ == "__main__":
    unittest.main()
