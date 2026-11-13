import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SimpleWorkflowStaticTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_active_files_compile(self):
        for name in ["config.py", "data.py", "model.py", "train.py", "run_experiment.py", "src/bloodmnist_baselines/data.py"]:
            ast.parse(self.read(name), filename=name)

    def test_three_modes_and_locked_test_policy(self):
        config = self.read("config.py")
        runner = self.read("run_experiment.py")
        provider = self.read("src/bloodmnist_baselines/data.py")
        self.assertIn("majority", config)
        self.assertIn("linear_probe", config)
        self.assertIn("finetune", config)
        self.assertIn('"test_evaluated": False', runner)
        self.assertIn('"test_policy": "locked_not_loaded"', provider)
        self.assertNotIn('split="test"', provider)
        self.assertNotIn("include_test", provider)

    def test_fixed_subset_cannot_fallback(self):
        source = self.read("data.py")
        self.assertIn("Fixed 1% mode requires", source)
        self.assertIn("Fixed subset manifest not found", self.read("src/bloodmnist_baselines/data.py"))

    def test_neural_selection_is_validation_macro_f1(self):
        source = self.read("run_experiment.py")
        self.assertIn('if validation_result["macro_f1"] > best_f1:', source)
        self.assertIn('"best_model.pt"', source)
        self.assertIn('"metrics.json"', source)
        self.assertIn('"learning_curves.png"', source)

    def test_notebook_is_thin(self):
        notebook = json.loads((ROOT / "notebooks/colab_run.ipynb").read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("assert len(COMMIT) == 40", text)
        self.assertNotIn("ed0f97568df549986ddf2aecf11dee3548135ca4", text)
        self.assertIn("python run_experiment.py", text)
        self.assertNotIn("DataLoader(", text)
        self.assertNotIn("resnet18(", text)


if __name__ == "__main__":
    unittest.main()
