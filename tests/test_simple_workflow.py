import ast
import json
from pathlib import Path
import unittest
import sys
import types
from unittest.mock import patch

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

    def test_subset_path_comes_from_audit_manifest(self):
        config = self.read("config.py")
        source = self.read("data.py")
        self.assertNotIn(chr(34) + "subset_manifest_path" + chr(34) + ":", config)
        self.assertIn("audit.get(\"subset_manifest\")", source)
        self.assertIn("override differs from the audited subset manifest", source)

    def test_provider_creates_data_root_before_dataset_setup(self):
        source = self.read("src/bloodmnist_baselines/data.py")
        mkdir = source.index("root.mkdir(parents=True, exist_ok=True)")
        train = source.index("BloodMNIST(split=\"train\"")
        self.assertLess(mkdir, train)

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


class ProviderAuditValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((ROOT / "artifacts/data_audit/audit_manifest.json").read_text(encoding="utf-8"))
        labels = {str(key): value for key, value in manifest["class_mapping"].items()}
        labels["3"] = labels["3"].replace("granulocytes (", "granulocytes(")
        medmnist = types.ModuleType("medmnist")
        medmnist.BloodMNIST = type("BloodMNIST", (), {})
        medmnist.INFO = {"bloodmnist": {"label": labels}}
        torch = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_data = types.ModuleType("torch.utils.data")
        torch_data.Dataset = type("Dataset", (), {})
        torch_data.Subset = type("Subset", (), {})
        modules = {"medmnist": medmnist, "torch": torch, "torch.utils": torch_utils, "torch.utils.data": torch_data}
        with patch.dict(sys.modules, modules):
            from bloodmnist_baselines import data as provider_data
        cls.provider_data = provider_data
        cls.upstream = dict(provider_data.CLASS_NAMES)

    def validate(self, mapping):
        with patch.object(self.provider_data, "CLASS_NAMES", self.upstream):
            self.provider_data._validate_class_mapping(mapping)

    def test_space_before_parenthesis_is_cosmetic(self):
        mapping = {str(key): value for key, value in self.upstream.items()}
        mapping["3"] = mapping["3"].replace("granulocytes(", "granulocytes (")
        self.validate(mapping)
        self.validate({str(key): value for key, value in self.upstream.items()})

    def test_meaningful_label_change_fails(self):
        mapping = {str(key): value for key, value in self.upstream.items()}
        mapping["3"] = mapping["3"].replace("immature", "mature")
        with self.assertRaisesRegex(ValueError, "class mapping differs"):
            self.validate(mapping)

    def test_swapped_assignments_fail(self):
        mapping = {str(key): value for key, value in self.upstream.items()}
        mapping["6"], mapping["7"] = mapping["7"], mapping["6"]
        with self.assertRaisesRegex(ValueError, "class mapping differs"):
            self.validate(mapping)

    def test_missing_or_extra_id_fails(self):
        mapping = {str(key): value for key, value in self.upstream.items()}
        mapping.pop("7")
        with self.assertRaisesRegex(ValueError, "class IDs differ"):
            self.validate(mapping)

    def test_non_string_label_fails(self):
        mapping = {str(key): value for key, value in self.upstream.items()}
        mapping["3"] = None
        with self.assertRaisesRegex(ValueError, "class mapping is invalid"):
            self.validate(mapping)


if __name__ == "__main__":
    unittest.main()
