import importlib.util
import sys
import os

def build_model(args):
    models_dir = os.path.dirname(os.path.abspath(__file__))
    prvg_path = os.path.join(models_dir, "PR-VG.py")
    spec = importlib.util.spec_from_file_location("models.PR_VG", prvg_path)
    prvg_module = importlib.util.module_from_spec(spec)
    prvg_module.__package__ = "models"
    prvg_module.__name__ = "models.PR_VG"
    sys.modules["models.PR_VG"] = prvg_module
    spec.loader.exec_module(prvg_module)
    return prvg_module.build(args)
