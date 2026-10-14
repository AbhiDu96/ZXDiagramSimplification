"""The core training import chain must import cleanly (catches the ZXEnv rename etc.)."""
import importlib

import pytest

CORE_MODULES = ["zx_env", "utils", "multiEnv", "TreePolicy", "models", "ppo"]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    importlib.import_module(name)


def test_zxenv_and_models_symbols():
    from zx_env import ZXEnv, extract_circuit, random_circuit
    from models import ActionModel, BundleNet, TreeNet

    assert ZXEnv.__name__ == "ZXEnv"
    assert all(callable(x) for x in (extract_circuit, random_circuit))
    assert all(x is not None for x in (ActionModel, BundleNet, TreeNet))
