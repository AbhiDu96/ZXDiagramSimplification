"""Shared pytest fixtures for the ZXDiagramSimplification test suite.

Adds the project root to sys.path (so the flat scripts `ppo`, `utils`, `models`, ...
are importable) and provides a Hydra config factory.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONF_DIR = os.path.join(ROOT, "conf")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def make_cfg():
    """Return a factory that composes a Hydra config like the training entry points do."""
    from hydra import compose, initialize_config_dir

    def _make(overrides=None, env="more_complex_more_rules_ranges", model="GATActionModel"):
        base = [
            # config.yaml already declares these groups in its defaults list, so
            # override them (no leading "+", which would *append* a second value
            # and raise "Multiple values for <group>").
            "algorithm=PPO",
            f"model={model}",
            f"env={env}",
            "exp_name=pytest",
            "device=cpu",
        ]
        if overrides:
            base += overrides
        # context manager clears the Hydra singleton on exit, so repeated calls are safe
        with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
            return compose(config_name="config", overrides=base)

    return _make


@pytest.fixture
def small_cfg(make_cfg):
    """A config with small circuits so env resets/steps are fast in tests."""
    return make_cfg(overrides=["env.n_qubits=[4,6]", "env.depth=[20,30]"])
