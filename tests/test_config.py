"""Hydra config composition — validates the conf/ YAMLs used by the entry points."""
import pytest


def test_compose_ppo_config(make_cfg):
    cfg = make_cfg()
    assert cfg.algorithm.num_steps > 0
    assert cfg.algorithm.total_timesteps > 0
    assert cfg.model.action_dim > 0
    assert cfg.model.model_type in ("GAT", "ActionAtt")
    assert cfg.env.reward_fn
    assert cfg.env.num_envs > 0


@pytest.mark.parametrize("env", ["simple", "more_complex_more_rules_ranges"])
def test_compose_env_variants(make_cfg, env):
    cfg = make_cfg(env=env)
    assert cfg.env is not None
    assert "reward_fn" in cfg.env


def test_overrides_apply(make_cfg):
    cfg = make_cfg(overrides=["env.n_qubits=[4,6]", "algorithm.num_steps=7"])
    assert list(cfg.env.n_qubits) == [4, 6]
    assert cfg.algorithm.num_steps == 7
