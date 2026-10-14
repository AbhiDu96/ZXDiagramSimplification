"""The GNN policy (ActionModel) must do a forward pass over real environment observations."""
import torch


def test_action_model_forward_and_value(small_cfg):
    import ppo
    from models import ActionModel

    env = ppo.make_env(small_cfg, "cpu")
    gm, info = env.reset()
    n_actions = gm.action_mask.shape[1]

    model = ActionModel(
        small_cfg.model.action_dim,
        n_actions,
        hidden_dim=small_cfg.model.hidden_dim,
        n_heads=small_cfg.model.n_heads,
        n_message_passing=small_cfg.model.n_message_passing,
        device="cpu",
        model_type=small_cfg.model.model_type,
    )

    actions, logprob, entropy, values = model.get_action_and_value([gm])
    assert actions.shape == (1, 2)          # (position, action) per observation
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert values.shape == (1,)
    assert torch.isfinite(values).all()
    assert torch.isfinite(logprob).all()

    # the sampled (position, action) must be a legal cell in the mask
    pos, act = actions[0].long().tolist()
    assert gm.action_mask[pos, act] == 1

    value_only = model.get_value([gm])
    assert value_only.shape == (1,)
    assert torch.isfinite(value_only).all()
