"""End-to-end: drive ZXEnv through the parent's make_env / wrapper / vector-env stack."""
import numpy as np


def _pick_legal(action_mask):
    """action_mask: (positions, n_actions); return a random legal (action, position) or None."""
    legal = (action_mask == 1).nonzero()
    if len(legal) == 0:
        return None
    pos, act = legal[np.random.randint(len(legal))].tolist()
    return int(act), int(pos)


def test_make_env_reset_and_step(small_cfg):
    import ppo

    env = ppo.make_env(small_cfg, "cpu")
    gm, info = env.reset()

    # GraphMask: expanded graph node count matches the action-mask position count
    assert gm.Graph.x.shape[0] == gm.action_mask.shape[0]
    assert gm.action_mask.shape[1] > 0
    assert np.isfinite(info["reward"])

    n_steps = 0
    for _ in range(3):
        choice = _pick_legal(gm.action_mask)
        if choice is None:
            gm, info = env.reset()
            continue
        act, pos = choice
        gm, reward, term, trunc, info = env.step(action=act, position=pos)
        n_steps += 1
        assert np.isfinite(reward)
        if term or trunc:
            gm, info = env.reset()
    assert n_steps >= 1


def test_single_vec_env_step(small_cfg):
    import ppo
    import utils

    n = 2
    envs = utils.SingleVecEnv([ppo.make_env(small_cfg, "cpu") for _ in range(n)], small_cfg)
    next_obs, _ = envs.reset()
    assert len(next_obs) == n

    actions = []
    for gm in next_obs:
        choice = _pick_legal(gm.action_mask)
        assert choice is not None, "expected at least one legal action right after reset"
        act, pos = choice
        actions.append(np.array([pos, act]))  # SingleVecEnv unpacks (pos, action)
    actions = np.array(actions)

    obs, reward, term, trunc, info = envs.step(actions)
    assert len(reward) == n
    assert all(np.isfinite(r) for r in reward)
