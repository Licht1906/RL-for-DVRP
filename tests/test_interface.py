# tests/test_interface.py
"""
Test interface contract giữa Thành viên A (Environment) và Thành viên B (Model).
Chạy: python -m pytest tests/test_interface.py -v
"""
import numpy as np
import torch
import pytest

# Import từ cả 2 module
from mardam.env.instance import generate_instance
from mardam.env.environment import DSCVRPTWEnv
from mardam.model.mardam import MARDAM


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
N = 10    # số customers
M = 3     # số vehicles
D = 128   # model dimension


@pytest.fixture
def env_and_instance():
    inst = generate_instance(N, M, p_dyn=0.3, rng=np.random.default_rng(42))
    env  = DSCVRPTWEnv(inst, p_slow=0.1)
    env.seed(42)
    return env, inst


@pytest.fixture
def model():
    return MARDAM(d_customer=7, d_vehicle=4, d_model=D)


# ─────────────────────────────────────────────────────────────
#  Test 1: state_dict shape contract
# ─────────────────────────────────────────────────────────────
def test_state_dict_shapes(env_and_instance):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    assert 'shared'   in state_dict
    assert 'vehicles' in state_dict
    assert 'turn'     in state_dict

    assert state_dict['shared'].shape   == (N + 1, 8), \
        f"Expected shared shape ({N+1}, 8), got {state_dict['shared'].shape}"
    assert state_dict['vehicles'].shape == (M, 4), \
        f"Expected vehicles shape ({M}, 4), got {state_dict['vehicles'].shape}"
    assert isinstance(state_dict['turn'], int), \
        f"turn should be int, got {type(state_dict['turn'])}"
    assert 0 <= state_dict['turn'] < M


# ─────────────────────────────────────────────────────────────
#  Test 2: mask shape và validity
# ─────────────────────────────────────────────────────────────
def test_mask_shape_and_validity(env_and_instance):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    assert mask.shape == (N + 1,), \
        f"Expected mask shape ({N+1},), got {mask.shape}"
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0}), "Mask should be binary"
    assert mask[0] == 1.0, "Depot (action 0) should always be valid"
    assert mask.sum() >= 1.0, "At least 1 valid action must exist"


# ─────────────────────────────────────────────────────────────
#  Test 3: MARDAM forward pass shapes
# ─────────────────────────────────────────────────────────────
def test_mardam_forward_shapes(env_and_instance, model):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    B = 4  # batch size
    shared   = torch.tensor(state_dict['shared'],   dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    vehicles = torch.tensor(state_dict['vehicles'], dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    turns    = torch.zeros(B, dtype=torch.long)
    mask_t   = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(B, 1)

    probs, log_probs = model(shared, vehicles, turns, mask_t)

    assert probs.shape     == (B, N + 1), f"probs shape mismatch: {probs.shape}"
    assert log_probs.shape == (B, N + 1)


# ─────────────────────────────────────────────────────────────
#  Test 4: Masking — invalid actions có prob = 0
# ─────────────────────────────────────────────────────────────
def test_masking_invalidates_actions(env_and_instance, model):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    shared   = torch.tensor(state_dict['shared'],   dtype=torch.float32).unsqueeze(0)
    vehicles = torch.tensor(state_dict['vehicles'], dtype=torch.float32).unsqueeze(0)
    turns    = torch.tensor([state_dict['turn']], dtype=torch.long)
    mask_t   = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    probs, _ = model(shared, vehicles, turns, mask_t)
    probs_np = probs.squeeze(0).detach().numpy()

    for j in range(N + 1):
        if mask[j] == 0.0:
            assert probs_np[j] < 1e-6, \
                f"Invalid action {j} has prob {probs_np[j]:.6f} > 0"


# ─────────────────────────────────────────────────────────────
#  Test 5: Probability distribution sums to 1
# ─────────────────────────────────────────────────────────────
def test_prob_distribution_sums_to_one(env_and_instance, model):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    shared   = torch.tensor(state_dict['shared'],   dtype=torch.float32).unsqueeze(0)
    vehicles = torch.tensor(state_dict['vehicles'], dtype=torch.float32).unsqueeze(0)
    turns    = torch.tensor([state_dict['turn']], dtype=torch.long)
    mask_t   = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    probs, _ = model(shared, vehicles, turns, mask_t)
    prob_sum = probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, f"Probs sum to {prob_sum}, expected 1.0"


# ─────────────────────────────────────────────────────────────
#  Test 6: Full episode với random policy
# ─────────────────────────────────────────────────────────────
def test_full_episode_with_random_policy(env_and_instance):
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    max_steps = (N + M) * 2  # safety limit
    total_reward = 0.0
    steps = 0

    while steps < max_steps:
        # Random valid action
        valid_actions = np.where(mask == 1.0)[0]
        action = int(np.random.choice(valid_actions))

        state_dict, mask, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            break

    assert done, f"Episode should finish within {max_steps} steps"
    print(f"\nRandom policy: {steps} steps, total reward: {total_reward:.2f}")


# ─────────────────────────────────────────────────────────────
#  Test 7: Vehicle capacity constraint
# ─────────────────────────────────────────────────────────────
def test_capacity_constraint(env_and_instance):
    """Không có xe nào vượt quá capacity."""
    from mardam.env.state import VEH_CAP, MAX_CAPACITY
    env, inst = env_and_instance
    state_dict, mask = env.reset()

    while True:
        valid_actions = np.where(mask == 1.0)[0]
        action = int(np.random.choice(valid_actions))
        state_dict, mask, reward, done, info = env.step(action)

        # Capacity không âm
        caps = state_dict['vehicles'][:, VEH_CAP]
        assert (caps >= -1e-5).all(), f"Negative capacity detected: {caps}"
        assert (caps <= MAX_CAPACITY + 1e-5).all(), f"Capacity exceeded: {caps}"

        if done:
            break