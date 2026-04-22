import numpy as np
import torch
from typing import Callable


def evaluate_policy(
    policy_fn: Callable,         # fn(state_dict, mask) -> action (int)
    instances: list,
    p_slow: float = 0.1,
    n_samples: int = 1,          # 1 = greedy, 100 = sampling mode
    device: str = 'cpu',
) -> dict:
    """Evaluate policy trên danh sách instances.
    
    Args:
        policy_fn:  hàm nhận (state_dict, mask) -> action
                    Với sampling mode: gọi n_samples lần, lấy best
        instances:  list of Instance
        p_slow:     stochastic travel time parameter
        n_samples:  số trajectories để sample (greedy = 1)
    
    Returns:
        dict với 'mean_cost', 'std_cost', 'mean_qos', 'costs' (list)
    """
    from ..env.environment import DSCVRPTWEnv

    all_costs = []
    all_qos   = []

    for inst in instances:
        if n_samples == 1:
            # Greedy mode
            cost, qos = _run_episode(policy_fn, inst, p_slow)
        else:
            # Sampling mode: chạy n_samples lần, lấy best
            best_cost = float('inf')
            best_qos  = 0.0
            for _ in range(n_samples):
                cost, qos = _run_episode(policy_fn, inst, p_slow)
                if cost < best_cost:
                    best_cost = cost
                    best_qos  = qos
            cost, qos = best_cost, best_qos

        all_costs.append(cost)
        all_qos.append(qos)

    costs = np.array(all_costs)
    return {
        'mean_cost': float(costs.mean()),
        'std_cost':  float(costs.std()),
        'mean_qos':  float(np.mean(all_qos)),
        'costs':     all_costs,
    }


def _run_episode(policy_fn, instance, p_slow):
    from ..env.environment import DSCVRPTWEnv
    env = DSCVRPTWEnv(instance, p_slow=p_slow)
    state_dict, mask = env.reset()
    total_reward = 0.0
    while True:
        action = policy_fn(state_dict, mask)
        state_dict, mask, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    qos = env.get_qos()
    return -total_reward, qos   # return cost (positive)


def print_comparison_table(results: dict):
    """In bảng so sánh như Table 6.2 trong luận án."""
    print(f"\n{'Method':<20} {'Mean Cost':>12} {'Std Cost':>10} {'QoS':>8}")
    print("-" * 55)
    for method, res in results.items():
        print(
            f"{method:<20} "
            f"{res['mean_cost']:>12.2f} "
            f"{res['std_cost']:>10.2f} "
            f"{res.get('mean_qos', 1.0)*100:>7.1f}%"
        )