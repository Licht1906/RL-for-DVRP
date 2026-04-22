# training/actor_critic.py
"""
Actor-Critic training loop theo Algorithm 6.1 trong luận án.

Hyperparameters (Table 6.1):
  B = 512        mini-batch size
  N_iter = 2500  iterations per epoch
  N_epoch = 100  training epochs
  lr_actor = 0.0001
  lr_critic = 0.001
  max_grad_norm = 2.0
  optimizer = Adam
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List
import time
import os

from ..model.mardam import MARDAM
from .critic import Critic
from ..env.environment import DSCVRPTWEnv
from ..env.instance import Instance


def collect_trajectory(
    policy:   MARDAM,
    instance: Instance,
    device:   str = 'cpu',
    greedy:   bool = False,  # False = sampling trong training
) -> dict:
    """Collect 1 trajectory theo current policy.
    
    Returns:
        trajectory: dict chứa tất cả thông tin cần cho policy gradient
    """
    env = DSCVRPTWEnv(instance, p_slow=0.1)
    state_dict, mask = env.reset()

    log_probs_list = []
    rewards_list   = []
    state_encodings_list = []
    masks_list     = []
    # Track xem có dynamic customer mới không (để quyết định recompute)
    prev_pending = state_dict['shared'][:, 7].sum()

    while True:
        # Convert to tensors
        shared   = torch.tensor(state_dict['shared'],   dtype=torch.float32, device=device).unsqueeze(0)
        vehicles = torch.tensor(state_dict['vehicles'], dtype=torch.float32, device=device).unsqueeze(0)
        turn     = torch.tensor([state_dict['turn']],   dtype=torch.long, device=device)
        mask_t   = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        # Kiểm tra có dynamic customer mới không
        curr_pending = state_dict['shared'][:, 7].sum()
        recompute = (curr_pending != prev_pending)
        prev_pending = curr_pending

        # Forward pass
        probs, log_probs = policy.forward(
            shared, vehicles, turn, mask_t,
            recompute_customers=recompute
        )

        # State encoding cho critic
        with torch.no_grad():
            enc = policy.get_state_encoding(shared, vehicles)
        state_encodings_list.append(enc)

        # Sample action
        action_dist = torch.distributions.Categorical(probs=probs.squeeze(0))
        action_t    = action_dist.sample()   # scalar tensor
        action      = int(action_t.item())

        # Log prob của action được chọn
        log_prob = log_probs.squeeze(0)[action]
        log_probs_list.append(log_prob)

        # Execute action
        state_dict, mask, reward, done, info = env.step(action)
        rewards_list.append(reward)

        if done:
            break

    return {
        'log_probs':       torch.stack(log_probs_list),          # (T,)
        'rewards':         torch.tensor(rewards_list, dtype=torch.float32, device=device),  # (T,)
        'state_encodings': torch.cat(state_encodings_list, dim=0),  # (T, 2D)
        'total_reward':    sum(rewards_list),
        'n_steps':         len(rewards_list),
    }


def compute_returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Tính discounted returns R_t = sum_{k=t}^{T} gamma^{k-t} * r_k.
    
    Trong luận án: gamma = 1 (undiscounted, finite horizon)
    """
    T = len(rewards)
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


class ActorCriticTrainer:
    """Trainer theo Algorithm 6.1."""

    def __init__(
        self,
        policy:  MARDAM,
        critic:  Critic,
        device:  str = 'cpu',
        # Hyperparameters (Table 6.1)
        lr_actor:       float = 1e-4,
        lr_critic:      float = 1e-3,
        max_grad_norm:  float = 2.0,
        gamma:          float = 1.0,
        batch_size:     int   = 512,
        n_epochs:       int   = 100,
        n_iters_per_epoch: int = 2500,
    ):
        self.policy  = policy.to(device)
        self.critic  = critic.to(device)
        self.device  = device
        self.gamma   = gamma
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.n_iters    = n_iters_per_epoch
        self.max_grad_norm = max_grad_norm

        self.opt_actor  = torch.optim.Adam(policy.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    def train_epoch(
        self,
        dataset: List[Instance],
        epoch: int,
        log_interval: int = 100,
    ) -> dict:
        """1 epoch = 1 pass qua toàn bộ dataset.
        
        Returns:
            metrics: dict với mean_reward, actor_loss, critic_loss
        """
        self.policy.train()
        self.critic.train()

        all_rewards    = []
        all_actor_loss = []
        all_critic_loss = []

        # Mini-batch loop (Algorithm 6.1, lines 3-11)
        idx = np.random.permutation(len(dataset))
        batch_start = 0

        for iter_i in range(self.n_iters):
            # Lấy mini-batch instances
            batch_idx = idx[batch_start: batch_start + self.batch_size]
            if len(batch_idx) == 0:
                idx = np.random.permutation(len(dataset))
                batch_start = 0
                batch_idx = idx[:self.batch_size]
            batch_start += self.batch_size

            batch_instances = [dataset[i] for i in batch_idx]

            # Collect trajectories cho batch
            # Trong paper: parallel execution; đây ta dùng sequential
            trajectories = [
                collect_trajectory(self.policy, inst, self.device)
                for inst in batch_instances[:32]  # subsample 32 để nhanh hơn
            ]

            # Tính loss trên batch
            actor_loss_batch  = []
            critic_loss_batch = []
            batch_rewards     = []

            for traj in trajectories:
                log_probs      = traj['log_probs']       # (T,)
                rewards        = traj['rewards']         # (T,)
                state_encodings = traj['state_encodings']  # (T, 2D)

                # Returns R_t
                returns = compute_returns(rewards, self.gamma)  # (T,)
                batch_rewards.append(float(returns[0].item()))

                # Value estimates V(s_t) từ critic
                values = self.critic(state_encodings)     # (T,)

                # Advantage = R_t - V(s_t)
                advantages = (returns - values.detach())  # (T,) — detach critic!

                # Actor loss: -E[log pi(a|s) * A(s,a)]
                actor_loss = -(log_probs * advantages).mean()
                actor_loss_batch.append(actor_loss)

                # Critic loss: MSE(V(s_t), R_t)
                critic_loss = nn.functional.mse_loss(values, returns)
                critic_loss_batch.append(critic_loss)

            # Aggregate losses
            total_actor_loss  = torch.stack(actor_loss_batch).mean()
            total_critic_loss = torch.stack(critic_loss_batch).mean()

            # Actor update
            self.opt_actor.zero_grad()
            total_actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt_actor.step()

            # Critic update
            self.opt_critic.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.opt_critic.step()

            all_rewards.append(np.mean(batch_rewards))
            all_actor_loss.append(float(total_actor_loss.item()))
            all_critic_loss.append(float(total_critic_loss.item()))

            if (iter_i + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch} | Iter {iter_i+1}/{self.n_iters} | "
                    f"Reward: {np.mean(all_rewards[-100:]):.3f} | "
                    f"A-Loss: {np.mean(all_actor_loss[-100:]):.4f} | "
                    f"C-Loss: {np.mean(all_critic_loss[-100:]):.4f}"
                )

        return {
            'mean_reward':    float(np.mean(all_rewards)),
            'actor_loss':     float(np.mean(all_actor_loss)),
            'critic_loss':    float(np.mean(all_critic_loss)),
        }

    def train(
        self,
        train_dataset: List[Instance],
        val_dataset:   List[Instance],
        save_dir: str = './checkpoints',
    ):
        """Full training loop."""
        os.makedirs(save_dir, exist_ok=True)
        best_val_reward = float('-inf')

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.n_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_dataset, epoch)

            # Validation
            val_reward = self._evaluate(val_dataset)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch} done | "
                f"Train reward: {train_metrics['mean_reward']:.3f} | "
                f"Val reward: {val_reward:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                torch.save({
                    'epoch': epoch,
                    'policy_state': self.policy.state_dict(),
                    'critic_state': self.critic.state_dict(),
                    'val_reward':   val_reward,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  -> Saved best model (val={val_reward:.3f})")

            # Save checkpoint mỗi 10 epoch
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'policy_state': self.policy.state_dict(),
                    'critic_state': self.critic.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt'))

    def _evaluate(self, val_dataset: List[Instance], n_samples: int = 100) -> float:
        """Quick evaluation trên val set."""
        self.policy.eval()
        rewards = []
        with torch.no_grad():
            for inst in val_dataset[:n_samples]:
                traj = collect_trajectory(self.policy, inst, self.device, greedy=True)
                rewards.append(traj['total_reward'])
        self.policy.train()
        return float(np.mean(rewards))