# training/critic.py
"""
Critic (baseline) network: ước lượng value function V(s).

Input:  global state encoding từ MARDAM.get_state_encoding()
Output: V(s) scalar

Tham chiếu: Section 3.3 (Actor-Critic), Algorithm 6.1
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    """Value function approximator.
    
    Dùng để giảm variance trong Policy Gradient (Actor-Critic).
    Input: concatenation của mean customer embeddings + mean vehicle embeddings
    """

    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: D (phải match với MARDAM)
        """
        super().__init__()
        # Input: 2*D (từ MARDAM.get_state_encoding)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_encoding: (B, 2*D)
        Returns:
            values: (B,) — estimated V(s) cho mỗi instance trong batch
        """
        return self.net(state_encoding).squeeze(-1)   # (B,)