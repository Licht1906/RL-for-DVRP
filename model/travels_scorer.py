"""
Travels Scorer: tính policy distribution π(a|s).

Tham chiếu: Section 5.2.4, Figure 5.9

Input:
  rho:         (B, D) — internal state của xe hiện tại
  h_customers: (B, n+1, D) — customer embeddings
  action_mask: (B, n+1) float — 1 = valid, 0 = invalid
Output:
  action_probs: (B, n+1) — probability distribution

Steps:
  1. Score = (rho * W_Q) dot (h^{0,j} * W_K) / sqrt(D_H)
  2. Tanh saturation: C_tanh * tanh(score / C_tanh)
  3. Masking: set score = -inf cho invalid actions
  4. Softmax -> probabilities
"""
import torch
import torch.nn as nn
import math


class TravelsScorer(nn.Module):
    """Travels Scorer với masking và tanh saturation.
    
    Tham chiếu: Section 5.2.4, Figure 5.9
    """

    def __init__(
        self,
        d_model: int = 128,  # D
        n_heads: int = 8,    # N_H (dùng 1 head trong scorer)
        c_tanh:  float = 10.0,  # C_tanh — tanh saturation
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head  = d_model // n_heads   # D_H = 16
        self.c_tanh  = c_tanh

        # Projections cho query (rho) và key (customer embeddings)
        self.W_q = nn.Linear(d_model, self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, self.d_head, bias=False)

    def forward(
        self,
        rho:         torch.Tensor,  # (B, D)
        h_customers: torch.Tensor,  # (B, n+1, D)
        action_mask: torch.Tensor,  # (B, n+1) float: 1=valid, 0=invalid
    ) -> torch.Tensor:
        """
        Returns:
            action_probs: (B, n+1) — valid probability distribution
                          (invalid actions have prob = 0)
        
        CRITICAL: masking phải set -inf TRƯỚC softmax để invalid prob = 0
        """
        B, n_plus_1, D = h_customers.shape
        
        # Project
        q = self.W_q(rho)           # (B, D_H)
        k = self.W_k(h_customers)   # (B, n+1, D_H)

        # Score: dot product
        # q: (B, D_H) -> (B, 1, D_H)
        # k: (B, n+1, D_H)
        scale = math.sqrt(self.d_head)
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) / scale  # (B, 1, n+1)
        scores = scores.squeeze(1)   # (B, n+1)

        # Tanh saturation (Section 5.2.4)
        # Mục đích: giới hạn range của scores để ổn định training
        scores = self.c_tanh * torch.tanh(scores / self.c_tanh)

        # ⚠️ MASKING: phải dùng -inf (không phải 0!) để softmax cho prob = 0
        # action_mask: 1 = valid, 0 = invalid
        # -> invalid mask = (action_mask == 0)
        invalid_mask = (action_mask < 0.5)
        scores = scores.masked_fill(invalid_mask, float('-inf'))

        # Softmax -> probabilities
        action_probs = torch.softmax(scores, dim=-1)   # (B, n+1)

        return action_probs