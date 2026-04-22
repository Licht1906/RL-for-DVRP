"""
Vehicle Encoder: MHA layer map vehicle states vào customer embedding space.

Tham chiếu: Section 5.2.2, Figure 5.7

Input:
  s̄^i: vehicle states (B, m, D_V=4)
  h^{0,j}: customer embeddings (B, n+1, D)
Output:
  h^i: vehicle embeddings (B, m, D)

MHA: query = vehicle states, key = value = customer embeddings
Weight sharing: cùng query projection cho mọi vehicle -> handle variable fleet
"""
import torch
import torch.nn as nn
from .customer_encoder import MultiHeadAttention


class VehicleEncoder(nn.Module):
    """MHA-based vehicle encoder.
    
    Tham chiếu: Section 5.2.2, Figure 5.7
    """

    def __init__(
        self,
        d_vehicle: int = 4,    # D_V: số features vehicle state
        d_model:   int = 128,  # D
        n_heads:   int = 8,    # N_H
    ):
        super().__init__()
        self.d_model = d_model
        
        # Project vehicle state D_V -> D (làm query)
        self.vehicle_proj = nn.Linear(d_vehicle, d_model)
        
        # MHA: query = vehicles, key/value = customers
        # Dùng lại class MultiHeadAttention nhưng Q ≠ K,V
        self.mha = MultiHeadAttention(d_model, n_heads)
        
        # Output normalization
        self.norm = nn.BatchNorm1d(d_model)

    def forward(
        self,
        vehicle_states:     torch.Tensor,  # (B, m, 4)
        customer_embeddings: torch.Tensor, # (B, n+1, D)
    ) -> torch.Tensor:
        """
        Returns:
            h_vehicles: (B, m, D) — vehicle embeddings
        
        Note: cùng query projection cho tất cả xe (shared weights)
        -> cho phép generalize với số xe khác nhau
        """
        B, m, _ = vehicle_states.shape

        # Project vehicle states -> query (B, m, D)
        q = self.vehicle_proj(vehicle_states)  # (B, m, D)
        
        # Customer embeddings làm key và value
        k = customer_embeddings   # (B, n+1, D)
        v = customer_embeddings   # (B, n+1, D)
        
        # Cross-attention: vehicle attends to customers
        h = self.mha(query=q, key=k, value=v)  # (B, m, D)
        
        # Skip connection + norm
        h = q + h
        # BatchNorm: cần (B, D, m) -> transpose
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)  # (B, m, D)
        
        return h