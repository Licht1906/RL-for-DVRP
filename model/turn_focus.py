"""
Turn Focus: xác định context cho xe đang hành động.

Tham chiếu: Section 5.2.3, Figure 5.8

Input:
  h_vehicles: (B, m, D) — all vehicle embeddings
  current_vehicle_idx: (B,) long — index xe đang hành động
Output:
  rho: (B, D) — internal state representation cho xe hiện tại

MHA: query = h^{i_k} (xe hiện tại), key/value = h^i @i (tất cả xe)
"""
import torch
import torch.nn as nn
from .customer_encoder import MultiHeadAttention


class TurnFocus(nn.Module):
    """Turn Focus block.
    
    Tham chiếu: Section 5.2.3, Figure 5.8
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.mha     = MultiHeadAttention(d_model, n_heads)
        self.norm    = nn.LayerNorm(d_model)

    def forward(
        self,
        h_vehicles:          torch.Tensor,  # (B, m, D)
        current_vehicle_idx: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        """
        Returns:
            rho: (B, D) — internal state cho xe đang hành động
        
        Cách hoạt động:
          1. Lấy embedding của xe hiện tại: h^{i_k} shape (B, 1, D)
          2. Dùng nó làm query, attend đến toàn bộ fleet
          3. Output = weighted combination của fleet, conditioned on current vehicle
        """
        B, m, D = h_vehicles.shape

        # Lấy embedding của xe đang hành động: (B, 1, D)
        # current_vehicle_idx: (B,) -> (B, 1, 1) -> expand -> (B, 1, D)
        idx = current_vehicle_idx.view(B, 1, 1).expand(B, 1, D)
        h_current = h_vehicles.gather(dim=1, index=idx)   # (B, 1, D)

        # MHA: current vehicle attends to all vehicles
        # query = h_current: (B, 1, D)
        # key = value = h_vehicles: (B, m, D)
        rho = self.mha(
            query=h_current,    # (B, 1, D)
            key=h_vehicles,     # (B, m, D)
            value=h_vehicles,   # (B, m, D)
        )  # (B, 1, D)

        # Skip connection
        rho = h_current + rho    # (B, 1, D)
        rho = self.norm(rho)     # (B, 1, D)
        
        return rho.squeeze(1)    # (B, D)