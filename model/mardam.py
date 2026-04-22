"""
MARDAM: Multi-Agent Routing Deep Attention Mechanisms.

Ghép 4 blocks thành policy network hoàn chỉnh.
Tham chiếu: Section 5.2, Figure 5.5
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from .customer_encoder import CustomerEncoder
from .vehicle_encoder import VehicleEncoder
from .turn_focus import TurnFocus
from .travels_scorer import TravelsScorer


class MARDAM(nn.Module):
    """Full MARDAM policy network.
    
    Forward pass:
      shared_state   -> CustomerEncoder  -> h_customers (B, n+1, D)
      vehicle_states -> VehicleEncoder   -> h_vehicles  (B, m, D)
      h_vehicles, turn -> TurnFocus     -> rho          (B, D)
      rho, h_customers, mask -> TravelsScorer -> probs  (B, n+1)
    
    Caching: h_customers chỉ cần recompute khi có dynamic customer mới xuất hiện.
    """

    def __init__(
        self,
        d_customer: int = 7,    # D_C (số features customer, không tính pending flag)
        d_vehicle:  int = 4,    # D_V
        d_model:    int = 128,  # D
        n_heads:    int = 8,    # N_H
        d_ff:       int = 512,  # D_F
        n_enc_layers: int = 3,  # N_L
        c_tanh:     float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model

        self.customer_encoder = CustomerEncoder(
            d_input=d_customer,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_enc_layers,
        )
        self.vehicle_encoder = VehicleEncoder(
            d_vehicle=d_vehicle,
            d_model=d_model,
            n_heads=n_heads,
        )
        self.turn_focus = TurnFocus(
            d_model=d_model,
            n_heads=n_heads,
        )
        self.travels_scorer = TravelsScorer(
            d_model=d_model,
            n_heads=n_heads,
            c_tanh=c_tanh,
        )

        # Cache cho customer embeddings (optimize cho dynamic customers)
        self._cached_h_customers: Optional[torch.Tensor] = None
        self._cached_pending_signature: Optional[str] = None

    def forward(
        self,
        shared_states:  torch.Tensor,   # (B, n+1, 8)
        vehicle_states: torch.Tensor,   # (B, m, 4)
        turns:          torch.Tensor,   # (B,) long — index xe hiện tại
        action_masks:   torch.Tensor,   # (B, n+1) float
        recompute_customers: bool = True,  # False nếu không có dynamic customer mới
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_probs: (B, n+1) — policy distribution
            log_probs:    (B, n+1) — log probabilities (cho policy gradient)
        """
        # Block 1: Customer Encoder
        if recompute_customers or self._cached_h_customers is None:
            h_customers = self.customer_encoder(shared_states)  # (B, n+1, D)
            self._cached_h_customers = h_customers
        else:
            h_customers = self._cached_h_customers

        # Block 2: Vehicle Encoder
        h_vehicles = self.vehicle_encoder(vehicle_states, h_customers)  # (B, m, D)

        # Block 3: Turn Focus
        rho = self.turn_focus(h_vehicles, turns)   # (B, D)

        # Block 4: Travels Scorer
        action_probs = self.travels_scorer(rho, h_customers, action_masks)  # (B, n+1)

        # Log probabilities cho policy gradient (cẩn thận numerical stability)
        log_probs = torch.log(action_probs + 1e-8)

        return action_probs, log_probs

    def select_action(
        self,
        state_dict: dict,
        mask: np.ndarray,
        device: str = 'cpu',
        greedy: bool = True,
    ) -> int:
        """Chọn 1 action (dùng lúc inference/evaluation).
        
        Args:
            state_dict: từ env.reset() hoặc env.step()
            mask:       action mask từ env
            greedy:     True = argmax, False = sample
        
        Returns:
            action: int
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors với batch dim
            shared   = torch.tensor(state_dict['shared'],   dtype=torch.float32, device=device).unsqueeze(0)
            vehicles = torch.tensor(state_dict['vehicles'], dtype=torch.float32, device=device).unsqueeze(0)
            turn     = torch.tensor([state_dict['turn']],   dtype=torch.long, device=device)
            mask_t   = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

            probs, _ = self.forward(shared, vehicles, turn, mask_t)
            probs    = probs.squeeze(0)   # (n+1,)

            if greedy:
                action = int(probs.argmax().item())
            else:
                action = int(torch.multinomial(probs, 1).item())
        return action

    def get_state_encoding(
        self,
        shared_states:  torch.Tensor,
        vehicle_states: torch.Tensor,
    ) -> torch.Tensor:
        """Lấy global state encoding cho Critic (mean pooling).
        
        Returns:
            encoding: (B, 2*D) — concat(mean(h_customers), mean(h_vehicles))
        """
        h_cust = self.customer_encoder(shared_states)       # (B, n+1, D)
        h_veh  = self.vehicle_encoder(vehicle_states, h_cust)  # (B, m, D)
        # Mean pooling
        mean_cust = h_cust.mean(dim=1)   # (B, D)
        mean_veh  = h_veh.mean(dim=1)    # (B, D)
        return torch.cat([mean_cust, mean_veh], dim=-1)     # (B, 2*D)