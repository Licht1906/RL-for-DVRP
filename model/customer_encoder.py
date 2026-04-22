"""
Customer Encoder: Transformer Encoder mã hóa tập khách hàng.

Tham chiếu: Section 5.2.1, Figure 5.6
Input:  s̄^0 — shared state (B, n+1, D_C)
Output: h^{0,j} — customer embeddings (B, n+1, D)

Architecture:
  Linear(D_C -> D)  [projection]
  x N_L layers of:
    MultiHeadAttention(D, N_H heads)
    + skip connection + BatchNorm
    FeedForward(D -> D_F -> D)
    + skip connection + BatchNorm
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer theo Vaswani et al. 2017.
    
    Tham chiếu: Section 2.5 (luận án), Figure 2.13
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: chiều embedding (D = 128)
            n_heads: số attention heads (N_H = 8)
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads   # D_H = 16

        # Shared weight matrix cho tất cả heads (hiệu quả hơn N_H separate matrices)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,      # (B, L_q, D)
        key:   torch.Tensor,      # (B, L_k, D)
        value: torch.Tensor,      # (B, L_k, D)
        mask:  torch.Tensor = None,  # (B, L_q, L_k) bool, True = masked out
    ) -> torch.Tensor:
        """
        Returns:
            output: (B, L_q, D)
        """
        B = query.size(0)
        H = self.n_heads
        D_H = self.d_head

        # Project và reshape thành (B, H, L, D_H)
        def split_heads(x):
            # x: (B, L, D) -> (B, H, L, D_H)
            B_, L, D = x.shape
            return x.view(B_, L, H, D_H).transpose(1, 2)

        Q = split_heads(self.W_q(query))   # (B, H, L_q, D_H)
        K = split_heads(self.W_k(key))     # (B, H, L_k, D_H)
        V = split_heads(self.W_v(value))   # (B, H, L_k, D_H)

        # Scaled dot-product attention
        scale = math.sqrt(D_H)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, L_q, L_k)

        if mask is not None:
            # mask: (B, L_q, L_k) -> (B, 1, L_q, L_k)
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, L_q, L_k)

        # Weighted combination
        out = torch.matmul(attn_weights, V)  # (B, H, L_q, D_H)
        
        # Merge heads: (B, H, L_q, D_H) -> (B, L_q, D)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """1 layer của Transformer Encoder.
    
    MHA + skip + norm + FF + skip + norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """
        Args:
            d_model: D = 128
            n_heads: N_H = 8
            d_ff:    D_F = 512 (feed-forward hidden dim)
        """
        super().__init__()
        self.mha   = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape
        
        # MHA sublayer
        h = self.mha(x, x, x)   # self-attention: Q=K=V=x
        x = x + h               # skip connection
        # BatchNorm expects (B, D, L) -> transpose
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)

        # FF sublayer
        h = self.ff(x)
        x = x + h
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        
        return x


class CustomerEncoder(nn.Module):
    """Transformer Encoder mã hóa tập khách hàng.
    
    Tham chiếu: Section 5.2.1, Figure 5.6, Table 6.1
    
    N_L = 3 layers, N_H = 8 heads, D = 128, D_F = 512
    """

    def __init__(
        self,
        d_input: int = 7,    # D_C: số features của customer (7 hoặc 8 với pending flag)
        d_model: int = 128,  # D
        n_heads: int = 8,    # N_H
        d_ff:    int = 512,  # D_F
        n_layers: int = 3,   # N_L
    ):
        """
        Args:
            d_input:  số features input per customer (default 7 = x,y,q,e,l,d,b)
                      Không dùng pending flag (IDX_XI) vì nó được xử lý riêng qua mask
            d_model:  chiều embedding output
            n_heads:  số attention heads
            d_ff:     feed-forward hidden dim
            n_layers: số Transformer layers
        
        Note: d_input phụ thuộc variant:
          - Full DS-CVRPTW: 7 (x,y,q,e,l,d,b)
          - CVRP only: 3 (x,y,q) — set D_C=3 trong Table 6.1
        """
        super().__init__()
        self.d_model = d_model
        
        # Input projection: D_C -> D
        self.input_proj = nn.Linear(d_input, d_model)
        
        # N_L Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        shared_state: torch.Tensor,    # (B, n+1, 8) — full shared state
        pending_mask: torch.Tensor = None,  # (B, n+1) float — IDX_XI column
    ) -> torch.Tensor:
        """Encode customers từ shared state.
        
        Args:
            shared_state: (B, n+1, 8) — đầy đủ shared state
            pending_mask: không dùng ở đây, được xử lý qua action masking
        
        Returns:
            h_customers: (B, n+1, D) — embeddings
        
        IMPORTANT: chỉ dùng 7 features đầu (bỏ IDX_XI=column 7)
        Pending flag được đưa vào action mask ở Travels Scorer, không ở đây.
        """
        # Lấy 7 features đầu: x, y, q, e, l, d, b
        x = shared_state[:, :, :7]    # (B, n+1, 7)
        
        # Input projection
        h = self.input_proj(x)        # (B, n+1, D)
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        return h                      # (B, n+1, D)