import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- DLR-BackLoss (Dynamic Linear Recurrent BackLoss) Implementation ---

class ParallelDLRBackLossLayer(nn.Module):
    """
    Core DLR-BackLoss Layer: Parallelized Linear Recurrence O(n)
    Uses Cumulative Statistics for dynamic gating.
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 3)

    def forward(self, x):
        B, N, D = x.shape
        # Linear projection to generate Q, K, V components
        gates = self.gate_proj(x)
        q, k, v = gates.chunk(3, dim=-1)

        # BackLoss Statistical Engine (Vectorized Cumulative Stats)
        k_cumsum = torch.cumsum(k, dim=1)
        counts = torch.arange(1, N + 1, device=x.device).view(1, -1, 1)
        mu = k_cumsum / counts  # Cumulative mean of keys

        # BackLoss Gating Mechanism (Selectivity based on deviation from mean)
        alpha = torch.sigmoid(q - mu)
        
        # Parallel Recurrence using Cumulative Weighted Average
        weighted_v = alpha * v
        out = torch.cumsum(weighted_v, dim=1) / (torch.cumsum(alpha, dim=1) + 1e-9)
        return out

class DLRBackLossModel(nn.Module):
    """
    Deep Language Model Architecture utilizing DLR-BackLoss Blocks
    """
    def __init__(self, vocab_size, d_model, n_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'core': ParallelDLRBackLossLayer(d_model),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        for layer in self.layers:
            res = x
            x = layer['core'](x)
            x = layer['norm'](x + res) # Residual Connection
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

print("DLR-BackLoss Architecture Loaded. Ready for GitHub Export.")
