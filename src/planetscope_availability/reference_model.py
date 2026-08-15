"""Reference reconstruction of the selected checkpoint's parameterized architecture.

The state-dict shapes uniquely support the layers and parameter count defined
below. Activation functions and sequence pooling are not encoded in tensor
shapes; therefore the forward path uses clearly documented reference choices
(ReLU + masked mean pooling) and must not be described as the archival training
source unless checked against the original script.
"""

from __future__ import annotations
import math
import torch
from torch import nn


class AvailabilityAwareTransformer(nn.Module):
    def __init__(
        self,
        n_features=14,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.10,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(n_features, d_model)
        self.doy_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),  # reference choice; verify against original source
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),  # reference choice; verify against original source
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @staticmethod
    def doy_components(doy):
        """Map day-of-year to two periodic components."""
        angle = 2.0 * math.pi * doy / 366.0
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def forward(self, features, doy, valid_mask):
        """Reference forward path using padding masks and masked mean pooling."""
        x = self.feature_projection(features) + self.doy_projection(self.doy_components(doy))
        padding_mask = ~valid_mask.bool()
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        w = valid_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return self.regression_head(pooled).squeeze(-1)


def parameter_count(model=None):
    model = AvailabilityAwareTransformer() if model is None else model
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = AvailabilityAwareTransformer()
    print(parameter_count(model))
