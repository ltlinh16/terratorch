from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from terratorch.registry import TERRATORCH_DECODER_REGISTRY



@TERRATORCH_DECODER_REGISTRY.register
class GlobalAveragePoolMLPDecoder(nn.Module):
    """
    Decoder that aggregates features for image-level tasks.

    It operates as follows:
    1. For each chosen feature map, it collapses spatial dimensions via Global Average Pooling.
    2. The resulting vectors are concatenated.
    3. The single concatenated vector is passed through a MLP.
    4. The final output can be optionally L2-normalized for constrastive learning.
    """
    includes_head = True
    def __init__(
            self, 
            channel_list:list[int],
            hidden_dim: int = 768, 
            out_dim: int = 128, 
            use_all_features: bool = True, 
            dropout: float = 0.1,
            normalized_output: bool = True,
            num_classes: Optional[int] = None
            ):
        super().__init__()
        self.use_all_features = use_all_features
        self.normalized_output = normalized_output
        self.out_channels = out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        if self.use_all_features:
            self.input_channels = channel_list
        else:
            self.input_channels = [channel_list[-1]]
        
        input_dim = sum(self.input_channels)
        self.pre_ln = nn.ModuleList([nn.LayerNorm(num_channels) for num_channels in self.input_channels])
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.out_channels)
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features: list of feature maps (B, C, H, W)
        Returns:
            z: (B, out_dim) optionally L2-normalized embeddings
        """
        if len(features) == 0:
            raise ValueError("Empty feature list")
        
        device = features[0].device

        # choose features: either all or just last
        selected_features = features if self.use_all_features else [features[-1]]
        pooled = []
        for i,f in enumerate(selected_features):
            x = self.pool(f).flatten(1)   # (B, C)
            x = self.pre_ln[i](x.to(device))
            pooled.append(x)

        x = torch.cat(pooled, dim=1)  # (B, C_total)
        z = self._mlp(x)
        if self.normalized_output:
            z = F.normalize(z, p=2, dim=1)
        return z



@TERRATORCH_DECODER_REGISTRY.register
class ProjectorHead(nn.Module):
    """
    Simple projector head for metric learning.
    Takes the final feature  and projects it.
    """
    includes_head = True
    def __init__(self, channel_list:list[int], out_dim: int = 128, normalized_output: bool = True, num_classes: Optional[int] = None) -> None:
        super().__init__()

        input_dim = channel_list[-1]
        self.normalized_output = normalized_output
        self.out_channels = out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, out_dim)
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        x = features[-1] 
        
        # If input is 4D (Spatial Grid: B, C, H, W), pool it to (B, C)
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        
        embeddings = self.projector(x) 
        
        if self.normalized_output:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings