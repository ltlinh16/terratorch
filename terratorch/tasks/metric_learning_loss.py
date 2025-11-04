from collections.abc import Callable
from typing import Literal, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


from terratorch.models.model import ModelOutput
from terratorch.tasks.loss_handler import LossHandler

class MetricLearningLoss(Protocol):
    """Protocol for losses that require embeddings"""
    requires_embeddings: bool = True
    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the loss given embeddings, logits and targets"""
        ...

class MultiLabelSupConLoss(nn.Module):
    """
    Soft Supervised Contrastive Loss for multi-label classification.
    
    Args:
        tau (float): temperature scaling for contrastive loss (default: 0.1)
        similarity_mode (string): Label similarity computation mode. ('dot', 'jaccard', 'asymmetric')
    """
    eps = 1e-8 # numerical stability constant

    def __init__(self, tau: float = 0.1, similarity_mode: Literal['dot', 'jaccard', 'asymmetric'] = 'dot'):
        super().__init__()
        self.tau = tau
        self.similarity_mode = similarity_mode
    
    def compute_label_similarity(self, targets: torch.Tensor) -> torch.Tensor:
        if self.similarity_mode == 'dot':
            label_sim = torch.matmul(targets.float(), targets.float().T)  # [N, N]
            label_sim = label_sim / (label_sim.max() + self.eps)

        elif self.similarity_mode == 'jaccard':
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = torch.matmul(targets.float(), targets.float().T)  # [N, N]
            
            # Compute cardinality of each label set
            cardinality = targets.sum(dim=1, keepdim=True)  # [N, 1]
            
            # |A ∪ B| = |A| + |B| - |A ∩ B|
            union = cardinality + cardinality.T - intersection
            
            label_sim = intersection / (union + self.eps)
        
        elif self.similarity_mode == 'asymmetric':
            # Asymmetric: |A ∩ B| / |A|
            intersection = torch.matmul(targets.float(), targets.float().T)  # [N, N]
            cardinality = targets.sum(dim=1, keepdim=True)  # [N, 1]
            
            label_sim = intersection / (cardinality + self.eps)
        
        else:
            raise ValueError(f"Unknown similarity mode: {self.similarity_mode}")
        
        return label_sim


    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (Tensor): model output features (after normalization), shape [N, D]
            targets (Tensor): multi-hot labels, shape [N, C]
        Returns:
            Tensor: scalar loss value
        """

        # Compute soft label similarity matrix
        label_sim = self.compute_label_similarity(targets)

        # Compute cosine similarity of features scaled by temperature
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.tau  # [N, N]

        # Mask out self-similarity
        N = embeddings.size(0)
        mask = torch.eye(N, device=embeddings.device).bool()
        #min_value = torch.finfo(embeddings.dtype).min
        sim_matrix.masked_fill_(mask, float("-inf"))
        label_sim.masked_fill_(mask, 0.0)

        # Compute log-softmax over rows
        log_prob = F.log_softmax(sim_matrix, dim=1)  # [N, N]


        log_prob = log_prob.masked_fill(mask, 0.0)

        # Weighted mean per sample
        supcon_loss = -(label_sim * log_prob).sum(dim=1) / (label_sim.sum(dim=1) + 1e-8)
        loss = supcon_loss.mean()

        return loss

    
class FocalLoss(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = sigmoid_focal_loss(outputs, targets, reduction='mean')
        return loss


class JointLoss(nn.Module):
    requires_embeddings = True
    def __init__(
            self, 
            alpha: float=0.5, 
            metric_loss_temperature: float=0.1,
            similarity_mode: Literal['dot', 'jaccard', 'asymmetric'] = 'dot',
            classification_loss: Literal['bce', 'focal'] = 'bce'
            ):
        super().__init__()
        self.alpha = alpha
        self.metric_loss_temperature = metric_loss_temperature
        self.similarity_mode = similarity_mode
        
        # Loss functions
        self.contrastive_loss = MultiLabelSupConLoss(tau=metric_loss_temperature)
        if classification_loss == 'bce':
            self.classification_loss = nn.BCEWithLogitsLoss()
        elif classification_loss == 'focal':
            self.classification_loss = FocalLoss()
        
    def forward(self, embeddings, logits, labels):
        # Metric learning loss
        contrastive_loss = self.contrastive_loss(embeddings, labels)
        
        # Classification loss
        classification_loss = self.classification_loss(logits, labels)
        
        # Combined loss
        total_loss = (self.alpha * contrastive_loss + 
                     (1-self.alpha) * classification_loss)
        
        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss
        }