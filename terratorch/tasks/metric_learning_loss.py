from abc import ABC, abstractmethod
from typing import Dict, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricLossBase(nn.Module, ABC):
    """
    Base class for all metric learning losses.
    Operates on a dict of embeddings.
    """
    requires_embeddings: bool = True

    @abstractmethod
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        ...

class LabelSimilarityMixin:
    eps = 1e-8

    def _compute_label_similarity(
        self,
        targets: torch.Tensor,
        label_embeddings: torch.Tensor | None = None,
        semantic: bool = False
    ) -> torch.Tensor:
        if not semantic:
            return (targets.unsqueeze(0) == targets.unsqueeze(1)).float()

        sample_emb = label_embeddings[targets]
        sample_emb = torch.nn.functional.normalize(sample_emb, dim=1)
        sim = sample_emb @ sample_emb.T
        return sim.clamp(min=0.0)

class SupConLoss(MetricLossBase, LabelSimilarityMixin):
    def __init__(
        self,
        tau: float = 0.07,
        similarity_mode: Literal['hard', 'semantic'] = "hard",
        label_embedding_path: str | None = None,
    ):
        super().__init__()
        self.tau = tau
        self.semantic = similarity_mode == "semantic"

        if self.semantic:
            if label_embedding_path is None:
                raise ValueError("Semantic mode requires label embeddings")
            self.register_buffer(
                "label_embeddings",
                torch.load(label_embedding_path)
            )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        
        all_embeddings = torch.cat(list(embeddings.values()), dim=0)
        all_labels = labels.repeat(len(embeddings))

        sim = (all_embeddings @ all_embeddings.T) / self.tau
        N = sim.size(0)
        mask_self = torch.eye(N, device=sim.device).bool()

        label_sim = self._compute_label_similarity(
            all_labels,
            getattr(self, "label_embeddings", None),
            semantic=self.semantic,
        )

        log_prob = F.log_softmax(sim, dim=1)
        log_prob = log_prob.masked_fill(mask_self, 0.0)
        label_sim = label_sim.masked_fill(mask_self, 0.0)

        pos_sum = label_sim.sum(1)
        valid = pos_sum > 0

        if not valid.any():
            return sim.sum() * 0.0

        loss = -(label_sim * log_prob).sum(1)[valid] / pos_sum[valid]
        return loss.mean()


class NCALoss(MetricLossBase, LabelSimilarityMixin):
    def __init__(
        self,
        tau: float = 0.07,
        similarity_mode: Literal['hard', 'semantic'] = "hard",
        label_embedding_path: str | None = None,
    ):
        super().__init__()
        self.tau = tau
        self.semantic = similarity_mode == "semantic"

        if self.semantic:
            self.register_buffer(
                "label_embeddings",
                torch.load(label_embedding_path)
            )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        all_embeddings = torch.cat(list(embeddings.values()), dim=0)
        all_labels = labels.repeat(len(embeddings))

        sim = (all_embeddings @ all_embeddings.T) / self.tau
        N = sim.size(0)
        mask = torch.eye(N, device=sim.device).bool()

        label_sim = self._compute_label_similarity(
            all_labels,
            getattr(self, "label_embeddings", None),
            semantic=self.semantic,
        )

        sim = sim.masked_fill_(mask, float("-inf"))
        label_sim = label_sim.masked_fill_(mask, 0.0)

        exp_sim = torch.exp(sim)
        num = (exp_sim * label_sim).sum(1) + self.eps
        den = exp_sim.sum(1) + self.eps

        return (-torch.log(num / den)).mean()


class JointLoss(nn.Module):
    requires_embeddings = True

    def __init__(
        self,
        metric_loss: Literal['supcon', 'nca'] = 'supcon',
        classification_loss: Literal['ce'] = 'ce',
        alpha: float = 0.5,
        metric_loss_temperature: float = 0.07,
        metric_similarity_mode: Literal['hard', 'semantic'] = 'hard',
        label_embedding_path: str = None
    ):
        super().__init__()
        if metric_loss == "supcon":
            self.metric_loss = SupConLoss(tau=metric_loss_temperature, similarity_mode=metric_similarity_mode, label_embedding_path=label_embedding_path)
        elif metric_loss == "nca":
            self.metric_loss = NCALoss(tau=metric_loss_temperature, similarity_mode=metric_similarity_mode, label_embedding_path=label_embedding_path)
        else:
            raise ValueError(f"Unknown metric learning loss: {metric_loss}")
        
        if classification_loss == 'ce':
            self.classification_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown classification loss: {classification_loss}")
    
        
        self.alpha = alpha

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        metric = self.metric_loss(embeddings, labels)
        cls = self.classification_loss(logits, labels)

        return {
            "loss": self.alpha * metric + (1 - self.alpha) * cls,
            "metric_loss": metric,
            "classification_loss": cls,
        }
