from collections.abc import Callable
from typing import Literal, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from collections import defaultdict


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

    def __init__(self, tau: float = 0.1, similarity_mode: Literal['dot', 'jaccard', 'asymmetric'] = 'dot', label_embedding_path: str = None):
        super().__init__()
        self.tau = tau
        self.similarity_mode = similarity_mode

        if self.similarity_mode == 'semantic':
            if label_embedding_path is None:
                raise ValueError("Must provide 'label_embedding_path' for 'semantic' mode.")
            label_embeddings = torch.load(label_embedding_path)
            self.register_buffer('label_embeddings', label_embeddings)
            print(f"Loaded semantic label embeddings of shape {self.label_embeddings.shape}")
    
    def compute_label_similarity(self, targets: torch.Tensor) -> torch.Tensor:
        targets_float = targets.float()
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

        elif self.similarity_mode == 'semantic':
            # 1. Compute a single semantic vector for each sample in the batch
            #    This is done by averaging the label embeddings for that sample.
            # `targets_float` is [N, 43]
            # `self.label_embeddings` is [43, D]
            
            # Sum of embeddings for present labels: [N, D]
            sum_embeddings = torch.matmul(targets_float, self.label_embeddings)
            
            # Count of labels per sample: [N, 1]
            label_counts = targets_float.sum(dim=1, keepdim=True) + self.eps
            
            # Average semantic embedding for each sample: [N, D]
            sample_semantic_embeddings = sum_embeddings / label_counts
            
            # 2. Compute cosine similarity between all pairs of sample embeddings
            sample_semantic_embeddings_norm = F.normalize(sample_semantic_embeddings, p=2, dim=1)
            label_sim = torch.matmul(sample_semantic_embeddings_norm, 
                                     sample_semantic_embeddings_norm.T)
            
            # 3. Clamp to [0, 1] range for the loss function
            #    Cosine similarity is [-1, 1]. We only care about positive relations.
            label_sim = label_sim.clamp(min=0.0)
        
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
            classification_loss: Literal['bce', 'focal'] = 'bce',
            label_embedding_path: str = None
            ):
        super().__init__()
        self.alpha = alpha
        
        # Loss functions
        self.contrastive_loss = MultiLabelSupConLoss(tau=metric_loss_temperature, similarity_mode=similarity_mode, label_embedding_path=label_embedding_path)
        if classification_loss == 'bce':
            self.classification_loss = nn.BCEWithLogitsLoss()
        elif classification_loss == 'focal':
            self.classification_loss = FocalLoss()
        print(f"Initialized JointLoss with alpha={alpha}, metric_loss_temperature={metric_loss_temperature}, similarity_mode={similarity_mode}, classification_loss={classification_loss}")
        
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


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for single-label (multiclass) classification.
    Based on "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020).
    
    Args:
        tau (float): Temperature scaling for contrastive loss (default: 0.07)
        similarity_mode (str): Label similarity computation mode ('hard', 'semantic')
    """
    eps = 1e-8  # numerical stability constant

    def __init__(
        self, 
        tau: float = 0.07, 
        similarity_mode: Literal['hard', 'semantic'] = 'hard',
        label_embedding_path: str = None
    ):
        super().__init__()
        self.tau = tau
        self.similarity_mode = similarity_mode

        if self.similarity_mode == 'semantic':
            if label_embedding_path is None:
                raise ValueError("Must provide 'label_embedding_path' for 'semantic' mode.")
            label_embeddings = torch.load(label_embedding_path)
            self.register_buffer('label_embeddings', label_embeddings)
            print(f"Loaded semantic label embeddings of shape {self.label_embeddings.shape}")
    
    def compute_label_similarity(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between samples based on their labels.
        
        Args:
            targets: Class labels, shape [N]
        
        Returns:
            Label similarity matrix, shape [N, N]
        """
        if self.similarity_mode == 'hard':
            # Hard matching: 1 if same class, 0 otherwise
            label_sim = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        
        elif self.similarity_mode == 'semantic':
            # Semantic similarity using pre-computed label embeddings
            # targets: [N] - class indices
            # self.label_embeddings: [num_classes, D]
            
            # Get embeddings for each sample's class: [N, D]
            sample_embeddings = self.label_embeddings[targets]
            
            # Compute cosine similarity between all pairs
            sample_embeddings_norm = F.normalize(sample_embeddings, p=2, dim=1)
            label_sim = torch.matmul(sample_embeddings_norm, sample_embeddings_norm.T)
            
            # Clamp to [0, 1] range
            label_sim = label_sim.clamp(min=0.0)
        
        else:
            raise ValueError(f"Unknown similarity mode: {self.similarity_mode}")
        
        return label_sim

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Model output features (after normalization), shape [N, D]
            targets: Class labels, shape [N]
        
        Returns:
            Scalar loss value
        """
        # Compute label similarity matrix
        label_sim = self.compute_label_similarity(targets)  # [N, N]

        # Compute cosine similarity of features scaled by temperature
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.tau  # [N, N]

        # Mask out self-similarity
        N = embeddings.size(0)
        mask = torch.eye(N, device=embeddings.device).bool()
        sim_matrix.masked_fill_(mask, float("-inf"))
        label_sim.masked_fill_(mask, 0.0)

        # Compute log-softmax over rows
        log_prob = F.log_softmax(sim_matrix, dim=1)  # [N, N]
        log_prob = log_prob.masked_fill(mask, 0.0)

        # Weighted mean per sample
        # For hard matching, this reduces to standard SupCon
        # For semantic matching, this uses soft label similarities
        supcon_loss = -(label_sim * log_prob).sum(dim=1) / (label_sim.sum(dim=1) + self.eps)
        loss = supcon_loss.mean()

        return loss


class MulticlassFocalLoss(nn.Module):
    """
    Focal Loss for multiclass classification.
    
    Args:
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits, shape [N, C]
            targets: Ground truth class indices, shape [N]
        
        Returns:
            Scalar loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get predicted probabilities
        p = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiClassJointLoss(nn.Module):
    """
    Joint loss combining metric learning (contrastive) and classification losses
    for single-label classification tasks.
    
    Args:
        alpha (float): Weight for contrastive loss. Classification loss weight is (1-alpha)
        metric_loss_temperature (float): Temperature parameter for contrastive loss
        similarity_mode (str): How to compute label similarity ('hard', 'semantic')
        classification_loss (str): Type of classification loss ('ce', 'focal')
        label_embedding_path (str): Path to label embeddings (required for 'semantic' mode)
        focal_alpha (float): Alpha parameter for focal loss
        focal_gamma (float): Gamma parameter for focal loss
    """
    requires_embeddings = True
    
    def __init__(
        self,
        alpha: float = 0.5,
        metric_loss_temperature: float = 0.07,
        similarity_mode: Literal['hard', 'semantic'] = 'hard',
        classification_loss: Literal['ce', 'focal'] = 'ce',
        label_embedding_path: str = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        
        # Contrastive loss
        self.contrastive_loss = SupConLoss(
            tau=metric_loss_temperature,
            similarity_mode=similarity_mode,
            label_embedding_path=label_embedding_path
        )
        
        # Classification loss
        if classification_loss == 'ce':
            self.classification_loss = nn.CrossEntropyLoss()
        elif classification_loss == 'focal':
            self.classification_loss = MulticlassFocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma
            )
        else:
            raise ValueError(f"Unknown classification loss: {classification_loss}")
        
        print(f"Initialized JointLoss for classification with alpha={alpha}, "
              f"metric_loss_temperature={metric_loss_temperature}, "
              f"similarity_mode={similarity_mode}, classification_loss={classification_loss}")
        
    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute joint loss.
        
        Args:
            embeddings: Normalized feature embeddings, shape [N, D]
            logits: Classification logits, shape [N, C]
            labels: Ground truth class indices, shape [N]
        
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Metric learning loss
        contrastive_loss = self.contrastive_loss(embeddings, labels)
        
        # Classification loss
        classification_loss = self.classification_loss(logits, labels)
        
        # Combined loss
        total_loss = (self.alpha * contrastive_loss + 
                     (1 - self.alpha) * classification_loss)
        
        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss
        }