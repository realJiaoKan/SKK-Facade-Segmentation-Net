import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# -----------------------------
# 1. Multi-class Dice Loss
# -----------------------------
class DiceLoss(nn.Module):
    """
    Multi-class Dice loss on logits.
    logits: (B, C, H, W)
    target: (B, H, W), long, class indices
    """

    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target.clone()
            target[~valid_mask] = 0
        else:
            valid_mask = None

        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B,H,W,C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1)
            target_one_hot = target_one_hot * valid_mask

        probs = F.softmax(logits, dim=1)
        if valid_mask is not None:
            probs = probs * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(target_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


# -----------------------------
# 2. Multi-class Focal Loss (softmax)
# -----------------------------
class FocalLoss(nn.Module):
    """
    Multi-class Focal loss on logits + index targets.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = logits.shape

        logits = logits.permute(0, 2, 3, 1).reshape(-1, c)  # (N, C)
        target = target.view(-1)  # (N,)

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            logits = logits[valid_mask]
            target = target[valid_mask]
            if target.numel() == 0:
                return logits.new_tensor(0.0, requires_grad=True)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target = target.long()
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, target)
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1.0 - p_t) ** self.gamma * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# -----------------------------
# 3. Tversky + Focal Tversky
# -----------------------------
class TverskyLoss(nn.Module):
    """
    Multi-class Tversky loss (Dice with FP/FN weighting).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target.clone()
            target[~valid_mask] = 0
        else:
            valid_mask = None

        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B,H,W,C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1)
            target_one_hot = target_one_hot * valid_mask

        probs = F.softmax(logits, dim=1)
        if valid_mask is not None:
            probs = probs * valid_mask

        dims = (0, 2, 3)
        tp = torch.sum(probs * target_one_hot, dims)
        fp = torch.sum(probs * (1.0 - target_one_hot), dims)
        fn = torch.sum((1.0 - probs) * target_one_hot, dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1.0 - tversky.mean()
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss: (1 - Tversky)^gamma
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 2.0,
        smooth: float = 1.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.base = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # base returns (1 - Tversky)
        base_loss = self.base(logits, target)  # = 1 - Tversky
        tversky = 1.0 - base_loss
        loss = (1.0 - tversky) ** self.gamma
        return loss


# -----------------------------
# 4. Combined losses: CE+Dice, Focal+Dice
# -----------------------------
class CEDiceLoss(nn.Module):
    """
    CrossEntropy + Dice.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return loss_ce + self.dice_weight * loss_dice


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
        self.dice = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(logits, target)
        loss_dice = self.dice(logits, target)
        return loss_focal + self.dice_weight * loss_dice


# -----------------------------
# 5. Simple wrappers for CE / BCE
# -----------------------------
def make_cross_entropy(
    weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    return nn.CrossEntropyLoss(weight=weight)


def make_bce_with_logits(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """
    For binary / multi-label segmentation.
    logits: (B,1,H,W) or (B,C,H,W)
    target: same shape, float 0/1
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# -----------------------------
# 6. Factory function
# -----------------------------
def get_criterion(
    name: str,
    num_classes: Optional[int] = None,
    weight: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    dice_weight: float = 1.0,
) -> nn.Module:
    """
    name: one of
      "ce", "bce", "dice",
      "ce_dice", "focal", "focal_dice",
      "tversky", "focal_tversky"
    """
    name = name.lower()

    if name == "ce":
        return make_cross_entropy(weight=weight)

    if name == "bce":
        return make_bce_with_logits(pos_weight=weight)

    if name == "dice":
        return DiceLoss(ignore_index=ignore_index)

    if name == "ce_dice":
        return CEDiceLoss(
            weight=weight,
            ignore_index=ignore_index,
            dice_weight=dice_weight,
        )

    if name == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            ignore_index=ignore_index,
            reduction="mean",
        )

    if name == "focal_dice":
        return FocalDiceLoss(
            gamma=gamma,
            alpha=alpha,
            ignore_index=ignore_index,
            dice_weight=dice_weight,
        )

    if name == "tversky":
        return TverskyLoss(ignore_index=ignore_index)

    if name == "focal_tversky":
        return FocalTverskyLoss(ignore_index=ignore_index, gamma=gamma)

    raise ValueError(f"Unknown loss name: {name}")
