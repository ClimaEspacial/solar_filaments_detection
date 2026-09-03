"""BCE + Dice loss (as in EdgeAttNet) - balances pixel-wise correctness with
the shape-overlap objective the competition actually scores on."""
import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_dice_loss(logits, target, bce_weight: float = 1.0):
    """`bce_weight` scales the BCE term relative to Dice. Crops are roughly
    balanced so 1.0 is fine there; on full-disk images filaments are ~1% of
    pixels, where a lower BCE weight keeps the loss from being dominated by
    trivially-correct background."""
    return bce_weight * F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def dice_score(logits, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thresh).float()
    num = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (num / den).mean().item()
