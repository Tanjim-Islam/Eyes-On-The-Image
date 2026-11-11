import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F


def alignment_loss(saliency, gaze, mask, box, quality):
    """Compare 14 x 14 saliency with duration-weighted human fixations."""
    valid = mask.any(dim=1)
    if not valid.any():
        return saliency.new_zeros(())
    scores = saliency[valid].float().flatten(1)
    fixes, mask = gaze[valid], mask[valid]
    size = saliency.shape[-1]
    axis = torch.linspace(0, 1, size, device=scores.device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
    distance = (fixes[..., :2].unsqueeze(2) - grid).square().sum(dim=-1)
    heat = (-0.5 * distance * (size - 1) ** 2).exp()
    target = (heat * (fixes[..., 2] * mask).unsqueeze(-1)).sum(dim=1)
    anatomy = F.adaptive_avg_pool2d(box[valid].float(), (size, size)).flatten(1)
    target = target * (1 + 0.2 * anatomy)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probability = scores.softmax(dim=1)
    expected = target / target.amax(dim=1, keepdim=True).clamp_min(1e-8)
    observed = scores.sigmoid()
    mse = (observed - expected).square().mean(dim=1)
    kl = F.kl_div(scores.log_softmax(dim=1), target, reduction="none").sum(dim=1)
    pearson = F.cosine_similarity(
        observed - observed.mean(dim=1, keepdim=True),
        expected - expected.mean(dim=1, keepdim=True),
    )
    centroid = ((probability - target) @ grid).norm(dim=1) / np.sqrt(2)
    trust = (mask.sum(dim=1) / mask.shape[1]).sqrt() * (0.5 + 0.5 * quality[valid])
    return (trust * (mse + kl + 1 - pearson + centroid)).mean()


def gaze_metrics(saliency, fixations):
    """Calculate spatial alignment and entropy in bits from the recorded fixations."""
    size = saliency.shape[0]
    points = np.asarray([row[:3] for row in fixations])
    xy = np.rint(points[:, :2] * (size - 1)).astype(int).clip(0, size - 1)
    density = np.zeros_like(saliency, dtype=float)
    np.add.at(density, (xy[:, 1], xy[:, 0]), points[:, 2])
    human = gaussian_filter(density, sigma=max(1, size / 16))
    human /= human.max()
    model = saliency.astype(float) - saliency.min()
    model /= max(model.max(), 1e-12)
    std = model.std()
    pearson = np.corrcoef(model.ravel(), human.ravel())[0, 1] if std and human.std() else 0
    nss = ((model - model.mean()) / std)[xy[:, 1], xy[:, 0]].mean() if std else 0
    p = (human.ravel() + 1e-12) / (human.sum() + human.size * 1e-12)
    q = (model.ravel() + 1e-12) / (model.sum() + model.size * 1e-12)
    m = (p + q) / 2
    return {
        "pearson": float(pearson),
        "mse": float(((model - human) ** 2).mean()),
        "jsd": float(((p * np.log2(p / m)).sum() + (q * np.log2(q / m)).sum()) / 2),
        "nss": float(nss),
        "human_entropy_bits": float(-(p * np.log2(p)).sum()),
        "model_entropy_bits": float(-(q * np.log2(q)).sum()),
    }
