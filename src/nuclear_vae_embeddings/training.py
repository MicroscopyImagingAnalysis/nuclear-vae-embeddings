"""Training helpers for variational microscopy representations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd
import torch
from torch.nn import functional as F


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction and KL losses using summed MSE."""
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    total_loss = reconstruction_loss + float(kl_weight) * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def _batch_images(batch, image_key: str) -> torch.Tensor:
    if isinstance(batch, Mapping):
        return batch[image_key]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _model_device(model: torch.nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return next(model.parameters()).device


def fit_vae(
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    device: str | torch.device | None = None,
    image_key: str = "image",
    kl_weight: float | None = None,
) -> pd.DataFrame:
    """Fit a VAE and return one row of loss diagnostics per epoch.

    The loader may yield image tensors directly, tuples whose first item is the
    image tensor, or mappings containing ``image_key``. Dataset splitting stays
    outside this helper so existing train/test manifests remain authoritative.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    target_device = _model_device(model, device)
    model.to(target_device)
    weight = float(model.lamb if kl_weight is None else kl_weight)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
        sample_count = 0
        for batch in dataloader:
            images = _batch_images(batch, image_key).to(target_device)
            optimizer.zero_grad()
            reconstruction, _, mean, log_variance = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                images,
                mean,
                log_variance,
                kl_weight=weight,
            )
            loss.backward()
            optimizer.step()
            batch_size = int(images.shape[0])
            sample_count += batch_size
            totals["loss"] += loss.detach().item()
            totals["reconstruction_loss"] += reconstruction_loss.detach().item()
            totals["kl_loss"] += kl_loss.detach().item()

        if sample_count == 0:
            raise ValueError("dataloader yielded no images")
        history.append(
            {
                "epoch": epoch,
                **{name: value / sample_count for name, value in totals.items()},
            }
        )
    return pd.DataFrame(history)


__all__ = ["fit_vae", "vae_loss"]
