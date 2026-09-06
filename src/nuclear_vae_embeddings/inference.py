"""Reconstruction and feature-table helpers for trained VAE models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import torch


def _model_device(model: torch.nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return next(model.parameters()).device


def reconstruct_images(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    use_mean: bool = False,
) -> torch.Tensor:
    """Reconstruct images from sampled latents, or means when requested."""
    target_device = _model_device(model, device)
    model.to(target_device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        mean, log_variance = model.encode(images.to(target_device))
        latent = mean if use_mean else model.reparameterize(mean, log_variance)
        reconstruction = model.decode(latent).cpu()
    model.train(was_training)
    return reconstruction


def _as_list(values) -> list:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    if isinstance(values, (str, bytes)):
        return [values]
    return list(values)


def latent_feature_table(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    *,
    image_key: str = "image",
    id_key: str = "object_id",
    metadata_keys: Iterable[str] = (),
    feature_prefix: str = "feature_",
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Encode a loader into an identity-preserving latent-feature dataframe.

    Latent means are used to match the deterministic feature extraction pattern
    in the analysis scripts. Each loader batch must contain images and object
    identifiers; requested metadata columns are copied alongside the features.
    """
    target_device = _model_device(model, device)
    model.to(target_device)
    was_training = model.training
    model.eval()
    tables: list[pd.DataFrame] = []

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError("latent_feature_table expects mapping batches")
            if image_key not in batch or id_key not in batch:
                raise KeyError(f"each batch must contain {image_key!r} and {id_key!r}")
            images = batch[image_key]
            if not isinstance(images, torch.Tensor):
                images = torch.as_tensor(images)
            mean, _ = model.encode(images.to(target_device))
            feature_table = pd.DataFrame(mean.cpu().numpy()).add_prefix(feature_prefix)
            metadata = {id_key: _as_list(batch[id_key])}
            for key in metadata_keys:
                metadata[key] = _as_list(batch[key])
            metadata_table = pd.DataFrame(metadata)
            if len(metadata_table) != len(feature_table):
                raise ValueError("metadata and image batch lengths differ")
            tables.append(pd.concat([metadata_table, feature_table], axis=1))

    model.train(was_training)
    if not tables:
        return pd.DataFrame(columns=[id_key])
    return pd.concat(tables, ignore_index=True)


__all__ = ["latent_feature_table", "reconstruct_images"]
