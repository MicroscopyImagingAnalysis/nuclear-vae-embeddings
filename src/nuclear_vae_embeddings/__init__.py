"""Variational representations for microscopy image crops."""

from .inference import latent_feature_table, reconstruct_images
from .training import fit_vae, vae_loss

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "fit_vae",
    "latent_feature_table",
    "reconstruct_images",
    "vae_loss",
]
