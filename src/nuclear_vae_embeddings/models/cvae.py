"""Conditional VAE components and latent-space classifiers."""

from __future__ import annotations

import torch.nn as nn

from .vae import VAE


class CVAE(VAE):
    """VAE with an additional classification-loss weight."""

    def __init__(
        self,
        nc=1,
        ngf=128,
        ndf=128,
        latent_variable_size=128,
        imsize=64,
        lamb=0.0001,
        lamb2=0.0001,
        batchnorm=True,
    ):
        super().__init__(nc, ngf, ndf, latent_variable_size, imsize, lamb, batchnorm)
        self.lamb2 = lamb2
        self.loss_names.append("clf_loss")


class LatentClassifier(nn.Module):
    """Multilayer classifier for latent representations."""

    def __init__(self, nz, n_hidden=1024, n_out=2):
        super().__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class LinearLatentClassifier(nn.Module):
    """Single-layer classifier for latent representations."""

    def __init__(self, nz, n_out=2):
        super().__init__()
        self.nz = nz
        self.n_out = n_out
        self.net = nn.Sequential(nn.Linear(nz, n_out))

    def forward(self, x):
        return self.net(x)


__all__ = ["CVAE", "LatentClassifier", "LinearLatentClassifier"]
