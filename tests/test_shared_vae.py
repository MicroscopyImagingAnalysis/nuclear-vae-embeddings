import pytest

torch = pytest.importorskip("torch")

from nuclear_vae_embeddings.models import VAE


def test_vae_output_contract():
    model = VAE(nc=1, ngf=2, ndf=2, latent_variable_size=4, imsize=64, batchnorm=False)
    model.eval()
    with torch.no_grad():
        reconstruction, latent, mean, log_variance = model(torch.zeros(2, 1, 64, 64))
    assert reconstruction.shape == (2, 1, 64, 64)
    assert latent.shape == mean.shape == log_variance.shape == (2, 4)
