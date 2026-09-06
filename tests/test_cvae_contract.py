import pytest

torch = pytest.importorskip("torch")

from nuclear_vae_embeddings.models import CVAE, LatentClassifier, LinearLatentClassifier


def test_cvae_and_classifier_shapes():
    model = CVAE(nc=1, ngf=2, ndf=2, latent_variable_size=4, imsize=64, batchnorm=False)
    model.eval()
    classifier = LatentClassifier(4, n_hidden=8, n_out=3)
    linear = LinearLatentClassifier(4, n_out=3)
    with torch.no_grad():
        _, latent, _, _ = model(torch.zeros(2, 1, 64, 64))
        assert classifier(latent).shape == (2, 3)
        assert linear(latent).shape == (2, 3)
