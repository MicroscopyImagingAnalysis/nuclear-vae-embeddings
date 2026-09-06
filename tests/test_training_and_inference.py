import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

from nuclear_vae_embeddings import fit_vae, latent_feature_table, reconstruct_images
from nuclear_vae_embeddings.models import VAE


class CropDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": torch.full((1, 64, 64), index / 10, dtype=torch.float32),
            "object_id": f"nucleus-{index}",
            "condition": "baseline" if index < 2 else "stimulated",
        }


def test_fit_reconstruct_and_extract_dataframe():
    model = VAE(nc=1, ngf=2, ndf=2, latent_variable_size=4, imsize=64, batchnorm=False)
    loader = DataLoader(CropDataset(), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = fit_vae(model, loader, optimizer, epochs=1)
    assert history.columns.tolist() == ["epoch", "loss", "reconstruction_loss", "kl_loss"]
    assert history.shape == (1, 4)

    reconstructed = reconstruct_images(model, torch.zeros(2, 1, 64, 64))
    assert reconstructed.shape == (2, 1, 64, 64)

    features = latent_feature_table(model, loader, metadata_keys=["condition"])
    assert features["object_id"].tolist() == [f"nucleus-{index}" for index in range(4)]
    assert features["condition"].tolist() == ["baseline", "baseline", "stimulated", "stimulated"]
    assert [column for column in features if column.startswith("feature_")] == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    assert isinstance(features, pd.DataFrame)
