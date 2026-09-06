# nuclear-vae-embeddings

Convolutional variational autoencoders for learning compact representations of
nuclear microscopy crops. The package includes an unsupervised VAE, a
conditional training model and latent-space classifiers.

## Install

```bash
python -m pip install .
```

## Example

```python
import torch
from torch.utils.data import DataLoader

from nuclear_vae_embeddings import fit_vae, latent_feature_table, reconstruct_images
from nuclear_vae_embeddings.models import VAE

model = VAE(nc=1, latent_variable_size=128, imsize=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Each dataset row contains image, object_id and any metadata to retain.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
feature_loader = DataLoader(feature_dataset, batch_size=64, shuffle=False)
history = fit_vae(model, train_loader, optimizer, epochs=20)

validation_images = next(iter(validation_loader))["image"]
reconstruction = reconstruct_images(model, validation_images)

latent_table = latent_feature_table(
    model,
    feature_loader,
    id_key="object_id",
    metadata_keys=["condition", "batch"],
)
latent_table.to_csv("nuclear_latent_features.csv", index=False)
```

Training consumes the caller's existing data split; it does not create or
modify train/test manifests. The extracted latent means retain object identity
and selected metadata so they can be joined to morphology and texture features
for downstream visualization, clustering or supervised analysis.
