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
from nuclear_vae_embeddings.models import VAE

model = VAE(nc=1, latent_variable_size=128, imsize=64)
reconstruction, latent, mean, log_variance = model(
    torch.zeros(8, 1, 64, 64)
)
```

The returned latent vectors can be joined to classical morphology and texture
features for downstream visualization, clustering or supervised analysis.
