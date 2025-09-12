"""Auto-encoder and Variational Auto-Encoder modules for generation tasks."""

from .autoencoder import AutoEncoder, create_autoencoder
from .vae import VariationalAutoEncoder, create_vae, vae_loss

# Conditional import for inference modules to avoid dependency issues
try:
    from .inference_autoencoder import AutoEncoderInference
    from .inference_vae import VAEInference

    __all__ = [
        "AutoEncoder",
        "create_autoencoder",
        "AutoEncoderInference",
        "VariationalAutoEncoder",
        "create_vae",
        "vae_loss",
        "VAEInference",
    ]
except ImportError:
    __all__ = [
        "AutoEncoder",
        "create_autoencoder",
        "VariationalAutoEncoder",
        "create_vae",
        "vae_loss",
    ]
