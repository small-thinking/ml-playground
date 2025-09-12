"""Auto-encoder module for generation tasks."""

from .autoencoder import AutoEncoder, create_autoencoder

# Conditional import for inference module to avoid dependency issues
try:
    from .inference_autoencoder import AutoEncoderInference

    __all__ = [
        "AutoEncoder",
        "create_autoencoder",
        "AutoEncoderInference",
    ]
except ImportError:
    __all__ = [
        "AutoEncoder",
        "create_autoencoder",
    ]
