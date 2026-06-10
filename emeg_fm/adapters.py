"""Model-agnostic HuggingFace adapter machinery.

A minimal adapter/registry layer that lets a frozen pretrained model be
plugged into the JAX factory pattern (``make_*() -> (params, forward_fn)``)
used throughout this package. EEG foundation-model adapters (REVE, LaBraM,
ZUNA) live in :mod:`emeg_fm.eeg_fm` and register themselves here.

This is the slimmed, EEG-only extraction of the original ``jaxoccoli``
``hf_encoder`` module — the fMRI-specific adapters (TRIBEv2, Raramuri) and the
cortical-projection code have been dropped.
"""
from __future__ import annotations

import abc
from typing import Any, NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# PyTorch -> JAX bridge
# ---------------------------------------------------------------------------

def torch_to_jax(tensor) -> "jnp.ndarray":
    """Convert a PyTorch tensor, numpy array, or JAX array to a JAX array."""
    import jax.numpy as jnp

    if isinstance(tensor, jnp.ndarray):
        return tensor
    if isinstance(tensor, np.ndarray):
        return jnp.array(tensor)
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return jnp.array(tensor.detach().cpu().numpy())
    except ImportError:
        pass
    return jnp.array(np.asarray(tensor))


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type["HFModelAdapter"]] = {}


def register_adapter(model_id: str, adapter_cls: type["HFModelAdapter"]) -> None:
    """Register an adapter class for a HuggingFace model ID."""
    _ADAPTER_REGISTRY[model_id] = adapter_cls


def get_adapter(model_id: str) -> type["HFModelAdapter"]:
    """Look up the adapter class for a model ID.

    Raises
    ------
    KeyError : if no adapter is registered for the model ID
    """
    if model_id not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for '{model_id}'. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}. "
            f"Register one with register_adapter(model_id, adapter_cls)."
        )
    return _ADAPTER_REGISTRY[model_id]


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class HFModelAdapter(abc.ABC):
    """Abstract base class for HuggingFace model adapters.

    Subclass this to integrate a new pretrained model. Each adapter must
    implement:
        - ``load_model`` — download/load the pretrained model
        - ``extract_features`` — run inference and return numpy arrays
        - ``output_dim`` — dimensionality of the feature output
        - ``output_space`` — description of the output space
    """

    @abc.abstractmethod
    def load_model(self, model_id: str, cache_dir: str | None = None,
                   **kwargs) -> Any:
        """Load the pretrained model and return the loaded object."""

    @abc.abstractmethod
    def extract_features(self, model: Any, inputs: dict, **kwargs) -> np.ndarray:
        """Run the model and return features as a numpy array."""

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the feature output."""

    @property
    @abc.abstractmethod
    def output_space(self) -> str:
        """Description of the output coordinate space."""


# ---------------------------------------------------------------------------
# make_hf_encoder factory
# ---------------------------------------------------------------------------

class HFEncoderParams(NamedTuple):
    """Parameters for a HuggingFace encoder in the factory pattern."""
    model_id: str
    model: Any
    adapter: HFModelAdapter


def make_hf_encoder(
    model_id: str,
    *,
    key: jax.Array | None = None,
    cache_dir: str | None = None,
    lazy: bool = False,
    adapter: HFModelAdapter | None = None,
    **load_kwargs,
) -> tuple[HFEncoderParams, callable]:
    """Create a HuggingFace encoder following the JAX factory pattern.

    Parameters
    ----------
    model_id : HuggingFace model identifier
    key : JAX PRNG key (unused by frozen models, kept for API consistency)
    cache_dir : local directory for model weights cache
    lazy : if True, defer model loading until first forward call
    adapter : optional pre-instantiated adapter (auto-detected if None)
    **load_kwargs : passed to adapter.load_model

    Returns
    -------
    (HFEncoderParams, forward_fn)
        forward_fn(params, inputs) -> jnp.ndarray
    """
    if adapter is None:
        adapter_cls = get_adapter(model_id)
        adapter = adapter_cls()

    if lazy:
        loaded_model = None
    else:
        loaded_model = adapter.load_model(
            model_id, cache_dir=cache_dir, **load_kwargs,
        )

    params = HFEncoderParams(
        model_id=model_id,
        model=loaded_model,
        adapter=adapter,
    )

    def forward_fn(params: HFEncoderParams, inputs: dict) -> jnp.ndarray:
        model = params.model
        if model is None:
            model = params.adapter.load_model(
                params.model_id, cache_dir=cache_dir, **load_kwargs,
            )
        features = params.adapter.extract_features(model, inputs)
        return torch_to_jax(features)

    return params, forward_fn
