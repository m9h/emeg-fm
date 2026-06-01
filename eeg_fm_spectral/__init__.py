"""eeg_fm_spectral — spectral diagnostics + SAE interpretability for EEG foundation models.

Submodules:
    adapters : model-agnostic HuggingFace adapter registry + factory
    eeg_fm   : REVE / LaBraM / ZUNA activation-extraction adapters
    sae      : TopK sparse autoencoder (Gao 2024 aux_k) in pure JAX
"""
__all__ = ["adapters", "eeg_fm", "sae"]
