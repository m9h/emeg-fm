"""emeg_fm — diagnostics, interpretability, and realtime decoding for E/MEG foundation models.

Submodules:
    adapters  : model-agnostic HuggingFace adapter registry + factory
    eeg_fm    : REVE / LaBraM / ZUNA activation-extraction adapters
    sae       : TopK sparse autoencoder (Gao 2024 aux_k) in pure JAX
    alljoined : realtime-EMEG EEG->image retrieval (Alljoined-1.6M) helpers
"""
__all__ = ["adapters", "eeg_fm", "sae", "alljoined"]
