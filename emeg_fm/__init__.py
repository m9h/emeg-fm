"""emeg_fm — diagnostics, interpretability, and realtime decoding for E/MEG foundation models.

Submodules:
    adapters  : model-agnostic HuggingFace adapter registry + factory
    eeg_fm    : REVE / LaBraM / ZUNA activation-extraction adapters
    sae       : TopK sparse autoencoder (Gao 2024 aux_k) in pure JAX
    alljoined : realtime-EMEG EEG->image retrieval (Alljoined-1.6M) helpers
    streaming : live LSL acquisition bridge + ring-buffer epoching (EEG-ExPy)
    device    : Layer-3 device front-end (mains/drift/reref) for live headsets
    montage   : channel-montage presets + headless MNE label validator
    stimuli   : image stimulus sets, CLIP gallery, presentation schedules
    decoder   : per-subject streaming REVE+ridge EEG->image decoder
"""
__all__ = ["adapters", "eeg_fm", "sae", "alljoined",
           "streaming", "device", "montage", "stimuli", "decoder"]
