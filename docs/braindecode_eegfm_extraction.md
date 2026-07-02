# Running the EEG-FM zoo with one extractor: what braindecode 1.5 gives us

*Notes from wiring the TDBRAIN brain-age benchmark's foundation-model column
(2026-07). Shareable — this is the pattern the NeuroTechX Container Center can ship.*

## The problem it solves

To benchmark an EEG **foundation-model zoo** (BIOT, BENDR, CBraMod, LUNA, LaBraM,
EEGPT, REVE, …) as frozen feature extractors, the naïve path is **one bespoke
adapter per model**. Each model differs in:

- **checkpoint format** (`.pth` vs safetensors vs HF custom-code),
- **input contract** (sampling rate, window length, channel count/montage,
  normalization),
- **forward API** (where the pooled embedding lives, how to hook it),
- **dependencies** (ZUNA needs `vector_quantize_pytorch` + `einx`; others need
  mamba-ssm, etc.).

Our hand-written `emeg_fm/eeg_fm.py` follows this route: a full `HFModelAdapter`
subclass per model (`REVEAdapter`, `LaBraMAdapter`, `ZunaAdapter`), each ~100
lines with its own loader, forward hook, and input-dict contract. Adding a model =
writing and debugging another adapter. That's why "run the whole zoo" reads as a
multi-week job.

## What braindecode 1.5.2 changes

braindecode now ships the **zoo itself** as first-class `braindecode.models.*`
classes. In our 26.06 container: `BIOT, BENDR, CBraMod, LUNA, LaBraM, EEGPT, REVE,
SignalJEPA (+ Interpolated variants)` and dozens more. Three properties make one
extractor cover all of them:

1. **`Model.from_pretrained(hf_id)`** — uniform Hugging-Face-hub loading with the
   checkpoint config baked in (`n_chans`, `n_times`, `sfreq`, `input_window_seconds`
   read back off the instance). No per-model checkpoint plumbing.

2. **`EEGModuleMixin` → a shared `final_layer`.** Every braindecode model ends in a
   `final_layer` classification head. So the pooled representation is *always* the
   **input to `final_layer`**, captured by one line that works for every model:

   ```python
   cap = {}
   model.final_layer.register_forward_pre_hook(
       lambda mod, args: cap.__setitem__("z", args[0].detach()))
   model(x)                      # forward
   embedding = cap["z"]          # (B, D) — the frozen feature, any model
   ```

   No per-model knowledge of block names or hook paths (contrast the hand adapters,
   which each hard-code a `hook_path` into the transformer blocks).

3. **`Interpolated{BIOT,BENDR,LaBraM,…}`** — pass your **`chs_info`** (an MNE montage)
   and the wrapper *spatially interpolates* your electrodes onto the model's
   pretrained layout. So arbitrary channel counts just work — TDBRAIN's 26×10-20 →
   BIOT's 18, BENDR's 20 — with **zero manual channel mapping**:

   ```python
   info = mne.create_info(ch_names, sfreq, "eeg"); info.set_montage("standard_1005")
   m = braindecode.models.InterpolatedBIOT.from_pretrained(
           hf_id, chs_info=info["chs"], n_outputs=2, n_times=win, sfreq=sfreq)
   ```

### The payoff

`scripts/braindecode_fm_brain_age.py` is **~150 lines total** and covers the whole
braindecode-native zoo. Adding a model is **one row of data**, not a new class:

```python
BD_SPEC = {                       # model -> (class, hf_id, sfreq, window_samples)
    "biot":  ("InterpolatedBIOT",  "braindecode/biot-...-18chs", 200, 1000),
    "bendr": ("InterpolatedBENDR", "braindecode/braindecode-bendr", 250, 1000),
    ...
}
```

The only genuinely per-model thing left is the **input contract** — but that is now
**data in a table** (target sfreq + window length), not code. Everything else
(load, montage-adapt, hook, pool) is shared.

## Why this matters for sharing (NeuroTechX Container Center)

- Ship **one** `braindecode`-based extractor + the `26.06` base image, and users get
  the entire braindecode zoo as frozen encoders — instead of N brittle adapters.
- The identity-free leaderboard / brain-age harness becomes model-agnostic: register
  a new EEG-FM by adding it upstream to braindecode (or a one-line `BD_SPEC` row),
  not by re-implementing a forward pass.
- License/hosting stays clean: `from_pretrained` fetches weights from the model's own
  HF repo at run time (thin, no re-hosting of gated checkpoints).

## Caveats / gotchas we hit

- **Use the real class, not the WeightWatcher wrapper.** Our
  `analyze_eegfm_weightwatcher.py` loads CBraMod/LUNA via `_state_dict_wrapper`
  (synthetic `nn.Linear`s whose `forward()` returns the input) — that is
  **analysis-only**; it cannot extract features. For forward extraction always
  instantiate the actual `braindecode.models.*` class.
- **Some checkpoints need help.** `CBraMod.from_pretrained` uses lazy modules and
  errors without `n_outputs`/a shape-materializing pass; `LUNA` is hosted at
  `PulpBio/LUNA` (the `braindecode/luna-pretrained` id 404s). These are per-model
  data fixes, not architecture work.
- **BENDR's Interpolated ctor** rejects some kwargs (`n_chans_pretrained`) — pass the
  minimal `chs_info/n_outputs/n_times/sfreq` and let it infer.
- **Not everything is in braindecode.** ZUNA is our custom-code HF model
  (`vector_quantize_pytorch` + `einx`); keep the hand `HFModelAdapter` for those.
  Where a model exists in **both** (REVE, LaBraM), prefer whichever you've already
  validated — the numbers should agree up to the pooling choice.

## One-line summary

braindecode 1.5 turned "write an adapter per EEG-FM" into "add a row to a table":
`from_pretrained` + a `final_layer` pre-hook + `Interpolated*` montage adaptation =
one frozen-encoder extractor for the whole zoo.
