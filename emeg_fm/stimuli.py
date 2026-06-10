"""Image stimulus sets, CLIP gallery, and EEG-ExPy presentation schedules.

The decoder retrieves against a *gallery* of CLIP image embeddings; the
presentation script shows images and tags each onset with an integer marker
*code*. This module is the single source of truth tying images ↔ codes ↔
gallery rows so the live runner, the decoder, and the EEG-ExPy experiment all
agree.

    ImageStimulusSet.from_dir(dir)          # discover images, assign codes
        .build_schedule(n_repeats, seed)    # randomized (code, path) order
        .compute_gallery(clip_model, dev)   # (n, d_clip) embeddings + ids
        .save_gallery(path) / load_gallery  # cache so calibration is instant

Marker codes are 1-indexed integers (LSL marker 0 is reserved for non-stimulus
events). torch/transformers/PIL are imported lazily inside ``compute_gallery``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ImageStimulusSet:
    """A fixed set of images with stable integer marker codes.

    ``codes[i]`` (1-indexed) ↔ ``paths[i]``. ``gallery``/``gallery_ids`` are
    populated by :meth:`compute_gallery` or :meth:`load_gallery`.
    """

    paths: list
    codes: np.ndarray
    gallery: np.ndarray | None = None
    gallery_ids: np.ndarray | None = None
    clip_model: str | None = None
    _basenames: list = field(default_factory=list)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dir(cls, image_dir, max_images=None, seed=0, recursive=True):
        """Discover images under ``image_dir`` and assign deterministic codes.

        Images are sorted by basename for a stable code assignment across runs.
        ``max_images`` keeps a reproducible random subset (seed-controlled).
        """
        found = []
        if recursive:
            for root, _d, files in os.walk(image_dir):
                for f in files:
                    if f.lower().endswith(_IMG_EXT):
                        found.append(os.path.join(root, f))
        else:
            for f in os.listdir(image_dir):
                if f.lower().endswith(_IMG_EXT):
                    found.append(os.path.join(image_dir, f))
        if not found:
            raise FileNotFoundError(f"no images under {image_dir}")
        found.sort(key=os.path.basename)
        if max_images is not None and len(found) > max_images:
            rng = np.random.default_rng(seed)
            sel = np.sort(rng.choice(len(found), size=max_images, replace=False))
            found = [found[i] for i in sel]
        codes = np.arange(1, len(found) + 1, dtype=int)        # 1-indexed
        return cls(paths=found, codes=codes,
                   _basenames=[os.path.basename(p) for p in found])

    # -- presentation schedule ----------------------------------------------

    def build_schedule(self, n_repeats=4, seed=0, shuffle=True):
        """Randomized presentation order → list of ``(code, path)``.

        Each image appears ``n_repeats`` times; repeats let the decoder
        trial-average for SNR (the offline smoke's key move). With a closed
        set of ~50–80 images × 4–5 repeats you stay inside a ~10-minute budget
        at ~1.5 s/trial.
        """
        order = np.repeat(np.arange(len(self.paths)), n_repeats)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        return [(int(self.codes[i]), self.paths[i]) for i in order]

    def save_schedule(self, path, n_repeats=4, seed=0):
        sched = self.build_schedule(n_repeats=n_repeats, seed=seed)
        with open(path, "w") as f:
            json.dump({"n_repeats": n_repeats, "seed": seed,
                       "trials": [[c, p] for c, p in sched]}, f, indent=2)
        return path

    @property
    def code_to_path(self) -> dict:
        return {int(c): p for c, p in zip(self.codes, self.paths)}

    @property
    def code_to_basename(self) -> dict:
        return {int(c): os.path.basename(p) for c, p in zip(self.codes, self.paths)}

    # -- CLIP gallery --------------------------------------------------------

    def compute_gallery(self, clip_model="openai/clip-vit-base-patch32",
                        device=None, batch_size=64):
        """Embed every image with CLIP → fills ``gallery`` / ``gallery_ids``."""
        self.gallery = clip_image_embeddings(self.paths, clip_model, device,
                                             batch_size=batch_size)
        self.gallery_ids = self.codes.copy()
        self.clip_model = clip_model
        return self.gallery

    def save_gallery(self, path):
        if self.gallery is None:
            raise RuntimeError("compute_gallery first")
        np.savez(path, gallery=self.gallery, gallery_ids=self.gallery_ids,
                 codes=self.codes, basenames=np.array(self._basenames),
                 paths=np.array(self.paths), clip_model=str(self.clip_model))
        return path

    @classmethod
    def load_gallery(cls, path):
        z = np.load(path, allow_pickle=True)
        s = cls(paths=list(z["paths"]), codes=z["codes"],
                gallery=z["gallery"], gallery_ids=z["gallery_ids"],
                clip_model=str(z["clip_model"]),
                _basenames=list(z["basenames"]))
        return s


def clip_image_embeddings(local_paths, model_id="openai/clip-vit-base-patch32",
                          device=None, batch_size=64) -> np.ndarray:
    """CLIP visual-projection embeddings for image paths → ``(n, d_clip)``.

    Mirrors the offline smoke's image-embed path (vision_model → visual_projection
    is stable across transformers versions, unlike get_image_features).
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    feats = []
    for i in range(0, len(local_paths), batch_size):
        chunk = local_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            vis = model.vision_model(pixel_values=inp["pixel_values"])
            emb = model.visual_projection(vis.pooler_output)
        feats.append(emb.float().cpu().numpy())
    return np.concatenate(feats, axis=0)
