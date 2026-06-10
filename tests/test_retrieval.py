"""Tests for emeg_fm.alljoined.topk_retrieval — cosine top-k image retrieval.

The Alljoined EEG→image smoke needs a retrieval metric: given a predicted
embedding per query (decoded from EEG) and a gallery of true image
embeddings, rank the gallery by cosine similarity to each prediction and
ask whether the correct image lands in the top-k.
"""

import numpy as np
import pytest

from emeg_fm.alljoined import topk_retrieval


def test_perfect_retrieval_top1_is_one():
    rng = np.random.RandomState(0)
    gallery = rng.randn(50, 16).astype(np.float32)
    # Prediction == the true embedding for each query (identity matching).
    out = topk_retrieval(gallery.copy(), gallery, ks=(1, 5))
    assert out["top1"] == pytest.approx(1.0)
    assert out["top5"] == pytest.approx(1.0)
    # Best possible: correct item is rank 1 for every query.
    assert np.all(np.asarray(out["ranks"]) == 1)


def test_chance_level_matches_inverse_gallery_size():
    rng = np.random.RandomState(1)
    n = 200
    gallery = rng.randn(n, 32).astype(np.float32)
    pred = rng.randn(n, 32).astype(np.float32)  # unrelated → chance
    out = topk_retrieval(pred, gallery, ks=(1, 5))
    assert out["chance_top1"] == pytest.approx(1.0 / n)
    # Empirically near chance for top1 (loose band) and clearly below 1.
    assert out["top1"] < 0.1
    # Median rank of a random predictor ≈ n/2.
    assert n * 0.25 < out["median_rank"] < n * 0.75


def test_topk_is_monotonic_nondecreasing():
    rng = np.random.RandomState(2)
    gallery = rng.randn(80, 24).astype(np.float32)
    # Noisy version of the truth so accuracy is between chance and perfect.
    pred = gallery + 0.8 * rng.randn(*gallery.shape).astype(np.float32)
    out = topk_retrieval(pred, gallery, ks=(1, 5, 10))
    assert out["top1"] <= out["top5"] <= out["top10"]
    assert 0.0 < out["top1"] < 1.0  # genuinely intermediate


def test_explicit_labels_permuted_gallery():
    rng = np.random.RandomState(3)
    n, d = 40, 12
    truth = rng.randn(n, d).astype(np.float32)
    perm = rng.permutation(n)
    gallery = truth[perm]               # gallery row j holds truth[perm[j]]
    labels = np.argsort(perm)           # query i's correct gallery row
    out = topk_retrieval(truth, gallery, ks=(1,), labels=labels)
    assert out["top1"] == pytest.approx(1.0)


def test_contract_keys_and_shapes():
    rng = np.random.RandomState(4)
    gallery = rng.randn(30, 8).astype(np.float32)
    pred = rng.randn(30, 8).astype(np.float32)
    out = topk_retrieval(pred, gallery, ks=(1, 5))
    for key in ("top1", "top5", "median_rank", "ranks", "chance_top1"):
        assert key in out
    assert np.asarray(out["ranks"]).shape == (30,)
    # Ranks are 1-indexed within [1, n_gallery].
    ranks = np.asarray(out["ranks"])
    assert ranks.min() >= 1 and ranks.max() <= 30


def test_mismatched_identity_shapes_raises():
    rng = np.random.RandomState(5)
    pred = rng.randn(10, 8).astype(np.float32)
    gallery = rng.randn(12, 8).astype(np.float32)
    # No labels + non-square → ambiguous identity matching → error.
    with pytest.raises(ValueError):
        topk_retrieval(pred, gallery, ks=(1,))
