"""Minimal T3A (Test-Time Classifier Adjustment, Iwasawa & Matsuo NeurIPS 2021)
for our frozen-embedding + linear-probe pipeline -- pure numpy, no PyTorch
needed, since T3A is fully optimization-free (no backprop into any backbone).

Ported from the mechanics in github.com/leegabriel/NeuroAdapt-Bench
(methods/t3a/t3a.py, itself adapted from the official DomainBed T3A): warm-start
per-class "supports" from the fitted classifier's weight vectors, then online
per-test-sample: predict via normalized-support-weighted nearest-centroid,
append the sample (with its own pseudo-label + prediction entropy) to the
support set, and keep only the lowest-entropy top-`filter_k` supports per
(pseudo-)class. Causal: prediction for sample i only ever uses supports built
from samples 0..i-1 plus the warm start (matches the upstream per-batch online
update, here at batch-size 1 for a fully sequential single-subject episode).

Why we built this: NeuroAdapt-Bench studies T3A as a TTA method for EEG-FMs
under distribution shift. Our identity-trap program's open question: T3A's
online support-set literally incorporates the held-out TEST SUBJECT's own
features -- structurally the same mechanism that could leak subject identity
into the classifier. If a T3A accuracy gain SHRINKS after LEACE identity
erasure (same frozen embeddings, same fitted probe, only the erasure differs),
that's evidence some of T3A's "adaptation" was subject-identity exploitation,
not genuine task-signal adaptation. Only meaningful for LogisticRegression
probes (`model=="reve"` in this codebase) -- our classical path uses a GBM,
which has no linear weight vectors to warm-start supports from (same scoping
constraint as upstream, which requires a torch.nn.Linear classifier head).
"""
from __future__ import annotations

import numpy as np


def _softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_entropy(logits):
    p = _softmax(logits)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def _binary_to_virtual_2class(coef, intercept):
    """sklearn binary LogisticRegression stores one weight row; T3A needs a
    per-class weight vector for each of the 2 classes to seed supports --
    same construction as upstream's `t3a_compatible_classifier`."""
    w = coef[0]
    b = intercept[0] if intercept is not None else 0.0
    W2 = np.stack([-0.5 * w, 0.5 * w])
    b2 = np.array([-0.5 * b, 0.5 * b])
    return W2, b2


def t3a_adapt_sequence(clf, X_test, filter_k=20):
    """clf: fitted sklearn LogisticRegression. X_test: (n, d) already scaled
    (and optionally identity-erased) frozen features, in the recording's own
    temporal order. Returns yhat (n,) -- T3A-adapted predictions, causal."""
    classes = clf.classes_
    n_classes = len(classes)
    if clf.coef_.shape[0] == 1 and n_classes == 2:
        W, b = _binary_to_virtual_2class(clf.coef_, clf.intercept_)
    else:
        W = clf.coef_
        b = clf.intercept_ if clf.intercept_ is not None else np.zeros(n_classes)

    # Each class's own weight vector IS its warm-start prototype by
    # construction -- assign identity labels directly rather than deriving
    # via argmax(supports @ W.T). Deriving via argmax (as the upstream
    # DomainBed/NeuroAdapt-Bench code effectively does via `classifier(
    # warmup_supports)`) is only self-consistent when class weight vectors
    # have comparable norms; for a plain sklearn multinomial LogisticRegression
    # fit (unequal L2 pull per class), unequal weight-vector norms can make
    # one class's self-dot-product dominate every row, mislabeling ALL warm
    # supports as that one class (verified: this happened on a real toy case
    # here before the fix -- caught by test_t3a_matches_baseline_accuracy_*).
    supports = W.copy()
    warm_logits = supports @ W.T + b
    labels = np.eye(n_classes)
    ent = _softmax_entropy(warm_logits)

    yhat = np.zeros(len(X_test), dtype=classes.dtype)
    for i in range(len(X_test)):
        x = X_test[i]
        norm_sup = supports / (np.linalg.norm(supports, axis=1, keepdims=True) + 1e-12)
        weights = norm_sup.T @ labels  # (d, C)
        weights_n = weights / (np.linalg.norm(weights, axis=0, keepdims=True) + 1e-12)
        logits_i = x @ weights_n
        pred_idx = int(np.argmax(logits_i))
        yhat[i] = classes[pred_idx]

        pseudo = np.eye(n_classes)[pred_idx]
        ent_i = _softmax_entropy(logits_i[None, :])[0]
        supports = np.vstack([supports, x[None, :]])
        labels = np.vstack([labels, pseudo[None, :]])
        ent = np.append(ent, ent_i)

        if filter_k != -1:
            pred_classes = labels.argmax(axis=1)
            keep = []
            for c in range(n_classes):
                idx = np.where(pred_classes == c)[0]
                if len(idx) == 0:
                    continue
                order = idx[np.argsort(ent[idx])]
                keep.append(order[:filter_k])
            keep = np.concatenate(keep) if keep else np.array([], dtype=int)
            supports, labels, ent = supports[keep], labels[keep], ent[keep]
    return yhat
