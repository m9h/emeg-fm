"""Luck-Lab ERPLAB-parity linear-SVM decoding probe.

Reimplements the decision protocol of ERPLAB12 ``pop_decoding`` /
``pop_decoding_regularization`` (Steve Luck lab, ERP CORE decoding scripts on
OSF ``un6pq``) so a frozen EEG-FM (REVE) embedding can be scored under the
*same* protocol the ERPLAB people use on raw scalp voltage. This is the
parity counterpart to the canonical FMScope LP (LogisticRegression, C=1,
recording-pooled 5-fold) in :mod:`fmscope.training.lp`.

ERPLAB DECODE protocol (as configured in
``m_1_decoding_ERPCORE_regularization.m``)::

    Method='SVM', classcoding='OneVsAll', crossfold=3, nIter=100,
    Decode_Every_Npoint=5, EqualizeTrials='classes',
    Gamma_Value=[1e-3 1e-2 1e-1 1 10 100 1000]

The decode unit is one participant. Within a participant, each iteration
randomly partitions every class's trials into ``crossfold`` blocks, averages
the trials *within* each block into a **pseudo-ERP**, then runs leave-one-
block-out linear-SVM over the pseudo-ERPs. Accuracy is averaged over the
``crossfold`` held-out blocks and ``n_iter`` resamples. EqualizeTrials
subsamples both classes to the minority count each iteration.

Two surfaces:

- :func:`luck_svm_decode` / :func:`luck_svm_grid` operate on a trial feature
  matrix ``(n_trials, n_features)`` — used directly for REVE embeddings (one
  embedding per trial, no time axis).
- :func:`erplab_decode_scalp` wraps the primitive for raw scalp voltage
  ``(n_trials, n_chan, n_times)``: it decodes at each timepoint (every
  ``decode_every`` samples) using the channel vector as features and reports
  the mean accuracy inside a component measurement window — the classical
  ERPLAB DECODE baseline.

Gamma→C mapping: ERPLAB's ``Gamma_Value`` regularization grid is swept and
mapped onto the linear-SVM box constraint ``C`` (LIBSVM/sklearn parameter),
documented per :func:`luck_svm_grid`. The grid values are kept verbatim so the
sweep range matches the published script.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ERPLAB Gamma_Value grid from m_1_decoding_ERPCORE_regularization.m, used here
# as the linear-SVM box-constraint (C) sweep.
DEFAULT_GAMMA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def _pseudo_erps(X_cls, n_blocks, rng):
    """Partition one class's trials into ``n_blocks`` blocks, average each.

    Returns ``(n_blocks, n_features)`` pseudo-ERPs. Trials are shuffled then
    split as evenly as possible (``np.array_split``), matching ERPLAB's
    random block assignment + within-block averaging.
    """
    n = len(X_cls)
    order = rng.permutation(n)
    blocks = np.array_split(order, n_blocks)
    return np.stack([X_cls[b].mean(0) for b in blocks], axis=0)


def _svm_lobo(per_class, crossfold, classes, C):
    """One leave-one-block-out SVM pass over per-class pseudo-ERP blocks.

    ``per_class[c]`` is ``(crossfold, n_features)``. Yields one accuracy per
    held-out block.
    """
    for held in range(crossfold):
        train_X, train_y, test_X, test_y = [], [], [], []
        for ci, c in enumerate(classes):
            erps = per_class[c]
            mask = np.ones(crossfold, dtype=bool)
            mask[held] = False
            train_X.append(erps[mask])
            train_y.append(np.full(mask.sum(), ci))
            test_X.append(erps[~mask])
            test_y.append(np.full((~mask).sum(), ci))
        Xtr = np.concatenate(train_X, 0)
        ytr = np.concatenate(train_y, 0)
        Xte = np.concatenate(test_X, 0)
        yte = np.concatenate(test_y, 0)
        sc = StandardScaler().fit(Xtr)
        clf = LinearSVC(C=C, dual="auto", max_iter=5000)
        clf.fit(sc.transform(Xtr), ytr)
        yield accuracy_score(yte, clf.predict(sc.transform(Xte)))


def luck_svm_decode(X, y, *, groups=None, crossfold=3, n_iter=100, C=1.0,
                    equalize=True, seed=0):
    """Decode binary ``y`` from ``X`` under the ERPLAB pseudo-ERP SVM protocol.

    Parameters
    ----------
    X : (n_trials, n_features) array
        Per-trial feature vectors (REVE embedding, or channel voltages at one
        timepoint for the scalp baseline).
    y : (n_trials,) array
        Binary class labels.
    groups : (n_trials,) array, optional
        When given, pseudo-ERP blocks are formed by partitioning *groups*
        (e.g. subjects) into ``crossfold`` leave-one-group-block-out folds and
        averaging each class's trials *within* a block's groups. This is the
        cross-subject decode used for the identity-free (LEACE) contrast —
        ERPLAB's native protocol is within-participant (``groups=None``).
    crossfold : int, default 3
        Number of pseudo-ERP blocks per class (== ERPLAB ``nCrossblocks``).
    n_iter : int, default 100
        Random block-assignment resamples (== ERPLAB ``nIter``).
    C : float, default 1.0
        Linear-SVM box constraint (mapped from ERPLAB ``Gamma_Value``).
    equalize : bool, default True
        Subsample both classes to the minority count each iteration
        (EqualizeTrials='classes').
    seed : int
        RNG seed.

    Returns
    -------
    float
        Mean leave-one-block-out accuracy over folds and iterations. ``NaN``
        when the data cannot fill ``crossfold`` blocks with both classes
        (undecodable rather than an error).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        return float("nan")
    rng = np.random.default_rng(seed)

    if groups is not None:
        return _decode_grouped(X, y, np.asarray(groups), classes,
                               crossfold, n_iter, C, equalize, rng)

    idx = {c: np.where(y == c)[0] for c in classes}
    min_n = min(len(idx[c]) for c in classes)
    if min_n < crossfold:
        return float("nan")

    accs: list[float] = []
    for _ in range(n_iter):
        # EqualizeTrials: subsample each class to the minority count.
        per_class = {}
        for c in classes:
            ix = idx[c]
            if equalize and len(ix) > min_n:
                ix = rng.choice(ix, min_n, replace=False)
            per_class[c] = _pseudo_erps(X[ix], crossfold, rng)
        accs.extend(_svm_lobo(per_class, crossfold, classes, C))

    return float(np.mean(accs)) if accs else float("nan")


def _decode_grouped(X, y, groups, classes, crossfold, n_iter, C, equalize, rng):
    """Cross-subject pseudo-ERP decode: groups → blocks, average within block."""
    uniq = np.unique(groups)
    if len(uniq) < crossfold:
        return float("nan")
    accs: list[float] = []
    for _ in range(n_iter):
        gblocks = np.array_split(rng.permutation(uniq), crossfold)
        per_class = {c: [] for c in classes}
        valid = True
        for gb in gblocks:
            block_mask = np.isin(groups, gb)
            # Equalize classes within the block before averaging into a
            # pseudo-ERP, so block SNR is class-balanced.
            block_idx = {c: np.where(block_mask & (y == c))[0] for c in classes}
            if any(len(block_idx[c]) == 0 for c in classes):
                valid = False
                break
            m = min(len(block_idx[c]) for c in classes)
            for c in classes:
                ix = block_idx[c]
                if equalize and len(ix) > m:
                    ix = rng.choice(ix, m, replace=False)
                per_class[c].append(X[ix].mean(0))
        if not valid:
            continue
        per_class = {c: np.stack(v, 0) for c, v in per_class.items()}
        accs.extend(_svm_lobo(per_class, crossfold, classes, C))
    return float(np.mean(accs)) if accs else float("nan")


def luck_svm_grid(X, y, *, gamma_grid=DEFAULT_GAMMA_GRID, crossfold=3,
                  n_iter=100, equalize=True, seed=0):
    """Sweep the ERPLAB ``Gamma_Value`` grid as the linear-SVM ``C``.

    Returns ``{"per_C": {C: acc}, "best_C": C*, "best_acc": acc*}``. ``best_C``
    is the grid value maximising mean pseudo-ERP decode accuracy (ties broken
    toward the first/smallest C, matching ``max`` over an ordered dict).
    """
    per_C = {
        c: luck_svm_decode(X, y, crossfold=crossfold, n_iter=n_iter,
                           C=c, equalize=equalize, seed=seed)
        for c in gamma_grid
    }
    finite = {c: v for c, v in per_C.items() if np.isfinite(v)}
    if not finite:
        return {"per_C": per_C, "best_C": float("nan"), "best_acc": float("nan")}
    best_C = max(finite, key=lambda c: finite[c])
    return {"per_C": per_C, "best_C": best_C, "best_acc": finite[best_C]}


def erplab_decode_scalp(epochs_data, y, times, *, window, decode_every=5,
                        crossfold=3, n_iter=100, C=1.0, equalize=True, seed=0):
    """Time-resolved ERPLAB DECODE baseline on raw scalp voltage.

    Decodes at each timepoint (every ``decode_every`` samples) using the
    per-channel voltage vector as features, then summarises the mean accuracy
    inside the component measurement ``window``.

    Parameters
    ----------
    epochs_data : (n_trials, n_chan, n_times) array
        Trial-by-channel-by-time scalp voltages.
    y : (n_trials,) array
        Binary class labels.
    times : (n_times,) array
        Epoch time axis (seconds).
    window : (lo, hi) tuple
        Measurement window in seconds; ``window_acc`` averages decode accuracy
        over decoded samples inside ``[lo, hi]``.
    decode_every : int, default 5
        Decode every Nth sample (== ERPLAB ``Decode_Every_Npoint``).

    Returns
    -------
    dict
        ``curve`` (acc per decoded sample), ``curve_times`` (their times),
        ``window_acc`` (mean acc inside the window).
    """
    epochs_data = np.asarray(epochs_data, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    sample_idx = np.arange(0, epochs_data.shape[2], decode_every)
    curve = np.array([
        luck_svm_decode(epochs_data[:, :, t], y, crossfold=crossfold,
                        n_iter=n_iter, C=C, equalize=equalize, seed=seed)
        for t in sample_idx
    ])
    curve_times = times[sample_idx]
    lo, hi = window
    in_win = (curve_times >= lo) & (curve_times <= hi)
    window_acc = (float(np.nanmean(curve[in_win]))
                  if in_win.any() and np.isfinite(curve[in_win]).any()
                  else float("nan"))
    return {"curve": curve, "curve_times": curve_times, "window_acc": window_acc}
