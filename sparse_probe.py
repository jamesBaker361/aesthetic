# Per-latent 1D ridge logistic probes (Eq. (7) of "Rediscovering SAEs",
# arXiv:2511.17735) fit directly on top-k-sparse SAE codes.
#
# Same math as generate_clean_inference.fit_1d_ridge_logistic /
# per_latent_f1_at_threshold (same w0=0, b0=prevalence init, same Newton
# updates, same ridge on w only), but never materializes the dense
# (n_patches, n_dirs) matrix. Every patch has at most k nonzero latents, and
# every zero entry of latent j contributes to the gradient/Hessian only
# through expit(b_j), so those contributions collapse into per-latent counts.
# That keeps the fit cheap and exact for any number of patches.

import numpy as np
from scipy.special import expit


def _flatten_entries(idx: np.ndarray, val: np.ndarray):
    '''
    idx/val: (n_patches, k) top-k latent indices/values per patch. Returns
    (rows, cols, z) for the strictly positive entries only - a relu'd zero is
    identical to "not in the top k" for these probes.
    '''
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1).astype(np.int64)
    z = val.reshape(-1).astype(np.float64)
    keep = z > 0
    return rows[keep], cols[keep], z[keep]


def fit_sparse_1d_ridge_logistic(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int,
                                 ridge: float = 1e-8, n_newton_steps: int = 30):
    '''
    Returns w, b, loss (each (n_dirs,)) exactly like the dense
    fit_1d_ridge_logistic: loss is the unregularized mean training BCE.
    '''
    y = labels.astype(bool)
    n = len(y)
    n_pos = float(y.sum())
    rows, cols, z = _flatten_entries(idx, val)
    y_e = y[rows].astype(np.float64)

    nnz = np.bincount(cols, minlength=n_dirs).astype(np.float64)
    nnz_pos = np.bincount(cols, weights=y_e, minlength=n_dirs)
    n0 = n - nnz  # patches where latent j is inactive (z=0)
    n0_pos = n_pos - nnz_pos

    w = np.zeros(n_dirs, dtype=np.float64)
    b = np.full(n_dirs, n_pos / n, dtype=np.float64)

    for _ in range(n_newton_steps):
        p_e = expit(w[cols] * z + b[cols])
        resid = p_e - y_e
        s_e = p_e * (1.0 - p_e)
        p0 = expit(b)

        g_w = np.bincount(cols, weights=resid * z, minlength=n_dirs) + 2.0 * ridge * w
        g_b = np.bincount(cols, weights=resid, minlength=n_dirs) + (n0 * p0 - n0_pos)
        h_ww = np.bincount(cols, weights=s_e * z * z, minlength=n_dirs) + 2.0 * ridge
        h_wb = np.bincount(cols, weights=s_e * z, minlength=n_dirs)
        h_bb = np.bincount(cols, weights=s_e, minlength=n_dirs) + n0 * p0 * (1.0 - p0)

        det = h_ww * h_bb - h_wb * h_wb
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        w = w - (h_bb * g_w - h_wb * g_b) / det
        b = b - (h_ww * g_b - h_wb * g_w) / det

    eps = 1e-12
    p_e = np.clip(expit(w[cols] * z + b[cols]), eps, 1 - eps)
    p0 = np.clip(expit(b), eps, 1 - eps)
    nll_e = -(y_e * np.log(p_e) + (1 - y_e) * np.log(1 - p_e))
    loss = np.bincount(cols, weights=nll_e, minlength=n_dirs)
    loss += -(n0_pos * np.log(p0) + (n0 - n0_pos) * np.log(1 - p0))
    return w, b, loss / n


def sparse_per_latent_confusion(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, w: np.ndarray,
                                b: np.ndarray, threshold: float = 0.5):
    '''
    Per-latent tp/fp/fn of each fitted probe thresholded at `threshold`.
    '''
    n_dirs = len(w)
    y = labels.astype(bool)
    n = len(y)
    n_pos = float(y.sum())
    rows, cols, z = _flatten_entries(idx, val)
    y_e = y[rows]

    nnz = np.bincount(cols, minlength=n_dirs).astype(np.float64)
    nnz_pos = np.bincount(cols, weights=y_e.astype(np.float64), minlength=n_dirs)
    n0 = n - nnz
    n0_pos = n_pos - nnz_pos

    pred_e = expit(w[cols] * z + b[cols]) >= threshold
    pred0 = (expit(b) >= threshold).astype(np.float64)

    tp = np.bincount(cols, weights=(pred_e & y_e).astype(np.float64), minlength=n_dirs) + pred0 * n0_pos
    fp = np.bincount(cols, weights=(pred_e & ~y_e).astype(np.float64), minlength=n_dirs) + pred0 * (n0 - n0_pos)
    fn = n_pos - tp
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1


def baseline_bce(labels: np.ndarray) -> float:
    '''
    BCE of the best constant predictor (the prevalence) - the loss a probe
    that ignores its latent entirely would get. "Loss explained" below is
    measured against this.
    '''
    prev = float(np.clip(np.mean(labels.astype(np.float64)), 1e-12, 1 - 1e-12))
    return float(-(prev * np.log(prev) + (1 - prev) * np.log(1 - prev)))


def latent_activation_stats(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, latent: int) -> dict:
    '''
    Activation of one latent over positive/negative patches (zeros included).
    '''
    y = labels.astype(bool)
    act = np.where(idx == latent, val, 0.0).sum(axis=1).astype(np.float64)
    pos, neg = act[y], act[~y]
    return {
        "pos_mean": float(pos.mean()) if len(pos) else 0.0,
        "pos_std": float(pos.std()) if len(pos) else 0.0,
        "neg_mean": float(neg.mean()) if len(neg) else 0.0,
        "pos_active_frac": float((pos > 0).mean()) if len(pos) else 0.0,
        "neg_active_frac": float((neg > 0).mean()) if len(neg) else 0.0,
    }


def scaled_csr(idx: np.ndarray, val: np.ndarray, n_dirs: int):
    '''The top-k codes as a (n_patches, n_dirs) CSR matrix, each latent scaled to max 1; also the scales.'''
    from scipy.sparse import csr_matrix
    rows, cols, z = _flatten_entries(idx, val)
    scale = np.zeros(n_dirs)
    np.maximum.at(scale, cols, z)
    scale[scale == 0] = 1.0
    return csr_matrix((z / scale[cols], (rows, cols)), shape=(len(idx), n_dirs)), scale


def lasso_select(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int, k: int,
                 log_c_range=(-4.0, 2.0), n_search: int = 14) -> dict:
    '''
    Joint L1-penalised logistic regression over every latent at once (the
    sparse codes as a scipy CSR matrix, each latent scaled to max 1 so the
    penalty treats them alike; liblinear). The penalty C is bisected on a log
    scale for the strongest penalty that keeps >= k positive-weight latents;
    the k with the largest weights are returned. Unlike ranking per-latent
    probes, a latent that only repeats another one's information gets no weight.
    Returns {"idx": [...], "coef": [...], "C", "n_selected"} (fewer than k
    latents if even the weakest penalty doesn't give k).
    '''
    from sklearn.linear_model import LogisticRegression

    X, scale = scaled_csr(idx, val, n_dirs)
    y = labels.astype(int)

    import sklearn
    # sklearn >= 1.8 deprecates penalty= in favour of l1_ratio
    new_api = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    l1 = {"l1_ratio": 1.0} if new_api else {"penalty": "l1"}

    def positive(log_c):
        # intercept_scaling: liblinear penalises the intercept too - make that negligible
        m = LogisticRegression(C=10.0 ** log_c, solver="liblinear", intercept_scaling=100.0,
                               max_iter=500, tol=1e-4, **l1)
        m.fit(X, y)
        coef = m.coef_[0]
        return coef, int((coef > 0).sum())

    lo, hi = log_c_range
    best_c, (best_coef, n_hi) = hi, positive(hi)
    if n_hi >= k:
        for _ in range(n_search):
            mid = (lo + hi) / 2
            coef, n = positive(mid)
            if n >= k:
                hi, best_c, best_coef = mid, mid, coef
            else:
                lo = mid
    order = [int(j) for j in np.argsort(-best_coef) if best_coef[j] > 0][:k]
    return {"idx": order, "coef": [float(best_coef[j] / scale[j]) for j in order],
            "C": float(10.0 ** best_c), "n_selected": int((best_coef > 0).sum())}


def smallest_k(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_dirs: int,
               order: list, frac: float = 0.95, metric: str = "accuracy", test_frac: float = 0.2,
               seed: int = 0, C: float = 1.0) -> dict:
    '''
    Fewest leading latents of `order` (most important first) whose joint
    logistic regression classifies the patches at least frac x as well as one
    on EVERY latent. Both are L2 logistic regressions (liblinear, latents
    scaled to max 1, same C), fit on the patches of ~80% of the images
    (groups = image id per patch) and scored on the held-out rest. k is found
    by binary search over 1..len(order), which assumes the score rises with k
    (every evaluated k is kept in "curve" to check that).
    Returns {"k", "latents", "full", "target", "score", "curve", "metric"}.
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    score_fn = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
                "f1": f1_score}[metric]
    X, _ = scaled_csr(idx, val, n_dirs)
    X = X.tocsc()
    y = labels.astype(int)
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    if len(ids) >= 2:
        test_ids = rng.choice(ids, max(1, int(round(test_frac * len(ids)))), replace=False)
        test = np.isin(groups, test_ids)
    else:  # one image: hold out random patches instead
        test = rng.random(len(y)) < test_frac
    train = ~test

    def score(cols) -> float:
        if len(np.unique(y[train])) < 2:
            return float("nan")
        Xc = X[:, cols] if cols is not None else X
        m = LogisticRegression(C=C, solver="liblinear", max_iter=500)
        m.fit(Xc[train], y[train])
        return float(score_fn(y[test], m.predict(Xc[test])))

    full = score(None)
    target = frac * full
    curve = {}

    def at(k):
        if k not in curve:
            curve[k] = score(list(order[:k]))
        return curve[k]

    lo, hi = 1, len(order)
    if hi == 0:
        return {"k": 0, "latents": [], "full": full, "target": target, "score": float("nan"), "curve": {},
                "metric": metric}
    if not at(hi) >= target:
        lo = hi  # even every candidate misses the target: use them all
    while lo < hi:
        mid = (lo + hi) // 2
        if at(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return {"k": lo, "latents": [int(j) for j in order[:lo]], "full": full, "target": target,
            "score": at(lo), "curve": {str(k): v for k, v in sorted(curve.items())}, "metric": metric}


def select_bce_and_f1(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int,
                      ridge: float = 1e-8, n_newton_steps: int = 30, top_k: int = 1,
                      lasso: bool = False, auto: dict = None) -> dict:
    '''
    Fits every latent's probe once and returns both the lowest-BCE latent
    and the highest-F1 latent, each with its BCE, loss explained, F1,
    precision, recall and activation stats. With top_k > 1 also returns
    "bce_top" / "f1_top": the best top_k latents per rule (the top-1 above,
    then the next best with a positive probe weight, i.e. active ON the mask).
    With lasso, also "lasso_by_k": {str(top_k): {"top": [describe(j) + its
    lasso_coef], "C", "n_selected"}} from lasso_select - keyed by k, since
    the penalty that keeps 3 latents is not the one that keeps 10.
    With auto = {"rules", "groups", "frac", "metric", "max_k", "seed"}, also
    "auto": {rule: smallest_k(...) + "top": [describe(j)]}, where each rule
    orders the latents by importance - "bce": per-latent BCE, "f1": per-latent
    F1 (both only positive-weight latents), "lasso": lasso_select's weights -
    and the smallest k reaching frac of the all-latent classifier is kept.
    '''
    w, b, loss = fit_sparse_1d_ridge_logistic(idx, val, labels, n_dirs, ridge, n_newton_steps)
    tp, fp, fn = sparse_per_latent_confusion(idx, val, labels, w, b)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    base = baseline_bce(labels)
    loss_explained = 1.0 - loss / base

    def describe(j: int) -> dict:
        d = {
            "idx": int(j),
            "bce": float(loss[j]),
            "loss_explained": float(loss_explained[j]),
            "f1": float(f1[j]),
            "precision": float(precision[j]),
            "recall": float(recall[j]),
            "w": float(w[j]),
            "b": float(b[j]),
            "bce_rank": int((loss < loss[j]).sum()),
            "f1_rank": int((f1 > f1[j]).sum()),
        }
        d.update(latent_activation_stats(idx, val, labels, j))
        return d

    bce_idx = int(np.argmin(loss))
    f1_idx = int(np.argmax(f1))
    out = {
        "n_patches": int(len(labels)),
        "n_pos": int(labels.sum()),
        "baseline_bce": base,
        "bce": describe(bce_idx),
        "f1": describe(f1_idx),
        "same_feature": bce_idx == f1_idx,
    }
    if top_k > 1:
        def top(order, first):
            rest = [int(j) for j in order if j != first and w[j] > 0 and (loss[j] < base or f1[j] > 0)]
            return [first] + rest[:top_k - 1]
        out["top_k"] = top_k
        out["bce_top"] = [describe(j) for j in top(np.argsort(loss), bce_idx)]
        out["f1_top"] = [describe(j) for j in top(np.argsort(-f1), f1_idx)]
    if auto:
        out["auto"] = {}
        for rule in auto["rules"]:
            if rule == "bce":
                order = [int(j) for j in np.argsort(loss) if w[j] > 0 and loss[j] < base]
            elif rule == "f1":
                order = [int(j) for j in np.argsort(-f1) if w[j] > 0 and f1[j] > 0]
            else:
                order = lasso_select(idx, val, labels, n_dirs, auto["max_k"])["idx"]
            res = smallest_k(idx, val, labels, auto["groups"], n_dirs, order[:auto["max_k"]],
                             frac=auto["frac"], metric=auto["metric"], seed=auto["seed"])
            res["top"] = [describe(j) for j in res["latents"]]
            out["auto"][rule] = res
    if lasso:
        sel = lasso_select(idx, val, labels, n_dirs, top_k)
        out["lasso_by_k"] = {str(top_k): {
            "top": [{**describe(j), "lasso_coef": c} for j, c in zip(sel["idx"], sel["coef"])],
            "C": sel["C"], "n_selected": sel["n_selected"]}}
    return out
