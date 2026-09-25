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


def select_bce_and_f1(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int,
                      ridge: float = 1e-8, n_newton_steps: int = 30) -> dict:
    '''
    Fits every latent's probe once and returns both the lowest-BCE latent
    and the highest-F1 latent, each with its BCE, loss explained, F1,
    precision, recall and activation stats.
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
    return {
        "n_patches": int(len(labels)),
        "n_pos": int(labels.sum()),
        "baseline_bce": base,
        "bce": describe(bce_idx),
        "f1": describe(f1_idx),
        "same_feature": bce_idx == f1_idx,
    }
