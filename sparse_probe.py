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


def image_mean_codes(idx_all, val_all, owner, n_images: int, n_dirs: int, patch_mask=None) -> np.ndarray:
    '''
    (n_images, n_dirs): each discovery image's SAE code averaged over its patches (SAeUron's sae_out.mean(1)).
    With patch_mask (one bool per patch), only over the masked patches; an image with none is all NaN.
    '''
    keep = np.ones(len(owner), dtype=bool) if patch_mask is None else np.asarray(patch_mask, dtype=bool)
    sums = np.zeros((n_images, n_dirs), dtype=np.float64)
    rows = np.repeat(owner[keep], idx_all.shape[1])
    np.add.at(sums, (rows, idx_all[keep].reshape(-1)), np.maximum(val_all[keep].reshape(-1), 0.0))
    counts = np.bincount(owner[keep], minlength=n_images)
    means = sums / np.maximum(counts, 1)[:, None]
    if patch_mask is not None:
        means[counts == 0] = np.nan
    return means


def scaled_csr(idx: np.ndarray, val: np.ndarray, n_dirs: int):
    '''The top-k codes as a (n_patches, n_dirs) CSR matrix, each latent scaled to max 1; also the scales.'''
    from scipy.sparse import csr_matrix
    rows, cols, z = _flatten_entries(idx, val)
    scale = np.zeros(n_dirs)
    np.maximum.at(scale, cols, z)
    scale[scale == 0] = 1.0
    return csr_matrix((z / scale[cols], (rows, cols)), shape=(len(idx), n_dirs)), scale


def penalty_kwargs(l1_ratio: float) -> dict:
    '''LogisticRegression's L1 (1.0) / elastic-net (0 < l1_ratio < 1) penalty: sklearn >= 1.8 deprecates penalty=
    in favour of l1_ratio alone.'''
    import sklearn
    new_api = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    if new_api:
        return {"l1_ratio": l1_ratio}
    return {"penalty": "l1"} if l1_ratio == 1.0 else {"penalty": "elasticnet", "l1_ratio": l1_ratio}


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
    l1 = penalty_kwargs(1.0)

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


def normalized_csr(idx: np.ndarray, val: np.ndarray, n_dirs: int):
    '''
    The top-k codes as a (n_patches, n_dirs) CSR matrix, each latent divided by 10 x its std over the patches (zeros
    included, std floored at 1e-3) - the official sparse-probing code's --normalize_activations
    (experiments/inner_loops.split_and_preprocess), which every run of the paper uses; also the divisors.
    '''
    from scipy.sparse import csr_matrix
    rows, cols, z = _flatten_entries(idx, val)
    n = len(idx)
    mean = np.bincount(cols, weights=z, minlength=n_dirs) / n
    var = np.bincount(cols, weights=z * z, minlength=n_dirs) / n - mean ** 2
    scale = 10.0 * np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-3)
    return csr_matrix((z / scale[cols], (rows, cols)), shape=(n, n_dirs)), scale


def mean_difference_ranking(X, labels: np.ndarray) -> np.ndarray:
    '''Every latent by |mean over positives - mean over negatives| of X, largest first (the official code's
    'mean_dif' heuristic, probes.get_heuristic_neuron_ranking).'''
    y = labels.astype(bool)
    diff = np.abs(np.asarray(X[y].mean(axis=0)).ravel() - np.asarray(X[~y].mean(axis=0)).ravel())
    return np.argsort(-diff, kind="stable")


def k_schedule(k_max: int, d: int) -> list:
    '''The official code's k list (inner_loops.make_k_list): 1..8, 10, 12, then 16, 32, ... and d, up to k_max.'''
    exp_ks = list(16 * 2 ** np.arange(int(np.log2(d)) - 3))
    if not exp_ks or exp_ks[-1] != d:
        exp_ks.append(d)
    return [int(k) for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] + exp_ks if k <= k_max]


def balanced_lr(C: float, l1_ratio: float = None, max_iter: int = 200):
    '''The official code's probes: balanced class weights, saga; L2, or elastic net with l1_ratio.'''
    from sklearn.linear_model import LogisticRegression
    pen = penalty_kwargs(l1_ratio) if l1_ratio is not None else {}
    return LogisticRegression(C=C, solver="saga", class_weight="balanced", max_iter=max_iter, **pen)


# Gurnee et al., "Finding Neurons in a Haystack: Case Studies with Sparse Probing" (arXiv:2305.01610), Sec. 3.2 and
# App. B.3-B.4, as their official code (github.com/wesg52/sparse-probing-paper, experiments/inner_loops.py and
# probes.py): two k-sparse probes on the codes normalized by normalized_csr, starting from the latents with the
# largest |class mean difference|. Each counts its own FLOPs - OSP stops at a timeout, so a re-run would not
# reproduce its cost - with search_flops.py's upper-bound constants: saga 4 x nnz per epoch, libsvm's SMO two kernel
# columns and the gradient update (4 x nnz + 4 x n) per iteration, the normalization + mean difference 4 x nnz.

def adaptive_threshold_select(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int, k_max: int = 64,
                              C: float = -1.0, l1_ratio: float = -1.0, max_iter: int = 1000) -> dict:
    '''
    Adaptive thresholding (B.3; the official telescopic_sparsity_sweep): for k in k_schedule(k_max) from the largest
    down, an elastic-net logistic regression (balanced, saga) on the current k latents - the top k_max by |mean
    difference| at first, then the k largest |coefficients| of the previous probe. C = k / 20 and l1_ratio =
    clip(e^(-k/20), 0.1, 0.9) per k unless given (> 0), as the official sweep (the paper's runs: k from 256; here
    k_max). The sets are nested, so they are one ranking: each set's latents beyond the next smaller set's are
    ordered by its own probe, so order[:k] is the top k of the probe fit at the smallest scheduled k' >= k (exactly
    the scheduled sets at k' itself).
    Returns {"order", "coef": {str(k): the probe's coefficients over order[:k], per scheduled k}, "flops"}.
    '''
    X, scale = normalized_csr(idx, val, n_dirs)
    flops = 4 * X.nnz
    X = X.tocsc()
    y = labels.astype(int)
    ks = k_schedule(k_max, n_dirs)[::-1]
    support = [int(j) for j in mean_difference_ranking(X, labels)[:ks[0]]] if ks else []
    sets = {}
    for k in ks:
        support = support[:k]
        Xc = X[:, support]
        m = balanced_lr(C if C > 0 else k / 20, l1_ratio if l1_ratio > 0 else float(np.clip(np.exp(-k / 20), 0.1, 0.9)),
                        max_iter=max_iter)
        m.fit(Xc, y)
        flops += int(np.max(m.n_iter_)) * 4 * Xc.nnz
        sets[k] = dict(zip(support, m.coef_[0]))
        support = [support[i] for i in np.argsort(-np.abs(m.coef_[0]), kind="stable")]
    order = []
    for k in sorted(sets):  # S_1 first, then the latents each larger set adds, by that set's probe
        order += sorted((j for j in sets[k] if j not in order), key=lambda j: -abs(sets[k][j]))
    return {"order": order, "coef": {str(k): [float(sets[k][j] / scale[j]) for j in order[:k]] for k in sets},
            "flops": float(flops)}


class OptimalSparseProbe:
    '''
    Optimal sparse probing (B.3; the official optimal_sparse_probing + probes.sparse_classification_oa): the
    k-sparse hinge-loss SVM
        min_{w, b} sum_i c_i max(0, 1 - y_i (w.x_i + b)) + ||w||^2 / (2 gamma)    s.t. ||w||_0 = k
    solved to provable optimality with Bertsimas, Pauphilet & Van Parys's outer approximation (arXiv:1710.01352,
    Sec. 2.2-2.3, Alg. 1), on the n_filter latents with the largest |mean difference| (official default 50) of the
    normalized codes, c_i the balanced class weights. With s the support, c(s) = max_alpha f(alpha, s) is convex in
    s (Thm. 2) and f(alpha, s) = sum_i beta_i - gamma / 2 sum_j s_j (X_j.alpha)^2 is linear in s for any
    dual-feasible alpha (beta_i = -y_i alpha_i in [0, c_i], sum_i alpha_i = 0), so each SVM solved on a support
    gives a cut eta >= f(alpha*, s) valid for every s. The official code solves that dual with Gurobi and adds the
    cuts lazily in one branch and bound; here libsvm solves the same dual and a MILP over the cuts (HiGHS)
    proposes the next support, until its bound meets the best SVM objective or time_limit (official: 60 s) runs out
    - then the best support so far is used, not proven optimal. gamma <= 0 = the official default 1 / sqrt(n).
    Warm start as the official telescoping over k: the top k by |mean difference| for the first k solved, then the
    top k |coefficients| of a balanced L2 logistic regression on the support of the nearest larger k solved.
    '''

    def __init__(self, idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int, n_filter: int = 50,
                 gamma: float = 0.0, time_limit: float = 60.0, gap_tol: float = 1e-4):
        X, scale = normalized_csr(idx, val, n_dirs)
        self.flops = 4 * X.nnz  # normalization + mean difference; each solve adds its own
        X = X.tocsc()
        self.filtered = [int(j) for j in mean_difference_ranking(X, labels)[:n_filter]]
        self.X = X[:, self.filtered].toarray()
        self.scale = scale[self.filtered]
        self.y = labels.astype(int)
        self.sign = 2.0 * self.y - 1.0
        n_pos = max(int(self.y.sum()), 1)
        self.class_weight = {1: len(self.y) / (2 * n_pos), 0: len(self.y) / (2 * max(len(self.y) - n_pos, 1))}
        self.c = np.where(self.y == 1, self.class_weight[1], self.class_weight[0])
        self.gamma = gamma if gamma > 0 else 1.0 / np.sqrt(len(self.y))
        self.time_limit, self.gap_tol = time_limit, gap_tol
        self.supports = {}  # k -> best support (positions in filtered), for the telescoping warm start

    def cut(self, support: list):
        '''The SVM on `support`: its primal objective (an upper bound on c), weights, and the cut eta + a.s >= b0.'''
        from sklearn.svm import SVC
        n, p = self.X.shape
        Xs = self.X[:, support]
        m = SVC(kernel="linear", C=self.gamma, class_weight=self.class_weight)
        m.fit(Xs, self.y)
        dual = np.zeros(n)  # = y_i beta_i gamma = -gamma alpha_i (sklearn's dual_coef_ carries the label's sign)
        dual[m.support_] = m.dual_coef_[0]
        w, b = m.coef_[0], float(m.intercept_[0])
        upper = float(self.c @ np.maximum(0.0, 1.0 - self.sign * (Xs @ w + b)) + w @ w / (2 * self.gamma))
        a = (self.X.T @ dual) ** 2 / (2 * self.gamma)  # gamma / 2 (X_j.alpha)^2 = -dc(s)/ds_j
        b0 = float(np.abs(dual).sum() / self.gamma)  # sum_i beta_i
        iters = int(np.sum(getattr(m, "n_iter_", 0)))
        self.flops += iters * (4 * Xs.size + 4 * n) + 2 * n * p + 4 * Xs.size
        return upper, w, a, b0

    def warm_start(self, k: int) -> list:
        larger = [kk for kk in self.supports if kk > k]
        if not larger:
            return list(range(k))  # filtered is in |mean difference| order
        base = self.supports[min(larger)]
        m = balanced_lr(1.0)
        m.fit(self.X[:, base], self.y)
        self.flops += int(np.max(m.n_iter_)) * 4 * self.X.shape[0] * len(base)
        return [base[i] for i in np.argsort(-np.abs(m.coef_[0]), kind="stable")[:k]]

    def solve(self, k: int) -> dict:
        '''
        {"idx": the support's latents (largest |w| first), "coef": their SVM weights (unnormalized), "status"
        ("optimal" / "timeout"), "lower" / "upper" (the MILP bound / best SVM objective), "gap", "n_cuts",
        "seconds", "gamma"}.
        '''
        import time
        from scipy.optimize import Bounds, LinearConstraint, milp

        p = len(self.filtered)
        k = min(k, p)
        start = time.time()
        cuts, rhs, seen = [], [], set()
        best, lower, status = None, 0.0, "timeout"
        support = self.warm_start(k)
        while True:
            upper, w, a, b0 = self.cut(support)
            cuts.append(a)
            rhs.append(b0)
            seen.add(frozenset(support))
            if best is None or upper < best[0]:
                best = (upper, support, w)
            if best[0] - lower <= self.gap_tol * max(abs(best[0]), 1e-12):
                status = "optimal"
                break
            left = self.time_limit - (time.time() - start)
            if left <= 0:
                break
            # variables (s_1..s_p, eta): min eta  s.t.  eta + a_t.s >= b0_t for every cut, sum s = k, s binary
            A = np.vstack([np.column_stack([np.array(cuts), np.ones(len(cuts))]), np.r_[np.ones(p), 0.0]])
            res = milp(c=np.r_[np.zeros(p), 1.0],
                       constraints=LinearConstraint(A, np.r_[rhs, k], np.r_[np.full(len(rhs), np.inf), k]),
                       integrality=np.r_[np.ones(p), 0.0], bounds=Bounds(np.zeros(p + 1), np.r_[np.ones(p), np.inf]),
                       options={"time_limit": left})
            if res.status != 0 or res.x is None:  # out of time inside the MILP
                break
            lower = max(lower, float(res.fun))
            support = [int(j) for j in np.flatnonzero(res.x[:p] > 0.5)]
            if best[0] - lower <= self.gap_tol * max(abs(best[0]), 1e-12) or frozenset(support) in seen:
                # a support already cut has eta >= its own dual value there: converged up to the SVM's tolerance
                status = "optimal"
                break
        upper, support, w = best
        self.supports[k] = support
        rank = np.argsort(-np.abs(w), kind="stable")
        return {"idx": [self.filtered[support[i]] for i in rank],
                "coef": [float(w[i] / self.scale[support[i]]) for i in rank],
                "status": status, "lower": lower, "upper": upper,
                "gap": float((upper - lower) / max(abs(upper), 1e-12)), "n_cuts": len(cuts),
                "seconds": time.time() - start, "gamma": self.gamma}


def smallest_k(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_dirs: int,
               order: list, frac: float = 0.95, metric: str = "accuracy", test_frac: float = 0.2,
               seed: int = 0, C: float = 1.0, subsets: dict = None) -> dict:
    '''
    Fewest leading latents of `order` (most important first) whose joint
    logistic regression classifies the patches at least frac x as well as one
    on EVERY latent. Both are L2 logistic regressions (liblinear, latents
    scaled to max 1, same C), fit on the patches of ~80% of the images
    (groups = image id per patch) and scored on the held-out rest. k is found
    by binary search over 1..len(order), which assumes the score rises with k
    (every evaluated k is kept in "curve" to check that).
    subsets ({k: latents}) replaces order[:k] for a probe whose k-sparse sets aren't nested (optimal sparse
    probing); the search is then over its k only.
    Returns {"k", "latents", "full", "target", "score", "curve", "metric", "order"} - "order" is the
    ranking searched, so a fixed top-k of the same ranking can be compared with the k found.
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
    ks = sorted(subsets) if subsets is not None else list(range(1, len(order) + 1))
    pick = subsets.get if subsets is not None else (lambda k: list(order[:k]))

    def at(k):
        if k not in curve:
            curve[k] = score(pick(k))
        return curve[k]

    if not ks:
        return {"k": 0, "latents": [], "full": full, "target": target, "score": float("nan"), "curve": {},
                "metric": metric, "order": []}
    lo, hi = 0, len(ks) - 1  # positions in ks
    if not at(ks[hi]) >= target:
        lo = hi  # even every candidate misses the target: use them all
    while lo < hi:
        mid = (lo + hi) // 2
        if at(ks[mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    k = ks[lo]
    return {"k": k, "latents": [int(j) for j in pick(k)], "full": full, "target": target,
            "score": at(k), "curve": {str(k): v for k, v in sorted(curve.items())}, "metric": metric,
            "order": [int(j) for j in order]}


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
    F1 (both only positive-weight latents), "lasso": lasso_select's weights,
    "adaptive": adaptive_threshold_select's nested sets (settings in
    auto["adaptive"]) - and the smallest k reaching frac of the all-latent
    classifier is kept. "osp" (settings in auto["osp"]: n_filter, gamma,
    time_limit, max_k) solves OptimalSparseProbe for every k of k_schedule(max_k),
    largest first (the official telescoping), and the search picks among those k. Both keep their settings,
    "selection_flops" and per-k sets ("coef_by_k" / "osp_by_k") in the result.
    auto["top_n"] (optional) also describes the first top_n latents of each
    ranking as "order_top", for fixed top-k sets of the same ranking.
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
            search, extra = {}, {}
            if rule == "bce":
                order = [int(j) for j in np.argsort(loss) if w[j] > 0 and loss[j] < base]
            elif rule == "f1":
                order = [int(j) for j in np.argsort(-f1) if w[j] > 0 and f1[j] > 0]
            elif rule == "adaptive":
                sel = adaptive_threshold_select(idx, val, labels, n_dirs, **auto["adaptive"])
                order = sel["order"]
                extra = {"settings": auto["adaptive"], "selection_flops": sel["flops"], "coef_by_k": sel["coef"]}
            elif rule == "osp":
                s = auto["osp"]
                probe = OptimalSparseProbe(idx, val, labels, n_dirs, s["n_filter"], s["gamma"], s["time_limit"])
                # as the official optimal_sparse_probing: every k of the schedule up to max_k, largest first
                # (each warm-starts the next), then the auto-k search picks among them
                ks = [k for k in k_schedule(min(s["max_k"], auto["max_k"]), n_dirs) if k <= len(probe.filtered)]
                solved = {k: probe.solve(k) for k in ks[::-1]}
                order = []
                search = {"subsets": {k: r["idx"] for k, r in solved.items()}}
            else:
                order = lasso_select(idx, val, labels, n_dirs, auto["max_k"])["idx"]
            res = smallest_k(idx, val, labels, auto["groups"], n_dirs, order[:auto["max_k"]],
                             frac=auto["frac"], metric=auto["metric"], seed=auto["seed"], **search)
            if rule == "osp":  # "order": the largest k's support, for --inject_top_k
                res["order"] = solved[max(solved)]["idx"] if solved else []
                extra = {"settings": s, "selection_flops": float(probe.flops), "n_filtered": len(probe.filtered),
                         "osp_by_k": {str(k): r for k, r in sorted(solved.items())}}
            res.update(extra)
            res["top"] = [describe(j) for j in res["latents"]]
            res["order_top"] = [describe(j) for j in res["order"][:auto.get("top_n", 0)]]
            out["auto"][rule] = res
    if lasso:
        sel = lasso_select(idx, val, labels, n_dirs, top_k)
        out["lasso_by_k"] = {str(top_k): {
            "top": [{**describe(j), "lasso_coef": c} for j, c in zip(sel["idx"], sel["coef"])],
            "C": sel["C"], "n_selected": sel["n_selected"]}}
    return out
