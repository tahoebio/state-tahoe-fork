#!/usr/bin/env python
"""
delta_metric_variants.py -- compare pseudobulk-delta Pearson under 3 aggregation/metric
spaces for the log1p Caspian STATE folds, to separate "broken model" from "metric fragility".

For each fold we compute group means per (cell, batch, pert) in BOTH linear (expm1) and log1p
space in a single pass, then build per-(cell,pert) delta vectors three ways and row-wise-Pearson
pred-vs-real (averaged per cell line + overall):

  A) canonical (STATE Fig 2E):  linear agg, linear delta
       pb = mean(expm1(X));  delta = pb_t - pb_c          [reproduces eval_arc_checkpoints --log1p]
  B) linear-agg + log-delta:    true pseudobulk, robust metric
       pb = mean(expm1(X));  delta = log1p(pb_t) - log1p(pb_c)
  C) log-agg + log-delta:       no expm1 at all (the earlier quick diagnostic)
       pb = mean(log1p-stored X);  delta = pb_t - pb_c
"""
import sys, gc, numpy as np, anndata as ad, pandas as pd
import scipy.sparse as sp

CTRL = "[('DMSO_TF', 0.0, 'uM')]"
BATCH, CELL, PERT = "plate", "Cell_ID_Cellosaur", "drugname_drugconc"
BASE = "/nvme-shared/models/state-arc/fewshot-caspian{}/eval_best.ckpt/{}"


def group_means(adata):
    """Return (cell_lbl, pert_lbl, batch_lbl, gm_lin, gm_log) per (cell,batch,pert) group.
    gm_lin = mean(expm1(X)); gm_log = mean(X) where X is the stored log1p values."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64, copy=False)
    Xlin = np.expm1(X)
    obs = adata.obs
    cc, uc = pd.factorize(obs[CELL].astype(str).values)
    bb, ub = pd.factorize(obs[BATCH].astype(str).values)
    pp, up = pd.factorize(obs[PERT].astype(str).values)
    nb, npe = len(ub), len(up)
    combo = (cc.astype(np.int64) * nb + bb) * npe + pp
    g, gu = pd.factorize(combo)
    G = len(gu)
    counts = np.bincount(g, minlength=G).astype(np.float64)
    onehot = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
    gm_lin = (onehot @ Xlin) / counts[:, None]
    gm_log = (onehot @ X) / counts[:, None]
    pu = gu % npe; bu = (gu // npe) % nb; cu = gu // (npe * nb)
    return uc[cu], up[pu], ub[bu], gm_lin, gm_log, up


def deltas(adata):
    """Build per-(cell,pert) delta DataFrames for the 3 variants. Returns dict variant->DataFrame
    indexed by (cell,pert) with gene columns."""
    cell_lbl, pert_lbl, batch_lbl, gm_lin, gm_log, up = group_means(adata)
    G = len(cell_lbl)
    # control group row per (cell,batch)
    ctrl_row = {}
    for gi in range(G):
        if pert_lbl[gi] == CTRL:
            ctrl_row[(cell_lbl[gi], batch_lbl[gi])] = gi
    from collections import defaultdict
    aggA, aggB, aggC = defaultdict(list), defaultdict(list), defaultdict(list)
    for gi in range(G):
        if pert_lbl[gi] == CTRL:
            continue
        cr = ctrl_row.get((cell_lbl[gi], batch_lbl[gi]))
        if cr is None:
            continue
        key = (cell_lbl[gi], pert_lbl[gi])
        aggA[key].append(gm_lin[gi] - gm_lin[cr])
        aggB[key].append(np.log1p(gm_lin[gi]) - np.log1p(gm_lin[cr]))
        aggC[key].append(gm_log[gi] - gm_log[cr])
    def to_df(agg):
        rows = [(c, p, np.mean(v, axis=0)) for (c, p), v in agg.items()]
        idx = pd.MultiIndex.from_tuples([(c, p) for c, p, _ in rows], names=["cell_line", "treatment"])
        return pd.DataFrame(np.vstack([r[2] for r in rows]), index=idx)
    return {"A": to_df(aggA), "B": to_df(aggB), "C": to_df(aggC)}


def rowwise_pearson(P, R):
    Pc = P - P.mean(1, keepdims=True); Rc = R - R.mean(1, keepdims=True)
    num = (Pc * Rc).sum(1)
    den = np.sqrt((Pc**2).sum(1) * (Rc**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r


def main():
    folds = [int(x) for x in (sys.argv[1:] or [5, 6, 7, 8, 9])]
    print(f"Folds: {folds}", flush=True)
    summary = {}
    for N in folds:
        print(f"\n===== FOLD {N} =====", flush=True)
        pred = ad.read_h5ad(BASE.format(N, "adata_pred.h5ad"))
        real = ad.read_h5ad(BASE.format(N, "adata_real.h5ad"))
        Dp = deltas(pred); Dr = deltas(real)
        del pred, real; gc.collect()
        for V, label in [("A", "canonical (lin agg, lin delta)"),
                         ("B", "lin agg, LOG delta (robust)"),
                         ("C", "log agg, log delta (no expm1)")]:
            dp, dr = Dp[V], Dr[V]
            common = dp.index.intersection(dr.index)
            P = dp.loc[common].to_numpy(); R = dr.loc[common].to_numpy()
            P = np.nan_to_num(P); R = np.nan_to_num(R)
            r = rowwise_pearson(P, R)
            overall = float(np.nanmean(r))
            bycl = pd.Series(r, index=common.get_level_values("cell_line")).groupby(level=0).mean()
            summary[(N, V)] = overall
            print(f"  [{V}] {label:34s} mean={overall:.4f}  median={float(np.nanmedian(r)):.4f}  n={len(r)}", flush=True)
            for cl, v in bycl.items():
                print(f"        {cl}: {v:.4f}", flush=True)
        gc.collect()
    print("\n===== SUMMARY (mean pearson_delta) =====", flush=True)
    print(f"  {'fold':>4}  {'A canonical':>12}  {'B lin/log':>12}  {'C no-expm1':>12}", flush=True)
    for N in folds:
        print(f"  {N:>4}  {summary[(N,'A')]:>12.4f}  {summary[(N,'B')]:>12.4f}  {summary[(N,'C')]:>12.4f}", flush=True)


if __name__ == "__main__":
    main()
