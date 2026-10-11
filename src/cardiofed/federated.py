"""Federated learning simulation: 4 hospitals that cannot share patient records.

    python -m cardiofed.federated

Each hospital is a client holding only its own training patients (the same split as
train.py). Raw rows never leave a client; only aggregate statistics and model parameters do.

Strategies compared on every hospital's held-out patients:
    local only         each hospital trains on its own data            (no collaboration)
    FedAvg (logistic)  McMahan et al. 2017: local SGD epochs, server averages weights
    federated XGBoost  each hospital trains locally; server averages the models' log-odds
                       (one round, model sharing only)
    centralised        all data pooled in one place                    (privacy-violating upper bound)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cardiofed import config as C
from cardiofed.data import load_clean
from cardiofed.evaluation import r3, train_test_indices
from cardiofed.features import build_features, lr_matrix
from cardiofed.imputation import StochasticImputer
from cardiofed.model import fit_xgb, mc_proba
from cardiofed.train import BASE_PARAMS

XGB_PARAMS = {**BASE_PARAMS, "max_depth": 2, "n_estimators": 150}


# ====================================================================== clients
class Client:
    def __init__(self, name: str, X: pd.DataFrame, y: np.ndarray):
        self.name, self.X, self.y = name, X, y
        self.n = len(y)

    # --- step 1 of FedAvg preprocessing: share only sums / counts, never rows
    def feature_stats(self, Z: pd.DataFrame) -> dict:
        v = Z.to_numpy(dtype=float)
        obs = ~np.isnan(v)
        return {"count": obs.sum(0), "sum": np.nansum(v, 0), "sumsq": np.nansum(v ** 2, 0)}

    # --- one FedAvg round of local training (full-batch gradient steps, L2 penalty)
    def local_update(self, w: np.ndarray, A: np.ndarray, epochs: int, lr: float, l2: float) -> np.ndarray:
        w = w.copy()
        y = self.y.astype(float)
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(A @ w)))
            grad = A.T @ (p - y) / len(y)
            grad[1:] += l2 * w[1:]
            w -= lr * grad
        return w


def _design(Z: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Global-mean imputation + standardisation, then a bias column."""
    v = Z.to_numpy(dtype=float)
    v = np.where(np.isnan(v), mean, v)
    v = (v - mean) / std
    return np.hstack([np.ones((len(v), 1)), v])


def auc_or_none(y, p):
    return float(roc_auc_score(y, p)) if min(y.sum(), (1 - y).sum()) >= 5 else None


def evaluate(pred_fn, tests: dict) -> dict:
    """AUC on each hospital's test patients + on all test patients pooled."""
    out, ys, ps = {}, [], []
    for name, (X, y) in tests.items():
        p = pred_fn(name, X)
        out[name] = auc_or_none(y, p)
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    # Headline: mean of per-hospital AUCs. Pooled AUC is reported but NOT used to compare,
    # because it rewards a model for knowing each hospital's base rate (a site shortcut).
    site_aucs = [v for v in out.values() if v is not None]
    out["hospital_mean"] = float(np.mean(site_aucs))
    out["pooled"] = float(roc_auc_score(y, p))
    out["pooled_brier"] = float(np.mean((p - y) ** 2))
    return out


# ====================================================================== strategies
def run(rounds: int = 100, local_epochs: int = 5, lr: float = 1.0, l2: float = 1e-2) -> dict:
    df = load_clean()
    X = build_features(df)
    y = df[C.TARGET].to_numpy()
    tr_idx, te_idx = train_test_indices(df)
    site = df["hospital"].to_numpy()
    Z = lr_matrix(X)  # one-hot view for the linear model

    clients, tests, ztr, zte = {}, {}, {}, {}
    for s in C.HOSPITALS:
        tr = tr_idx[site[tr_idx] == s]
        te = te_idx[site[te_idx] == s]
        clients[s] = Client(s, X.iloc[tr].reset_index(drop=True), y[tr])
        tests[s] = (X.iloc[te].reset_index(drop=True), y[te])
        ztr[s], zte[s] = Z.iloc[tr], Z.iloc[te]
    print("[clients] " + ", ".join(f"{s}: {c.n} train / {len(tests[s][1])} test" for s, c in clients.items()))
    results = {}

    # ---------------- 1. local only (XGBoost, each hospital alone)
    local = {}
    for s, c in clients.items():
        imp = StochasticImputer().fit(c.X)
        local[s] = (fit_xgb(XGB_PARAMS, c.X, c.y, imp), imp)
    results["local only (own model, own test set)"] = evaluate(
        lambda s, Xt: mc_proba(local[s][0], Xt, local[s][1], 20)[:, 1], tests)

    # ---------------- 2. federated XGBoost: average local models' log-odds, weighted by n
    weights = {s: c.n / sum(c.n for c in clients.values()) for s, c in clients.items()}

    def fed_xgb(_, Xt):
        logit = sum(weights[s] * np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
                    for s, (m, imp) in local.items()
                    for p in [mc_proba(m, Xt, imp, 20)[:, 1]])
        return 1 / (1 + np.exp(-logit))
    results["federated XGBoost (weighted log-odds ensemble)"] = evaluate(fed_xgb, tests)

    # ---------------- 3. FedAvg logistic regression
    stats = [c.feature_stats(ztr[s]) for s, c in clients.items()]          # clients -> server
    cnt = sum(st["count"] for st in stats)
    mean = sum(st["sum"] for st in stats) / np.maximum(cnt, 1)
    var = sum(st["sumsq"] for st in stats) / np.maximum(cnt, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-6))
    designs = {s: _design(ztr[s], mean, std) for s in clients}
    w = np.zeros(designs["cleveland"].shape[1])
    history = []
    total = sum(c.n for c in clients.values())
    for r in range(1, rounds + 1):
        updates = [(clients[s].local_update(w, designs[s], local_epochs, lr, l2), clients[s].n) for s in clients]
        w = sum(u * n for u, n in updates) / total                            # server: weighted average
        res = evaluate(lambda s, Xt: 1 / (1 + np.exp(-(_design(zte[s], mean, std) @ w))), tests)
        history.append({"round": r, "hospital_mean_auc": res["hospital_mean"], "w": w.copy()})
    results["FedAvg logistic regression"] = evaluate(
        lambda s, Xt: 1 / (1 + np.exp(-(_design(zte[s], mean, std) @ w))), tests)
    params_per_round = len(w) * len(clients) * 2  # up + down
    comm = {"parameters": int(len(w)), "rounds": rounds,
            "floats_exchanged": int(params_per_round * rounds),
            "kilobytes_exchanged_float32": round(params_per_round * rounds * 4 / 1024, 1)}

    # centralised LR with identical preprocessing, to show FedAvg converges to it
    A_all = np.vstack([designs[s] for s in clients]); y_all = np.concatenate([clients[s].y for s in clients])
    wc = Client("all", None, y_all).local_update(np.zeros_like(w), A_all, 5000, lr, l2)  # converged
    for h in history:  # how close is the federated model to the one trained on pooled data?
        h["relative_distance_to_centralised"] = float(np.linalg.norm(h.pop("w") - wc) / np.linalg.norm(wc))
    results["centralised logistic regression (same preprocessing)"] = evaluate(
        lambda s, Xt: 1 / (1 + np.exp(-(_design(zte[s], mean, std) @ wc))), tests)

    # ---------------- 4. centralised XGBoost (upper bound)
    Xall = pd.concat([c.X for c in clients.values()], ignore_index=True)
    imp = StochasticImputer().fit(Xall)
    central = fit_xgb(XGB_PARAMS, Xall, y_all, imp)
    results["centralised XGBoost (pooled data)"] = evaluate(lambda s, Xt: mc_proba(central, Xt, imp, 20)[:, 1], tests)

    print("\n[results] AUC on each hospital's held-out patients (None = too few patients in one class)")
    hdr = ["cleveland", "hungarian", "switzerland", "va", "hospital_mean", "pooled"]
    print(f"  {'strategy':54s}" + "".join(f"{h[:11]:>12s}" for h in hdr))
    for name, res in results.items():
        print(f"  {name:54s}" + "".join(f"{res[h]:12.3f}" if res[h] is not None else f"{'-':>12s}" for h in hdr))
    print("  (switzerland: only 2 healthy test patients, so no AUC; pooled AUC is inflated by base-rate"
          " differences between hospitals - compare on hospital_mean)")
    print("\n[FedAvg] round: hospital-mean AUC, distance to centralised weights")
    for h in history:
        if h["round"] in (1, 5, 10, 25, 50) or h["round"] == rounds:
            print(f"  r{h['round']:<3d} AUC {h['hospital_mean_auc']:.3f}  distance {h['relative_distance_to_centralised']:.3f}")
    print(f"[FedAvg] communication: {comm}")
    return r3({"clients": {s: {"train": c.n, "test": int(len(tests[s][1])), "train_prevalence": float(c.y.mean())}
                           for s, c in clients.items()},
               "results": results, "fedavg_history": history, "communication": comm,
               "settings": {"rounds": rounds, "local_epochs": local_epochs, "lr": lr, "l2": l2,
                            "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"}}})


def figure(rep: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    names = list(rep["results"])
    short = {"local only (own model, own test set)": "Local only",
             "federated XGBoost (weighted log-odds ensemble)": "Federated XGBoost",
             "FedAvg logistic regression": "FedAvg LogReg",
             "centralised logistic regression (same preprocessing)": "Central LogReg",
             "centralised XGBoost (pooled data)": "Central XGBoost"}
    vals = [rep["results"][n]["hospital_mean"] for n in names]
    colors = ["#8a8a8a", "#c0392b", "#c0392b", "#d9b8b3", "#d9b8b3"]
    a.barh([short[n] for n in names][::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        a.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    a.set(xlim=(0.75, 0.88), xlabel="Mean per-hospital test AUC (Cleveland, Hungary, VA)",
          title="Federated vs local vs centralised")
    h = rep["fedavg_history"]
    b.plot([x["round"] for x in h], [x["relative_distance_to_centralised"] for x in h], color="#c0392b", lw=2)
    b.set(xlabel="Communication round", ylabel="||w_fed - w_central|| / ||w_central||",
          title="FedAvg approaches the pooled-data model", ylim=(0, None))
    fig.tight_layout(); fig.savefig(C.FIGURES / "federated.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    rep = run(rounds=args.rounds)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "federated.json").write_text(json.dumps(rep, indent=2))
    if not args.no_figures:
        figure(rep)
    print("\n[done] reports/federated.json, reports/figures/federated.png")


if __name__ == "__main__":
    main()
