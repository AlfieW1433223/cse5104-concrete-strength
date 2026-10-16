#!/usr/bin/env python3
"""Reproduce the Concrete Compressive Strength Project 1 analysis.

Usage:
  python analyze_concrete.py --data Concrete_Data.xls --output-dir results

The held-out test set is observations 501-630 (one-based, original row order).
All preprocessing is fitted on the 900 training observations only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
TARGET = "Concrete compressive strength"


def load_data(path: Path) -> pd.DataFrame:
    # pandas reads .xls via xlrd and .xlsx via openpyxl. Header names come from UCI.
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) != 9:
        raise ValueError(f"Expected 9 columns, found {len(df.columns)}: {list(df.columns)}")
    df.columns = FEATURES + [TARGET]
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in supplied data")
    if len(df) != 1030:
        raise ValueError(f"Expected 1,030 observations, found {len(df)}")
    return df


def split_by_assignment(df: pd.DataFrame):
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[500:630] = True  # original observations 501 through 630 inclusive
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)


def mse_r2(y, pred):
    err = np.asarray(y) - np.asarray(pred)
    mse = float(np.mean(err ** 2))
    r2 = float(1 - mse / np.var(y, ddof=0))
    return mse, r2


def gradient_descent(X, y, alpha, max_iter=200_000, tol=1e-10):
    """Batch gradient descent on mean squared error; returns parameters and loss trace."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    Xa = np.column_stack([np.ones(len(X)), X])
    theta = np.zeros(Xa.shape[1], dtype=float)
    losses = []
    for it in range(max_iter):
        residual = Xa @ theta - y
        loss = float(np.mean(residual ** 2))
        if not np.isfinite(loss):
            raise FloatingPointError("Gradient descent diverged; reduce learning rate")
        if it % 100 == 0:
            losses.append((it, loss))
        grad = (2.0 / len(y)) * (Xa.T @ residual)
        theta -= alpha * grad
        if np.linalg.norm(grad, ord=np.inf) < tol:
            losses.append((it + 1, float(np.mean((Xa @ theta - y) ** 2))))
            break
    else:
        losses.append((max_iter, float(np.mean((Xa @ theta - y) ** 2))))
    return theta, pd.DataFrame(losses, columns=["iteration", "training_mse"])


def predict_gd(X, theta):
    return np.column_stack([np.ones(len(X)), np.asarray(X)]) @ theta


def eval_line(name, model, Xtr, Xte, ytr, yte):
    ptrain, ptest = model.predict(Xtr), model.predict(Xte)
    tr_mse, tr_r2 = mse_r2(ytr, ptrain)
    te_mse, te_r2 = mse_r2(yte, ptest)
    return {"model": name, "train_mse": tr_mse, "test_mse": te_mse,
            "train_r2": tr_r2, "test_r2": te_r2}


def run(data: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(data)
    tr, te = split_by_assignment(df)
    Xtr, Xte = tr[FEATURES].to_numpy(float), te[FEATURES].to_numpy(float)
    ytr, yte = tr[TARGET].to_numpy(float), te[TARGET].to_numpy(float)

    # Descriptive statistics and study plots use training observations only.
    desc = tr[FEATURES + [TARGET]].agg(["mean", "std", "min", "max"]).T
    desc.to_csv(outdir / "training_descriptive_statistics.csv", float_format="%.6f")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(ytr, bins=24, color="#2E6F95", edgecolor="white")
    ax.set(title="Training-set compressive strength distribution", xlabel="Strength (MPa)", ylabel="Number of mixes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "figure_strength_distribution.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, col, color in zip(axes, ["Cement", "Age"], ["#2E6F95", "#C56E35"]):
        ax.scatter(tr[col], ytr, s=15, alpha=.55, color=color, edgecolors="none")
        ax.set(xlabel=("Cement (kg/m³)" if col == "Cement" else "Age (days)"), ylabel="Strength (MPa)", title=f"{col} and measured strength")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "figure_key_relationships.png", dpi=220); plt.close(fig)

    # Gradient descent convergence: raw inputs, train-only standardized inputs, log1p inputs.
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    Ltr, Lte = np.log1p(Xtr), np.log1p(Xte)
    gd_specs = [("Untransformed", Xtr, 2.3e-7), ("Standardized", Ztr, 0.15), ("Log-transformed", Ltr, 0.002)]
    traces, gd_rows = {}, []
    for name, X, alpha in gd_specs:
        theta, trace = gradient_descent(X, ytr, alpha=alpha, max_iter=200_000)
        traces[name] = trace
        pred = predict_gd(X, theta)
        gmse, gr2 = mse_r2(ytr, pred)
        gd_rows.append({"model": name, "learning_rate": alpha, "iterations_recorded": int(trace.iteration.iloc[-1]), "gd_train_mse": gmse, "gd_train_r2": gr2})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, trace in traces.items():
        ax.plot(trace.iteration, trace.training_mse, label=name, linewidth=2)
    ax.set(xlabel="Iteration", ylabel="Training MSE (MPa²)", title="Gradient descent training loss")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=.2); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "figure_gradient_descent_loss.png", dpi=220); plt.close(fig)
    pd.concat([t.assign(model=n) for n, t in traces.items()], ignore_index=True).to_csv(outdir / "gradient_descent_loss.csv", index=False)
    pd.DataFrame(gd_rows).to_csv(outdir / "gradient_descent_settings.csv", index=False)

    # Optimized models use external scikit-learn solver; scaling/log scaling are train-only transforms.
    opt_rows, models = [], {}
    for name, A, B in [("Untransformed", Xtr, Xte), ("Standardized", Ztr, Zte), ("Log-transformed", Ltr, Lte)]:
        model = LinearRegression().fit(A, ytr)
        models[name] = model
        opt_rows.append(eval_line(name, model, A, B, ytr, yte))
    opt = pd.DataFrame(opt_rows)
    opt.to_csv(outdir / "optimal_model_metrics.csv", index=False, float_format="%.8f")

    # Univariate OLS slopes against corresponding coefficients in the full multivariate OLS fit.
    multi = LinearRegression().fit(Xtr, ytr)
    uni_rows = []
    for j, f in enumerate(FEATURES):
        one = LinearRegression().fit(Xtr[:, [j]], ytr)
        uni_rows.append({"feature": f, "univariate_slope": one.coef_[0], "univariate_intercept": one.intercept_, "multivariate_slope": multi.coef_[j]})
    pd.DataFrame(uni_rows).to_csv(outdir / "univariate_vs_multivariate_coefficients.csv", index=False, float_format="%.8f")

    # Required custom gradient-descent runs for each raw univariate predictor.
    uni_gd_rows = []
    for j, feature in enumerate(FEATURES):
        Xi = Xtr[:, [j]]
        Xa = np.column_stack([np.ones(len(Xi)), Xi])
        lipschitz = float(np.linalg.eigvalsh((2.0 / len(Xi)) * (Xa.T @ Xa))[-1])
        alpha = 0.8 / lipschitz
        theta, trace = gradient_descent(Xi, ytr, alpha=alpha)
        pred = predict_gd(Xi, theta)
        umse, ur2 = mse_r2(ytr, pred)
        uni_gd_rows.append({"feature": feature, "learning_rate": alpha,
                            "iterations_recorded": int(trace.iteration.iloc[-1]),
                            "train_mse": umse, "train_r2": ur2})
    pd.DataFrame(uni_gd_rows).to_csv(outdir / "univariate_gradient_descent.csv", index=False, float_format="%.8f")

    # OLS inference for three designs. The predictor p-values are invariant to invertible
    # rescaling of columns; log1p changes the modeled functional form.
    p_rows = []
    for name, X in [("Untransformed", Xtr), ("Standardized", Ztr), ("Log-transformed", Ltr)]:
        fit = sm.OLS(ytr, sm.add_constant(X, has_constant="add")).fit()
        for f, coef, p in zip(FEATURES, fit.params[1:], fit.pvalues[1:]):
            p_rows.append({"model": name, "feature": f, "coefficient": coef, "p_value": p})
    ptab = pd.DataFrame(p_rows)
    ptab.to_csv(outdir / "feature_significance.csv", index=False, float_format="%.10g")

    # Readable summary with key numeric facts for downstream reporting.
    pd.DataFrame({"rows": [len(df), len(tr), len(te)], "partition": ["all", "training", "test"]}).to_csv(outdir / "partition_counts.csv", index=False)
    print("PARTITION", len(tr), len(te))
    print("OPTIMAL METRICS\n", opt.to_string(index=False))
    print("OLS UNTRANSFORMED COEFFICIENTS\n", pd.DataFrame({"feature": FEATURES, "coef": multi.coef_}).to_string(index=False))
    print("PVALUES\n", ptab.pivot(index="feature", columns="model", values="p_value").to_string())
    print("GD\n", pd.DataFrame(gd_rows).to_string(index=False))
    print("UNIVARIATE GD\n", pd.DataFrame(uni_gd_rows).to_string(index=False))
    print("DESCRIPTIVES\n", desc.round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to UCI Concrete_Data.xls or converted .xlsx")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data, args.output_dir)

if __name__ == "__main__":
    main()
