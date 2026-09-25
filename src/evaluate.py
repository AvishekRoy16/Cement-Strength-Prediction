"""Reproducible benchmark for the UCI Concrete Compressive Strength dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Concrete.xls"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SEED = 42

SHORT_NAMES = [
    "Cement",
    "Blast-furnace slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
    "Strength",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_excel(DATA_PATH, engine="xlrd")
    if frame.shape != (1030, 9):
        raise ValueError(f"Expected UCI shape (1030, 9), received {frame.shape}")
    frame.columns = SHORT_NAMES
    if frame.isna().any().any():
        raise ValueError("Unexpected missing values in the concrete dataset")
    return frame.drop(columns="Strength"), frame["Strength"]


def models() -> dict[str, object]:
    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Ridge regression": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            transformer=StandardScaler(),
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=SEED,
        ),
    }


def evaluate(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    rows = []
    for name, model in models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "mae_mean_mpa": -scores["test_mae"].mean(),
                "mae_std_mpa": scores["test_mae"].std(),
                "rmse_mean_mpa": -scores["test_rmse"].mean(),
                "rmse_std_mpa": scores["test_rmse"].std(),
                "r2_mean": scores["test_r2"].mean(),
                "r2_std": scores["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_mean_mpa").reset_index(drop=True)


def save_model_comparison(results: pd.DataFrame) -> None:
    ordered = results.sort_values("mae_mean_mpa", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(
        ordered["model"],
        ordered["mae_mean_mpa"],
        xerr=ordered["mae_std_mpa"],
        color=["#2563eb", "#60a5fa", "#93c5fd", "#dbeafe"],
        capsize=4,
    )
    ax.set_xlabel("Repeated 5-fold cross-validation MAE (MPa; lower is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model-comparison.png", dpi=180)
    plt.close(fig)


def save_diagnostics(X: pd.DataFrame, y: pd.Series) -> None:
    model = models()["Gradient boosting"]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    residuals = y.to_numpy() - predictions

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(y, predictions, alpha=0.55, s=22, color="#2563eb", edgecolors="none")
    limits = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
    axes[0].plot(limits, limits, "--", color="#111827", linewidth=1)
    axes[0].set(xlabel="Observed strength (MPa)", ylabel="Cross-validated prediction (MPa)")
    axes[0].set_title("Observed vs predicted")

    axes[1].scatter(predictions, residuals, alpha=0.55, s=22, color="#0f766e", edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="#111827", linewidth=1)
    axes[1].set(xlabel="Cross-validated prediction (MPa)", ylabel="Residual (MPa)")
    axes[1].set_title("Residual pattern")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "prediction-diagnostics.png", dpi=180)
    plt.close(fig)

    model.fit(X, y)
    importance = permutation_importance(
        model, X, y, scoring="neg_mean_absolute_error", n_repeats=20, random_state=SEED, n_jobs=-1
    )
    order = np.argsort(importance.importances_mean)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        X.columns[order],
        importance.importances_mean[order],
        xerr=importance.importances_std[order],
        color="#0f766e",
        capsize=3,
    )
    ax.set_xlabel("Increase in MAE after permutation (MPa)")
    ax.set_title("Permutation importance (descriptive, not causal)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation-importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    results = evaluate(X, y)
    results.to_csv(REPORT_DIR / "model-results.csv", index=False, float_format="%.4f")
    save_model_comparison(results)
    save_diagnostics(X, y)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
