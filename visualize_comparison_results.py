"""
Visualize and tabulate comparison results from three_stage_compare_20260220_141755.
Reads all model result CSVs and produces:
  1. A combined summary table (printed + saved as CSV)
  2. Classification metrics bar plots
  3. Regression metrics bar plots
  4. Per-channel breakdown plots
  5. Training loss curves (all stages)
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "three_stage_compare_20260220_141755",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_plots_20260220")
os.makedirs(OUT_DIR, exist_ok=True)

# ── model files ──────────────────────────────────────────────────────────────
MODEL_FILES = {
    "STPN":           "stpn_results_table.csv",
    "CNN":            "cnn_results_table.csv",
    "GRU":            "gru_results_table.csv",
    "LSTM":           "lstm_results_table.csv",
    "BiLSTM":         "bilstm_results_table.csv",
    "Attn-LSTM":      "attnlstm_results_table.csv",
    "CNN-LSTM":       "cnnlstm_results_table.csv",
}

HISTORY_FILES = {
    "STPN":      "stpn_history.csv",
    "CNN":       "cnn_history.csv",
    "GRU":       "gru_history.csv",
    "LSTM":      "lstm_history.csv",
    "BiLSTM":    "bilstm_history.csv",
    "Attn-LSTM": "attnlstm_history.csv",
    "CNN-LSTM":  "cnnlstm_history.csv",
}

# colour palette (one per model, consistent across all plots)
PALETTE = {
    "STPN":      "#e41a1c",
    "CNN":       "#377eb8",
    "GRU":       "#4daf4a",
    "LSTM":      "#984ea3",
    "BiLSTM":    "#ff7f00",
    "Attn-LSTM": "#a65628",
    "CNN-LSTM":  "#f781bf",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_summary_section(path: str) -> dict:
    """Extract the SUMMARY (raw metric/value) block from a results_table CSV."""
    in_summary = False
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("SUMMARY"):
                in_summary = True
                continue
            if in_summary:
                parts = line.split(",")
                if len(parts) == 2:
                    key, val = parts
                    try:
                        rows[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
    return rows


def load_all_results() -> pd.DataFrame:
    records = []
    for model, fname in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        data = parse_summary_section(path)
        data["model"] = model
        records.append(data)
    df = pd.DataFrame(records).set_index("model")
    return df


def load_history(model: str) -> pd.DataFrame:
    fname = HISTORY_FILES[model]
    path = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(path)
    return df


# ── load data ─────────────────────────────────────────────────────────────────
df = load_all_results()

# ── 1. COMBINED SUMMARY TABLE ─────────────────────────────────────────────────
COLS_TABLE = [
    ("classification_precision",     "Clf Precision"),
    ("classification_recall",        "Clf Recall"),
    ("classification_f1",            "Clf F1"),
    ("classification_accuracy",      "Clf Accuracy"),
    ("regression_mae_overall",       "Reg MAE (overall)"),
    ("regression_rmse_overall",      "Reg RMSE (overall)"),
    ("regression_mae_delayed",       "Reg MAE (delayed)"),
    ("regression_rmse_delayed",      "Reg RMSE (delayed)"),
    ("regression_mae_nondelayed",    "Reg MAE (non-delayed)"),
    ("regression_rmse_nondelayed",   "Reg RMSE (non-delayed)"),
]

table = df[[c for c, _ in COLS_TABLE]].copy()
table.columns = [label for _, label in COLS_TABLE]
table = table.round(4)

print("\n" + "=" * 90)
print("COMBINED RESULTS TABLE")
print("=" * 90)
print(table.to_string())
print("=" * 90)

csv_path = os.path.join(OUT_DIR, "combined_results_table.csv")
table.to_csv(csv_path)
print(f"\nSaved combined table → {csv_path}")

# ── 2. CLASSIFICATION BAR CHART ───────────────────────────────────────────────
clf_metrics = ["Clf Precision", "Clf Recall", "Clf F1", "Clf Accuracy"]
models = list(PALETTE.keys())
x = np.arange(len(clf_metrics))
n = len(models)
width = 0.11
offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

fig, ax = plt.subplots(figsize=(12, 6))
for i, model in enumerate(models):
    vals = [table.loc[model, m] for m in clf_metrics]
    ax.bar(x + offsets[i], vals, width=width,
           color=PALETTE[model], label=model, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(clf_metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Classification Metrics — All Models (Macro Avg)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
path = os.path.join(OUT_DIR, "classification_metrics.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 3. REGRESSION MAE / RMSE BAR CHART ───────────────────────────────────────
reg_groups = [
    ("Overall MAE", "regression_mae_overall"),
    ("Overall RMSE", "regression_rmse_overall"),
    ("Delayed MAE", "regression_mae_delayed"),
    ("Non-del MAE", "regression_mae_nondelayed"),
]

x = np.arange(len(reg_groups))
fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(models):
    vals = [df.loc[model, col] for _, col in reg_groups]
    ax.bar(x + offsets[i], vals, width=width,
           color=PALETTE[model], label=model, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([label for label, _ in reg_groups], fontsize=11)
ax.set_ylabel("Minutes", fontsize=11)
ax.set_title("Regression Metrics — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
path = os.path.join(OUT_DIR, "regression_metrics.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 4. PER-CHANNEL CLASSIFICATION (Arrival vs Departure) ─────────────────────
channels = ["Arrival", "Departure"]
channel_cols = {
    "Arrival":   ("classification_precision_arrival",
                  "classification_recall_arrival",
                  "classification_f1_arrival",
                  "classification_accuracy_arrival"),
    "Departure": ("classification_precision_departure",
                  "classification_recall_departure",
                  "classification_f1_departure",
                  "classification_accuracy_departure"),
}
metric_labels = ["Precision", "Recall", "F1", "Accuracy"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, channel in zip(axes, channels):
    cols = channel_cols[channel]
    x = np.arange(len(metric_labels))
    for i, model in enumerate(models):
        vals = [df.loc[model, c] for c in cols]
        ax.bar(x + offsets[i], vals, width=width,
               color=PALETTE[model], label=model, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title(f"{channel} Channel", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if ax is axes[0]:
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
fig.suptitle("Per-Channel Classification Metrics", fontsize=13, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUT_DIR, "per_channel_classification.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 5. PER-CHANNEL REGRESSION (MAE only) ─────────────────────────────────────
reg_ch_groups = [
    ("Delayed\nArrival",     "regression_mae_delayed_arrival"),
    ("Delayed\nDeparture",   "regression_mae_delayed_departure"),
    ("Non-del\nArrival",     "regression_mae_nondelayed_arrival"),
    ("Non-del\nDeparture",   "regression_mae_nondelayed_departure"),
    ("Overall\nArrival",     "regression_mae_overall_arrival"),
    ("Overall\nDeparture",   "regression_mae_overall_departure"),
]
x = np.arange(len(reg_ch_groups))
n_wide = len(models)
width2 = 0.09
offsets2 = np.linspace(-(n_wide - 1) / 2 * width2, (n_wide - 1) / 2 * width2, n_wide)

fig, ax = plt.subplots(figsize=(16, 6))
for i, model in enumerate(models):
    vals = [df.loc[model, col] for _, col in reg_ch_groups]
    ax.bar(x + offsets2[i], vals, width=width2,
           color=PALETTE[model], label=model, edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([label for label, _ in reg_ch_groups], fontsize=9)
ax.set_ylabel("MAE (minutes)", fontsize=11)
ax.set_title("Per-Channel Regression MAE — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
path = os.path.join(OUT_DIR, "per_channel_regression_mae.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 6. RADAR CHART (classification metrics) ───────────────────────────────────
radar_metrics = ["Clf Precision", "Clf Recall", "Clf F1", "Clf Accuracy"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for model in models:
    vals = [table.loc[model, m] for m in radar_metrics]
    vals += vals[:1]
    ax.plot(angles, vals, color=PALETTE[model], linewidth=2, label=model)
    ax.fill(angles, vals, color=PALETTE[model], alpha=0.07)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Classification Metrics Radar", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
fig.tight_layout()
path = os.path.join(OUT_DIR, "classification_radar.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 7. F1 vs MAE scatter (trade-off) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for model in models:
    f1  = df.loc[model, "classification_f1"]
    mae = df.loc[model, "regression_mae_overall"]
    ax.scatter(mae, f1, color=PALETTE[model], s=120, zorder=5)
    ax.annotate(model, (mae, f1), textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_xlabel("Overall Regression MAE (minutes)", fontsize=11)
ax.set_ylabel("Classification F1 (macro)", fontsize=11)
ax.set_title("F1 vs. Regression MAE — Trade-off View", fontsize=13, fontweight="bold")
ax.grid(linestyle="--", alpha=0.5)
fig.tight_layout()
path = os.path.join(OUT_DIR, "f1_vs_mae_scatter.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 8. TRAINING LOSS CURVES (stage 3) ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stage_labels = {1: "Stage 1 – Classification", 2: "Stage 2 – Regression Head", 3: "Stage 3 – Fine-tune"}

for ax, stage in zip(axes, [1, 2, 3]):
    for model in models:
        hist = load_history(model)
        # STPN has different columns
        if "stage" in hist.columns:
            sub = hist[hist["stage"] == stage]
            loss_col = "train_loss"
        else:
            sub = hist[hist["stage"] == stage]
            loss_col = "train_loss"
        if sub.empty:
            continue
        ep = sub["epoch"].values if "epoch" in sub.columns else np.arange(1, len(sub) + 1)
        ax.plot(ep, sub[loss_col].values, color=PALETTE[model], label=model, linewidth=1.8)

    ax.set_title(stage_labels[stage], fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

fig.suptitle("Training Loss Curves by Stage", fontsize=13, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUT_DIR, "training_loss_curves.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── 9. HEATMAP of all key metrics ────────────────────────────────────────────
heat_cols_raw = [c for c, _ in COLS_TABLE]
heat_labels   = [l for _, l in COLS_TABLE]
heat_data = df[heat_cols_raw].copy()

# For regression metrics, lower is better → invert for display colour
reg_cols = [c for c in heat_cols_raw if c.startswith("regression")]

# Normalise each column 0-1 (higher = better visually)
norm = heat_data.copy()
for col in heat_cols_raw:
    mn, mx = heat_data[col].min(), heat_data[col].max()
    if col in reg_cols:
        norm[col] = (mx - heat_data[col]) / (mx - mn + 1e-9)
    else:
        norm[col] = (heat_data[col] - mn) / (mx - mn + 1e-9)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(heat_labels)))
ax.set_xticklabels(heat_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels(models, fontsize=10)

# annotate cells with raw values
for r, model in enumerate(models):
    for c, col in enumerate(heat_cols_raw):
        raw = heat_data.loc[model, col]
        ax.text(c, r, f"{raw:.3f}", ha="center", va="center", fontsize=7.5,
                color="black")

plt.colorbar(im, ax=ax, label="Normalised score (green = better)")
ax.set_title("Model Performance Heatmap", fontsize=13, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUT_DIR, "performance_heatmap.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved → {path}")

# ── done ──────────────────────────────────────────────────────────────────────
print(f"\nAll outputs saved to: {OUT_DIR}")
