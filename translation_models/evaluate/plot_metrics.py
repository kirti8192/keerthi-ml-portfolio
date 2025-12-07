"""
Plot training metrics stored in config.METRICS_DIR/
and save the plots to config.PLOTS_DIR/
"""

from pathlib import Path
import csv
import os
import matplotlib.pyplot as plt

import config


def load_loss_history(metrics_path: Path):
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    epochs, train_losses, dev_losses = [], [], []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            dev_losses.append(float(row["dev_loss"]))

    if not epochs:
        raise ValueError(f"No rows found in metrics file: {metrics_path}")

    return epochs, train_losses, dev_losses


def plot_losses(epochs, train_losses, dev_losses, save_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, dev_losses, label="Dev Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Seq2Seq Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def main():
    metrics_path = config.METRICS_DIR / "loss_history.csv"
    plot_path = config.PLOTS_DIR / "loss_curve.png"

    epochs, train_losses, dev_losses = load_loss_history(metrics_path)
    plot_losses(epochs, train_losses, dev_losses, plot_path)


if __name__ == "__main__":
    main()
