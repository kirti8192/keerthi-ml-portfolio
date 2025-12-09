"""
Plot training metrics stored in the metrics directory and save figures to a plots directory.

CLI flags:
  --local (default): use paths from config.py
  --drive: use a fixed Google Drive root for metrics/plots
"""

from pathlib import Path
import sys
import os
import argparse
import csv
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402



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


def resolve_paths(use_drive: bool):
    if use_drive:
        drive_root = Path("/Users/kirti8192/Library/CloudStorage/GoogleDrive-kirti8192@gmail.com/My Drive/colab_data/translation_models")
        metrics_dir = drive_root / "outputs" / "metrics" / config.LANG_PAIR
        plots_dir = drive_root / "outputs" / "plots" / config.LANG_PAIR
    else:
        metrics_dir = config.METRICS_DIR
        plots_dir = config.PLOTS_DIR
    return metrics_dir, plots_dir


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from saved metrics.")
    parser.add_argument("--model-name", default=config.MODEL_NAME, help="Model name prefix for metrics files (default: config.MODEL_NAME).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Use local paths from config.py (default).")
    group.add_argument("--drive", action="store_true", help="Use Google Drive paths for metrics/plots.")
    args = parser.parse_args()

    model_name = args.model_name
    use_drive = args.drive
    metrics_dir, plots_dir = resolve_paths(use_drive)

    metrics_path = metrics_dir / f"{model_name}_loss_history.csv"
    plot_path = plots_dir / f"{model_name}_loss_curve.png"

    epochs, train_losses, dev_losses = load_loss_history(metrics_path)
    plot_losses(epochs, train_losses, dev_losses, plot_path)


if __name__ == "__main__":
    main()
