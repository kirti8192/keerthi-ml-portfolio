"""
Plot training metrics stored in the metrics directory and save figures to a plots directory.

CLI flags:
  --local (default): use paths from config.py
  --drive: use a fixed Google Drive root for metrics/plots
  --model-name: optional filter to only plot a specific model
"""

from pathlib import Path
import sys
import os
import argparse
import csv
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def find_metric_files(metrics_dir: Path):
    """Yield (model_name, path) for each metrics CSV in the directory."""
    for path in sorted(metrics_dir.glob("*_loss_history.csv")):
        model_name = path.stem.replace("_loss_history", "")
        yield model_name, path


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
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linestyle="-")
    plt.plot(epochs, dev_losses, label="Dev Loss", marker="o", linestyle="--")


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
    parser.add_argument("--model-name", help="Optional: only plot this model name (otherwise plot all found).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Use local paths from config.py (default).")
    group.add_argument("--drive", action="store_true", help="Use Google Drive paths for metrics/plots.")
    args = parser.parse_args()

    use_drive = args.drive
    metrics_dir, plots_dir = resolve_paths(use_drive)

    metric_files = list(find_metric_files(metrics_dir))
    if args.model_name:
        metric_files = [(name, path) for name, path in metric_files if name == args.model_name]

    if not metric_files:
        print(f"No metrics files found in {metrics_dir}")
        return

    entries = []
    for name, metrics_path in metric_files:
        epochs, train_losses, dev_losses = load_loss_history(metrics_path)
        entries.append((name, epochs, train_losses, dev_losses))
        print(f"Loaded {name} from {metrics_path}")

    plt.figure(figsize=(8, 5))
    for name, epochs, train_losses, dev_losses in entries:
        plt.plot(epochs, train_losses, label=f"{name} (train)", marker="o", linestyle="-")
        plt.plot(epochs, dev_losses, label=f"{name} (dev)", marker="o", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = f"{args.model_name}_loss_curve.png" if args.model_name else "loss_curves.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=150)
    print(f"Saved combined plot to {plot_path}")


if __name__ == "__main__":
    main()
