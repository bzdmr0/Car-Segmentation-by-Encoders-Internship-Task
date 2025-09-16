import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

ROOT = os.path.dirname(__file__)
LOGS_DIR = os.path.join(ROOT, 'lightning_logs')


def find_manual_metrics_files(base_dir: str):
    pattern = os.path.join(base_dir, 'version_*', 'metrics_manual_*.csv')
    return sorted(set(glob.glob(pattern)))


def plot_train_val_pairs(df: pd.DataFrame, out_dir: str, base: str, version: str):
    pairs = [
        ('train_f1_score', 'val_f1_score', 'F1-Score'),
        ('train_accuracy', 'val_accuracy', 'Accuracy'),
        ('train_dataset_iou', 'val_dataset_iou', 'IoU_dataset'),
        ('train_loss', 'val_loss', 'Loss'),
    ]
    for train_col, val_col, title in pairs:
        present = [c for c in (train_col, val_col) if c in df.columns]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        if 'epoch' in df.columns:
            x = df['epoch']
        else:
            x = range(len(df))
        if train_col in df.columns:
            ax.plot(x, df[train_col], label='train', marker='o')
        if val_col in df.columns:
            ax.plot(x, df[val_col], label='val', marker='o')
        ax.set_title(f"{version} - {base} - {title}")
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{base}_{title.lower()}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    paths = find_manual_metrics_files(LOGS_DIR)
    if not paths:
        print(f"No manual metrics files found under {LOGS_DIR}")
        return
    print(f"Found {len(paths)} manual metrics files")

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue
        df = df.sort_values('epoch') if 'epoch' in df.columns else df
        out_dir = os.path.dirname(p)
        base = os.path.splitext(os.path.basename(p))[0]
        version = os.path.basename(out_dir)
        plot_train_val_pairs(df, out_dir, base, version)


if __name__ == '__main__':
    main()
