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


def determine_best_epoch(df: pd.DataFrame, col: str, mode: str = 'max'):
    """Pick the best epoch using common validation metrics.

    Preference order: val_dataset_iou (max) -> val_f1_score (max) -> val_accuracy (max) -> val_loss (min)
    Returns (best_epoch:int|None, best_col:str|None, best_value:float|None)
    """
    if col in df.columns and df[col].notna().any():
        idx = df[col].idxmax() if mode == 'max' else df[col].idxmin()
        try:
            best_epoch = int(df.loc[idx, 'epoch']) if 'epoch' in df.columns else int(idx)
        except Exception:
            best_epoch = int(idx)
        best_value = float(df.loc[idx, col])
        return best_epoch, col, best_value
        
    return None, None, None


def plot_train_val_pairs(df: pd.DataFrame, out_dir: str, base: str, version: str):
    pairs = [
        ('train_f1_score', 'val_f1_score', 'F1-Score'),
        ('train_accuracy', 'val_accuracy', 'Accuracy'),
        ('train_dataset_iou', 'val_dataset_iou', 'IoU_dataset'),
        ('train_loss', 'val_loss', 'Loss'),
    ]
    for train_col, val_col, title in pairs:
        if title == "Loss":
            best_epoch, best_col, best_val = determine_best_epoch(df,val_col, mode='min')
        else:
            
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
        # Best epoch indicator
        subtitle = ""
        if best_epoch is not None:
            ax.axvline(best_epoch, color='crimson', linestyle='--', alpha=0.7, linewidth=1.5, label=f'best epoch ({best_epoch})')
            # highlight best point on validation curve if available
            if val_col in df.columns:
                if 'epoch' in df.columns:
                    row = df[df['epoch'] == best_epoch]
                    if not row.empty and pd.notna(row[val_col].values[0]):
                        ax.scatter(best_epoch, float(row[val_col].values[0]), color='crimson', zorder=5)
                else:
                    # when no epoch column, x is position
                    if best_epoch < len(df) and pd.notna(df.loc[best_epoch, val_col]):
                        ax.scatter(best_epoch, float(df.loc[best_epoch, val_col]), color='crimson', zorder=5)
            if best_col is not None and best_val is not None:
                subtitle = f" (best: {best_col}={best_val:.4f} @ epoch {best_epoch})"

        ax.set_title(f"{version} - {base} - {title}{subtitle}")
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
        # Sort by epoch and reset index to make plotting and best-epoch alignment robust
        df = df.sort_values('epoch').reset_index(drop=True) if 'epoch' in df.columns else df
        out_dir = os.path.dirname(p)
        base = os.path.splitext(os.path.basename(p))[0]
        version = os.path.basename(out_dir)
        plot_train_val_pairs(df, out_dir, base, version)


if __name__ == '__main__':
    main()
