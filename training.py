import os
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Reuse definitions from main.py (safe to import, guarded main())
from main import (
    SegmentationModel,
    TimingCallback,
    ENCODER_LIST,
    LOGS_DIR,
    find_manual_metrics_files,
    plot_train_val_pairs,
    build_dataloader,
)

DATA_DIR = "./car-segmentation.v1i.coco-segmentation/"

x_train_dir = os.path.join(DATA_DIR, "train/images")
y_train_dir = os.path.join(DATA_DIR, "train/masks")

x_valid_dir = os.path.join(DATA_DIR, "valid/images")
y_valid_dir = os.path.join(DATA_DIR, "valid/masks")

EPOCHS = 20


        
def train_all_encoders(batch_size: int = 8):
    train_loader = build_dataloader(x_train_dir, y_train_dir, batch_size=batch_size)
    val_loader = build_dataloader(x_valid_dir, y_valid_dir, batch_size=batch_size)

    for encoder_id, (encoder_name, arch) in ENCODER_LIST.items():
        print(f"\n=== Training encoder={encoder_name} arch={arch} ===")
        model = SegmentationModel(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=3,
        )
        
        model.t_max_setter(EPOCHS * len(train_loader))  # Set T_max for LR scheduler

        checkpoint_callback = ModelCheckpoint(
            monitor="val_dataset_iou",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename=f"{encoder_name}-{{epoch:02d}}-{{val_dataset_iou:.4f}}",
            auto_insert_metric_name=False,
        )
        early_stop = EarlyStopping(monitor="val_dataset_iou", mode="max", patience=3)
        timing_cb = TimingCallback(encoder_name=encoder_name)

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, early_stop, timing_cb],
            enable_checkpointing=True,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print(f"Best checkpoint for {encoder_name}: {checkpoint_callback.best_model_path}")
        print(f"Best {encoder_name} val_dataset_iou: {checkpoint_callback.best_model_score}")

        metrics_path = find_manual_metrics_files(encoder_name)
        if not metrics_path:
            print(f"No manual metrics files found under {LOGS_DIR}")
            return
        print(f"Found {len(metrics_path)} manual metrics files")
        
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Failed to read {metrics_path}: {e}")
            continue
        # Sort by epoch and reset index to make plotting and best-epoch alignment robust
        df = df.sort_values('epoch').reset_index(drop=True) if 'epoch' in df.columns else df
        out_dir = os.path.dirname(metrics_path)
        base = os.path.splitext(os.path.basename(metrics_path))[0]
        version = os.path.basename(out_dir)
        plot_train_val_pairs(df, out_dir, base, version)


if __name__ == "__main__":
    train_all_encoders()
