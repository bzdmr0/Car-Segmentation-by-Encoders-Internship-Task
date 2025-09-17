
import torch
import glob
import json
import os
from typing import Dict, List, Optional, Tuple
import time
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from main import (
    Dataset,
    get_validation_augmentation,
    SegmentationModel,
    ENCODER_LIST,
    visualize_test_predictions,
    find_manual_metrics_files,
    find_best_checkpoint_for_encoder,
    benchmark_inference,
    save_benchmark_result,
    build_dataloader,
)

DATA_DIR = "./car-segmentation.v1i.coco-segmentation/"

x_test_dir = os.path.join(DATA_DIR, "test/images")
y_test_dir = os.path.join(DATA_DIR, "test/masks")


def test_all_encoders(batch_size: int = 8):
    test_loader = build_dataloader(x_test_dir, y_test_dir, batch_size=batch_size)
    summary: List[Tuple[str, Optional[str], dict]] = []

    for encoder_id, (encoder_name, arch) in ENCODER_LIST.items():
        ckpt = find_best_checkpoint_for_encoder(encoder_name)
        if not ckpt:
            print(f"No checkpoint found for {encoder_name}, skipping.")
            continue
        print(f"\n=== Testing encoder={encoder_name} arch={arch} ===\nckpt: {ckpt}")
        try:
            model = SegmentationModel.load_from_checkpoint(ckpt, arch=arch, encoder_name=encoder_name, in_channels=3)
        except Exception as e:
            print(f"Failed to load checkpoint for {encoder_name}: {e}")
            continue

        trainer = pl.Trainer(logger=False, enable_checkpointing=False)
        metrics = trainer.test(model=model, dataloaders=test_loader, verbose=False)
        print(metrics)

        print("\nGenerating test visualizations...")
        visualize_test_predictions(model, test_loader, num_samples=3, encoder_name=encoder_name)

        metrics_path = find_manual_metrics_files(encoder_name)
        out_folder = os.path.dirname(metrics_path)

        result = benchmark_inference(model, test_loader, (encoder_name, arch))
        out_path = save_benchmark_result(result, encoder_name, out_folder)
        print(f"Saved benchmark for {encoder_name} -> {out_path}")
        summary.append((encoder_name, ckpt, result))


if __name__ == "__main__":
    test_all_encoders()
