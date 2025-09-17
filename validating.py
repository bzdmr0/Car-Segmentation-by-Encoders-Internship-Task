import os
import glob
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from main import (
    Dataset,
    get_validation_augmentation,
    SegmentationModel,
    ENCODER_LIST,
    find_best_checkpoint_for_encoder,
    build_dataloader,
)

DATA_DIR = "./car-segmentation.v1i.coco-segmentation/"

x_valid_dir = os.path.join(DATA_DIR, "valid/images")
y_valid_dir = os.path.join(DATA_DIR, "valid/masks")


def validate_all_encoders(batch_size: int = 8):
    val_loader = build_dataloader(x_valid_dir, y_valid_dir, batch_size=batch_size)

    for encoder_id, (encoder_name, arch) in ENCODER_LIST.items():
        ckpt = find_best_checkpoint_for_encoder(encoder_name)
        if not ckpt:
            print(f"No checkpoint found for {encoder_name}, skipping.")
            continue
        print(f"\n=== Validating encoder={encoder_name} arch={arch} ===\nckpt: {ckpt}")
        try:
            model = SegmentationModel.load_from_checkpoint(ckpt, arch=arch, encoder_name=encoder_name, in_channels=3)
        except Exception as e:
            print(f"Failed to load checkpoint for {encoder_name}: {e}")
            continue

        trainer = pl.Trainer(logger=False, enable_checkpointing=False)
        metrics = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
        print(metrics)


if __name__ == "__main__":
    validate_all_encoders()

