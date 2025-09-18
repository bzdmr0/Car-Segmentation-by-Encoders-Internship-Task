import os
import pytorch_lightning as pl
import torch

from main import (
    SegmentationModel,
    ENCODER_LIST,
    find_best_weight_for_encoder,
    build_dataloader,
)

DATA_DIR = "./car-segmentation-dataset/"

x_valid_dir = os.path.join(DATA_DIR, "valid/images")
y_valid_dir = os.path.join(DATA_DIR, "valid/masks")


def validate_all_encoders(batch_size: int = 8):
    val_loader = build_dataloader(x_valid_dir, y_valid_dir, batch_size=batch_size)

    for encoder_id, (encoder_name, arch) in ENCODER_LIST.items():
        weight_path = find_best_weight_for_encoder(encoder_name, arch)
        if not weight_path:
            print(f"No weight found for {encoder_name}, skipping.")
            continue
        print(f"\n=== Validating encoder={encoder_name} arch={arch} ===\nweight path: {weight_path}")
        try:
            model = SegmentationModel.load_from_checkpoint(weight_path, arch=arch, encoder_name=encoder_name, in_channels=3)
        except Exception as e:
            print(f"Failed to load weight for {encoder_name}: {e}")
            continue

        trainer = pl.Trainer(logger=False, enable_checkpointing=False)
        metrics = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
        print(metrics)

def ckpt_to_pth(ckpt_path: str, out_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Remove Lightning prefixes like "model." if needed
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    torch.save(cleaned, out_path)

# Example


if __name__ == "__main__":
    validate_all_encoders()
    
    #encoder_name, arch = ENCODER_LIST[1]  # Example for encoder_id 1
    #ckpt = find_best_checkpoint_for_encoder(encoder_name)
    #ckpt_to_pth(ckpt, "best_weights.pt")
    