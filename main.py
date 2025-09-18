import os
import cv2
import glob
from typing import Optional
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import time
from torch.utils.data import DataLoader
import json


ENCODER_LIST = {1: ("densenet121","segformer"), 2: ("resnet18","DeepLabV3plus"), 3: ("mobilenet_v2","DeepLabV3plus"), 4: ("resnet18","segformer"), 5: ("mobilenet_v2","segformer")}

ROOT = os.path.dirname(__file__)
LOGS_DIR = os.path.join(ROOT, 'lightning_logs')

def build_dataloader(x_dir: str, y_dir: str, batch_size: int = 8):
    is_train = ("train" in x_dir and "train" in y_dir)
    ds = Dataset(
        x_dir,
        y_dir,
        augmentation=get_training_augmentation() if is_train else get_validation_augmentation(),
    )
    # Shuffle only during training
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train)
    return data_loader

def save_benchmark_result(json_obj: dict, encoder_name: str, arch: str, output_folder: Optional[str]) -> str:
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"inference_benchmark_{encoder_name}-{arch}.json")
    
    # Ensure encoder name is present in the JSON object
    json_obj = dict(json_obj)
    if encoder_name and "encoder_name" not in json_obj:
        json_obj["encoder_name"] = encoder_name
        
    existing: list = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
            elif isinstance(data, dict):
                existing = [data]
        except Exception as e:
            print(f"Warning: could not read {out_path}: {e}. Recreating file.")
            existing = []
            
    # Update matching encoder entry or append if missing
    idx = next((i for i, item in enumerate(existing)
                if item.get("encoder_name") == encoder_name), None)
    if idx is None:
        existing.append(json_obj)
    else:
        existing[idx].update(json_obj)
    
     # Sort by encoder_name for readability
    existing = sorted(existing, key=lambda d: (d.get("encoder_name")))
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return out_path

def benchmark_inference(model: pl.LightningModule, dataloader: DataLoader, encoder_info: tuple, device: str | None = None, warmup_batches: int = 3, measure_batches: int = 20):
    """Benchmark inference latency and throughput on the given dataloader.

    Returns a dict with:
      - device, batch_size, batches_measured, total_images
      - total_time_s, avg_batch_s, avg_image_s, throughput_imgs_per_s
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    model = model.eval().to(device)
    encoder_name, arch = encoder_info

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0

    with torch.inference_mode():
        # Warmup
        for i, (x, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = model(x)

        # Measure
        measured = 0
        for i, (x, _) in enumerate(dataloader):
            if measured >= measure_batches:
                break
            x = x.to(device, non_blocking=True)
            _sync()
            t0 = time.perf_counter()
            _ = model(x)
            _sync()
            dt = time.perf_counter() - t0
            total_time += dt
            total_images += int(x.shape[0])
            measured += 1

    batches_measured = measured
    avg_batch_s = total_time / batches_measured if batches_measured > 0 else float('nan')
    avg_image_s = total_time / total_images if total_images > 0 else float('nan')
    throughput = total_images / total_time if total_time > 0 else float('nan')

    return {
        "device": device,
        "gpu_name": gpu_name,
        "batch_size": getattr(dataloader, "batch_size", None),
        "batches_measured": batches_measured,
        "total_images": total_images,
        "total_time_s": total_time,
        "avg_batch_s": avg_batch_s,
        "avg_image_s": avg_image_s,
        "throughput_imgs_per_s": throughput,
        "encoder_name": encoder_name,
        "architecture": arch,
    }


def find_best_weight_for_encoder(encoder_name: str, arch: str):
    pattern = os.path.join(LOGS_DIR, "**", "checkpoints", f"{encoder_name}-{arch}-*pt")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))

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


def plot_train_val_pairs(df: pd.DataFrame, out_dir: str, encoder_name: str, arch: str, version: str):
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
            best_epoch, best_col, best_val = determine_best_epoch(df,val_col)
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

        ax.set_title(f"{version} - {encoder_name} - {title}{subtitle}")
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{encoder_name}-{arch}_{title.lower()}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

def find_manual_metrics_files(encoder_name: str):
    # SegmentationModel saves as: metrics_manual_{encoder_name}.csv
    pattern = os.path.join(LOGS_DIR, 'version_*', f'metrics_manual_{encoder_name}.csv')
    return sorted(set(glob.glob(pattern)))

class Dataset(BaseDataset):
    """car-segmentation-dataset Dataset. Read images, apply augmentation transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_value = 8  # cars

        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        # BGR-->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # extract certain classes from mask
        mask = np.stack([mask == self.class_value], axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
    
# training set images augmentation
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)

class SegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, **kwargs):
        super().__init__()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            **kwargs,
        )
        self.encoder_name = encoder_name
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        # Dice loss for binary segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.t_max = None  # to be set later for LR scheduler

        # Manual per-epoch history for plotting later
        self.history = {"train": [], "val": [], "test": []}
    
    def t_max_setter(self, t_max):
        self.t_max = t_max
            
    def forward(self, image):
        # Normalize the input image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def shared_step(self, batch, stage):
        image, mask = batch
        
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        
        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 84x84 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        assert mask.ndim == 4
        
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        
        loss = self.loss_fn(logits_mask, mask)
        
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # Average loss over the epoch for this stage
        loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        # Calculate F1-Score (Dice coefficient)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        # Calculate Accuracy
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_accuracy": accuracy,
            f"{stage}_loss": loss_mean,
        }
        
        self.log_dict(metrics, prog_bar=True)

        # Store manual history as scalars
        metrics_scalar = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics.items()}
        metrics_scalar["epoch"] = int(self.current_epoch)
        self.history[stage].append(metrics_scalar)
        
    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        return
    
    def validation_step(self, batch, batch_idx):
        val_loss_info = self.shared_step(batch, "val")
        self.validation_step_outputs.append(val_loss_info)
        return val_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()
        return
    
    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        return

    def _save_history(self):
        # Merge train/val/test on epoch and save to CSV in the logger directory
        frames = []
        for stage in ["train", "val", "test"]:
            if self.history[stage]:
                frames.append(pd.DataFrame(self.history[stage]))
        if not frames:
            return
        df = frames[0]
        for f in frames[1:]:
            df = pd.merge(df, f, on="epoch", how="outer")
        df = df.sort_values("epoch").reset_index(drop=True)
        log_dir = None
        try:
            log_dir = self.trainer.logger.log_dir
        except Exception:
            pass
        if not log_dir:
            log_dir = self.trainer.default_root_dir
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"metrics_manual_{self.encoder_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved manual metrics to {out_path}")

    def on_fit_end(self):
        # Save after training (contains train and val)
        self._save_history()
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-5) #TODO: optimize T_max setting
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return

def visualize_test_predictions(model, test_loader, num_samples=3, encoder_name="", arch=""):
    """Visualize test predictions with highlighted differences"""
    # Get a batch of test data
    images, masks = next(iter(test_loader))
    with torch.inference_mode():
        model.eval()
        logits = model(images)
    pred_masks = logits.sigmoid()
    ll_root = os.path.join(".", "lightning_logs")
    visualization_folder = os.path.join(ll_root,f'{encoder_name}-{arch}_visualizations/')
    os.makedirs(visualization_folder, exist_ok=True)

    # Create visualizations a. Yanyana gösterim:
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(min(num_samples, len(images))):
        image = images[idx].numpy().transpose(1, 2, 0)
        true_mask = masks[idx].numpy().squeeze()
        pred_mask = pred_masks[idx].numpy().squeeze()
        
        # Original Image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")
        
        # True Mask
        axes[idx, 1].imshow(true_mask, cmap='gray')
        axes[idx, 1].set_title("True Mask")
        axes[idx, 1].axis("off")
        
        # Predicted Mask
        axes[idx, 2].imshow(pred_mask, cmap='gray')
        axes[idx, 2].set_title("Predicted Mask")
        axes[idx, 2].axis("off")
        
        # Highlighted Difference
        # True positive (correct predictions) - green
        # False positive (wrong predictions) - red
        # False negative (missed predictions) - blue
        
        diff_mask = np.zeros((*pred_mask.shape, 3))
        
        # True positives: both true and predicted are 1
        tp = (true_mask > 0.5) & (pred_mask > 0.5)
        diff_mask[tp] = [0, 1, 0]  # Green
        
        # False positives: predicted is 1 but true is 0
        fp = (true_mask <= 0.5) & (pred_mask > 0.5)
        diff_mask[fp] = [1, 0, 0]  # Red
        
        # False negatives: true is 1 but predicted is 0
        fn = (true_mask > 0.5) & (pred_mask <= 0.5)
        diff_mask[fn] = [0, 0, 1]  # Blue
        
        true_count = np.count_nonzero(true_mask > 0.5)
        # Count colors in diff_mask
        is_green = np.all(diff_mask == [0, 1, 0], axis=-1)
        is_red   = np.all(diff_mask == [1, 0, 0], axis=-1)
        is_blue  = np.all(diff_mask == [0, 0, 1], axis=-1)
        green_count = int(is_green.sum())
        red_count   = int(is_red.sum())
        blue_count  = int(is_blue.sum())
        total_px    = diff_mask.shape[0] * diff_mask.shape[1]
        
        tp_percent = (green_count / true_count * 100) if true_count > 0 else 0.0
        fn_percent = (blue_count / true_count * 100) if true_count > 0 else 0.0
        
        axes[idx, 3].imshow(diff_mask)
        axes[idx, 3].set_title(f"Highlighted Difference\n(Green: TP({tp_percent:.1f}%), Red: FP, Blue: FN({fn_percent:.1f}%))")
        axes[idx, 3].axis("off")
    
    plt.tight_layout()
    H_D_img = os.path.join(visualization_folder, f'Highlighted_Differences.png')
    plt.savefig(H_D_img, dpi=300, bbox_inches='tight')
    
    # Create visualizations b. Yanyana gösterim:
    fig, axes = plt.subplots(num_samples, 3, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(min(num_samples, len(images))):
        image = images[idx].numpy().transpose(1, 2, 0)
        true_mask = masks[idx].numpy().squeeze()
        pred_mask = pred_masks[idx].numpy().squeeze()
        
        # Original Image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")
        
        # Predicted Mask
        axes[idx, 1].imshow(pred_mask, cmap='gray')
        axes[idx, 1].set_title("Predicted Mask")
        axes[idx, 1].axis("off")
        
        # Highlighted Difference
        # True positive (correct predictions) - green
        # False positive (wrong predictions) - red
        # False negative (missed predictions) - blue
        
        org_and_pred = image.copy()
        mask = pred_mask <= 0.5
        org_and_pred[mask] = [0,0,0]
        
        axes[idx, 2].imshow(org_and_pred)
        axes[idx, 2].set_title("Original + Predicted Mask")
        axes[idx, 2].axis("off")
    
    plt.tight_layout()
    O_P_img = os.path.join(visualization_folder, f'Original_and_Predicted_Mask.png')
    plt.savefig(O_P_img, dpi=300, bbox_inches='tight')

# -----------------------------
# Timing utilities
# -----------------------------

class TimingCallback(pl.Callback):
    """Measure training/validation epoch durations and total fit time.

    Saves a CSV to the logger directory with columns:
      - epoch
      - train_epoch_seconds
      - val_epoch_seconds (if validation runs)
      - fit_total_seconds (same value on each row for convenience)
    """

    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder_name = encoder_name
        self.fit_start = None
        self.epoch_start = None
        self.val_start = None
        self.rows = []  # list of dicts per epoch

    def on_fit_start(self, trainer, pl_module):
        self.fit_start = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        # Training epoch duration
        if self.epoch_start is not None:
            train_sec = time.perf_counter() - self.epoch_start
        else:
            train_sec = float('nan')
        self.rows.append({
            "epoch": int(trainer.current_epoch),
            "train_epoch_seconds": float(train_sec),
            "val_epoch_seconds": float('nan'),  # will be filled in validation end
        })
        self.epoch_start = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_start is not None and self.rows:
            val_sec = time.perf_counter() - self.val_start
            self.rows[-1]["val_epoch_seconds"] = float(val_sec)
        self.val_start = None

    def on_fit_end(self, trainer, pl_module):
        fit_total_seconds = None
        if self.fit_start is not None:
            fit_total_seconds = time.perf_counter() - self.fit_start
        # Save to logger dir
        log_dir = None
        try:
            log_dir = trainer.logger.log_dir
        except Exception:
            pass
        if not log_dir:
            log_dir = trainer.default_root_dir
        os.makedirs(log_dir, exist_ok=True)
        # Write CSV
        df = pd.DataFrame(self.rows)
        if df.empty:
            # Ensure at least a summary row exists
            df = pd.DataFrame([{"epoch": -1, "train_epoch_seconds": float('nan'), "val_epoch_seconds": float('nan')}])
        df["fit_total_seconds"] = float(fit_total_seconds) if fit_total_seconds is not None else float('nan')
        out_csv = os.path.join(log_dir, f"timings_{self.encoder_name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved training timings to {out_csv}")