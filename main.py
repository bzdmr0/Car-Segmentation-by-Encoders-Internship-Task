import os
import cv2

import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = "./car-segmentation.v1i.coco-segmentation/"

x_train_dir = os.path.join(DATA_DIR, "train/images")
y_train_dir = os.path.join(DATA_DIR, "train/masks")

x_valid_dir = os.path.join(DATA_DIR, "valid/images")
y_valid_dir = os.path.join(DATA_DIR, "valid/masks")

x_test_dir = os.path.join(DATA_DIR, "test/images")
y_test_dir = os.path.join(DATA_DIR, "test/masks")

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation transformations.

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

    def __len__(self):
        return len(self.ids)
    
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image)
    plt.show()
    
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
    
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
)

train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False )
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False )


EPOCHS = 20
T_MAX = EPOCHS * len(train_loader)

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

        # Manual per-epoch history for plotting later
        self.history = {"train": [], "val": [], "test": []}
    
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

    def on_test_end(self):
        # Save again after test to include test_* columns if present
        self._save_history()
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return

def visualize_test_predictions(model, test_loader, num_samples=3, encoder_name=""):
    """Visualize test predictions with highlighted differences"""
    # Get a batch of test data
    images, masks = next(iter(test_loader))
    with torch.inference_mode():
        model.eval()
        logits = model(images)
    pred_masks = logits.sigmoid()
    
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
        
        axes[idx, 3].imshow(diff_mask)
        axes[idx, 3].set_title("Highlighted Difference\n(Green: TP, Red: FP, Blue: FN)")
        axes[idx, 3].axis("off")
    
    plt.tight_layout()
    plt.savefig(f'./differences/last/Highlighted_Differences_{encoder_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    plt.savefig(f'./differences/last/Original_and_Predicted_Mask_{encoder_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


#encoder = int(input("For DenseNet121 encoder (1), MobileNetV2 encoder (2), ResNet18 encoder (3) : "))
#
encoderList = {1: "densenet121", 2: "resnet18", 3: "mobilenet_v2"}

for encoder in [2]:
    encoder_name = encoderList[encoder]
    if encoder == 1:
        model = SegmentationModel(
            arch="SegFormer",
            encoder_name=encoder_name,
            in_channels=3,
        )
    else:
        model = SegmentationModel(
            arch="DeepLabV3plus",
            encoder_name=encoder_name,
            in_channels=3,
        )

    # Callbacks: save best (by val_dataset_iou) and last checkpoints; optionally stop early
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dataset_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        # Keep default directory under logger: lightning_logs/version_*/checkpoints
        # Optional: customize filename below
        filename=f"{encoder_name}-{{epoch:02d}}-{{val_dataset_iou:.4f}}",
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(monitor="val_dataset_iou", mode="max", patience=3)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop],
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Report best checkpoint for this encoder
    print(f"Best checkpoint for {encoder_name}: {checkpoint_callback.best_model_path}")
    print(f"Best {encoder_name} val_dataset_iou: {checkpoint_callback.best_model_score}")

    # Validate and Test using the best checkpoint
    try:
        valid_metrics = trainer.validate(ckpt_path="best", dataloaders=valid_loader, verbose=False)
    except Exception:
        # Fallback if 'best' alias not supported
        valid_metrics = trainer.validate(model=model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    try:
        test_metrics = trainer.test(ckpt_path="best", dataloaders=test_loader, verbose=False)
    except Exception:
        test_metrics = trainer.test(model=model, dataloaders=test_loader, verbose=False)
    print(test_metrics)


# Enhanced test visualization with highlighted differences
    print("\nGenerating test visualizations...")
    visualize_test_predictions(model, test_loader, num_samples=3, encoder_name=encoderList[encoder])