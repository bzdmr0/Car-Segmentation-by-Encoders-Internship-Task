import glob
import json
import os
from typing import Optional

import torch
import glob
import json
import os
from typing import Dict, List, Optional, Tuple
import time
import pytorch_lightning as pl

from torch.utils.data import DataLoader


# Import required items from main. This assumes main.py is safe to import (training loop is guarded).
from main import (
    SegmentationModel,
    Dataset,
    get_validation_augmentation,
    ENCODER_LIST,
    x_test_dir as x_test_dir_from_main,
    y_test_dir as y_test_dir_from_main,
)

DATA_DIR = "./car-segmentation.v1i.coco-segmentation/"
x_test_dir = os.path.join(DATA_DIR, "test/images")
y_test_dir = os.path.join(DATA_DIR, "test/masks")

if not os.path.isdir(x_test_dir):
    x_test_dir = x_test_dir_from_main
if not os.path.isdir(y_test_dir):
    y_test_dir = y_test_dir_from_main


def benchmark_inference(model: pl.LightningModule, dataloader: DataLoader, encoder_info: tuple, device: str | None = None, warmup_batches: int = 3, measure_batches: int = 20):
    """Benchmark inference latency and throughput on the given dataloader.

    Returns a dict with:
      - device, batch_size, batches_measured, total_images
      - total_time_s, avg_batch_s, avg_image_s, throughput_imgs_per_s
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

def find_checkpoints_for_encoder(root: str, encoder_name: str) -> List[str]:
    """Find all checkpoint files for a given encoder under lightning_logs."""
    pattern = os.path.join(root, "**", "checkpoints", f"{encoder_name}-*.ckpt")
    return glob.glob(pattern, recursive=True)


def pick_latest(path_list: List[str]) -> Optional[str]:
    if not path_list:
        return None
    return max(path_list, key=lambda p: os.path.getmtime(p))


def build_test_loader(batch_size: int = 8) -> DataLoader:
    
    ds = Dataset(x_test_dir, y_test_dir, augmentation=get_validation_augmentation())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_encoder_list() -> Dict[int, str]:
    """Try to import ENCODER_LIST from main; fallback to known mapping if not available."""
    try:
        from main import ENCODER_LIST  # type: ignore
        if isinstance(ENCODER_LIST, dict):
            return ENCODER_LIST
    except Exception:
        pass


def get_arch_for_encoder_id(encoder_id: int) -> str:
    # Matches training logic: id 1 => SegFormer, others => DeepLabV3plus
    return "SegFormer" if encoder_id == 1 else "DeepLabV3plus"


def save_result(json_obj: dict, encoder_name: str, checkpoint_path: Optional[str], arch: str) -> str:
    # Prefer saving next to the checkpoint directory; else in lightning_logs root; else CWD.
    out_dir = None
    if checkpoint_path:
        out_dir = os.path.dirname(checkpoint_path)
    if not out_dir:
        ll_root = os.path.join(".", "lightning_logs")
        if os.path.isdir(ll_root):
            out_dir = ll_root
    if not out_dir:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"inference_benchmark_{encoder_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
    return out_path


def run():
    encoders = ENCODER_LIST
    loader = build_test_loader(batch_size=8)

    ll_root = os.path.join(".", "lightning_logs")
    if not os.path.isdir(ll_root):
        print(f"No lightning_logs directory found at {os.path.abspath(ll_root)}. Exiting.")
        return

    summary: List[Tuple[str, Optional[str], dict]] = []

    for enc_id in sorted(encoders.keys()):
        encoder_name, arch = encoders[enc_id]
        ckpts = find_checkpoints_for_encoder(ll_root, encoder_name)
        latest_ckpt = pick_latest(ckpts)

        if not latest_ckpt:
            print(f"No checkpoints found for encoder {encoder_name}. Skipping.")
            continue

        print(f"Benchmarking encoder={encoder_name} arch={arch}\n  checkpoint: {latest_ckpt}")
        try:
            model = SegmentationModel.load_from_checkpoint(
                latest_ckpt, arch=arch, encoder_name=encoder_name, in_channels=3
            )
        except Exception as e:
            print(f"Failed to load checkpoint for {encoder_name}: {e}")
            continue

        result = benchmark_inference(model, loader, (encoder_name, arch))
        out_path = save_result(result, encoder_name, latest_ckpt, arch)
        print(f"Saved benchmark for {encoder_name} -> {out_path}")
        summary.append((encoder_name, latest_ckpt, result))

    # Print final summary
    if summary:
        print("\n==== Benchmark Summary ====")
        for enc, ckpt, res in summary:
            print(f"{enc}: device={res.get('device')} batch={res.get('batch_size')} avg_image_s={res.get('avg_image_s'):.6f} throughput={res.get('throughput_imgs_per_s'):.2f} img/s")
    else:
        print("No benchmarks were produced. Ensure checkpoints exist in lightning_logs and try again.")


if __name__ == "__main__":
    run()
