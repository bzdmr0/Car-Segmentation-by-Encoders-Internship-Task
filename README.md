# Car Segmentation by Encoders - Internship Task

This project implements semantic segmentation for car detection using the segmentation-models-pytorch library. The project compares the performance of different encoder architectures with various segmentation models to achieve optimal car segmentation results.

## Project Overview

The project trains and evaluates multiple encoder-decoder combinations for car segmentation:

- **Encoder Models**: DenseNet-121, ResNet-18, MobileNet-V2
- **Decoder Models**: SegFormer, DeepLabV3Plus
- **Task**: Binary segmentation (car vs. background)

### Model Configurations

The project tests 5 different encoder-decoder combinations:

1. DenseNet-121 + SegFormer
2. ResNet-18 + DeepLabV3Plus  
3. MobileNet-V2 + DeepLabV3Plus
4. ResNet-18 + SegFormer
5. MobileNet-V2 + SegFormer

## Dataset

This project uses a custom car segmentation dataset with:

- **Total Images**: 949 images
- **Training Set**: 498 images  
- **Validation Set**: 161 images
- **Test Set**: 290 images
- **Image Dimensions**: 480×360 pixels
- **Format**: PNG images with corresponding PNG masks
- **Annotation**: COCO Segmentation format available

### Dataset Structure
```
car-segmentation-dataset/
├── train/
│   ├── images/     # Training images
│   └── masks/      # Training masks
├── valid/
│   ├── images/     # Validation images  
│   └── masks/      # Validation masks
├── test/
│   ├── images/     # Test images
│   └── masks/      # Test masks
└── _annotations.coco.json  # COCO format annotations
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.1 (for GPU acceleration)
- Internet connection (for downloading pretrained encoder weights)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bzdmr0/Car-Segmentation-by-Encoders-Internship-Task.git
cd Car-Segmentation-by-Encoders-Internship-Task
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **PyTorch**: 2.5.1+cu121 (with CUDA support)
- **Lightning**: 2.5.5 (for training framework)
- **segmentation-models-pytorch**: 0.5.0 (segmentation models)
- **Albumentations**: 2.0.8 (data augmentation)
- **OpenCV**: 4.12.0.88 (image processing)
- **Pandas**: 2.3.2 (metrics tracking)
- **Matplotlib**: 3.10.6 (visualization)

## Usage

### Training

Train all encoder-decoder combinations:
```bash
python training.py
```

This will:
- Train each of the 5 model configurations for 20 epochs
- Apply early stopping based on validation IoU
- Save best model checkpoints
- Generate training plots and metrics
- Log results to `lightning_logs/`

### Testing

Test all trained models:
```bash
python testing.py
```

This will:
- Load best checkpoints for each model
- Evaluate on test dataset
- Generate prediction visualizations
- Benchmark inference speed
- Save test results and performance metrics

### Validation

Validate models on validation set:
```bash
python validating.py
```

This script evaluates the trained models on the validation dataset.

## Architecture & Design

This project follows a modular design pattern:

- **main.py** - Core framework with PyTorch Lightning model, dataset handling, and utilities
- **training.py** - Automated training pipeline for all encoder combinations  
- **testing.py** - Comprehensive testing and benchmarking suite
- **validating.py** - Validation pipeline for model evaluation
- **Separation of Concerns** - Each script has a specific purpose and can be run independently

### Core Components (main.py)

- **SegmentationModel**: PyTorch Lightning module with training/validation logic
- **Dataset**: Custom dataset class for loading images and masks
- **TimingCallback**: Callback for tracking training/validation time
- **Data Augmentation**: Albumentations-based augmentation pipeline
- **Utilities**: Functions for checkpointing, plotting, and benchmarking

## Project Structure

```
├── main.py              # Core classes and utilities
├── training.py          # Training script for all models  
├── testing.py           # Testing and evaluation script
├── validating.py        # Validation script
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── car-segmentation-dataset/  # Dataset directory
└── lightning_logs/      # Training logs and checkpoints (created during training)
```

## Features

### Data Augmentation
The training pipeline includes comprehensive data augmentation:
- Horizontal flip (50% probability)
- Scale/rotate/shift transformations  
- Random cropping to 320×320
- Gaussian noise injection
- Perspective transformation
- Brightness/contrast adjustments
- Blur and sharpening effects

### Metrics Tracking
- **IoU (Intersection over Union)**: Primary metric for segmentation quality
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Pixel-level accuracy
- **Training/Validation Loss**: Cross-entropy loss tracking
- **Inference Speed**: Benchmarking for deployment considerations

### Visualization
- Training/validation curves for all metrics
- Test prediction visualizations showing:
  - Original images
  - Ground truth masks  
  - Model predictions
  - Difference highlighting

### Model Checkpointing
- Automatic saving of best models based on validation IoU
- Early stopping to prevent overfitting
- Checkpoint format: `.pt` files for easy deployment

## Results

After training, results are available in `lightning_logs/` including:
- Model checkpoints (`best_*.pt`)
- Training metrics CSV files
- Visualization plots (Loss, IoU, F1-Score, Accuracy curves)
- Test benchmark results with inference timing
- Prediction visualizations

## Technical Details

### Training Configuration
- **Optimizer**: Adam with learning rate 2e-4
- **Scheduler**: Cosine Annealing LR with T_max based on epochs
- **Loss Function**: Cross-entropy loss
- **Batch Size**: 8 (configurable)
- **Max Epochs**: 20 with early stopping (patience=5)
- **Image Processing**: Resize/padding to make divisible by 32

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (tested with CUDA 12.1)
- **Memory**: ~4GB GPU memory for batch size 8
- **Storage**: ~2GB for dataset + model checkpoints

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Dataset Attribution

Dataset provided via Roboflow Universe:
- **Source**: https://universe.roboflow.com/visea/car-segmentation-vacre  
- **License**: CC BY 4.0
- **Format**: COCO Segmentation
