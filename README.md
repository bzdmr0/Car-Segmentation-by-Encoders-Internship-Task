# Car-Segmentation-by-Encoders-Internship-Task
Vehicle segmentation using the segmentation-models-pytorch library and comparing the results with different encoders.

## Dependencies
This project requires the following Python packages (installed in a virtual environment):

- certifi==2025.8.3
- charset-normalizer==3.4.3
- colorama==0.4.6
- filelock==3.19.1
- fsspec==2025.9.0
- huggingface-hub==0.34.4
- idna==3.10
- Jinja2==3.1.6
- MarkupSafe==3.0.2
- mpmath==1.3.0
- networkx==3.4.2
- numpy==2.2.6
- packaging==25.0
- pillow==11.3.0
- PyYAML==6.0.2
- requests==2.32.5
- safetensors==0.6.2
- segmentation_models_pytorch==0.5.0
- sympy==1.13.1
- timm==1.0.19
- torch==2.5.1+cu121
- torchaudio==2.5.1+cu121
- torchvision==0.20.1+cu121
- tqdm==4.67.1
- typing_extensions==4.15.0
- urllib3==2.5.0

To install these dependencies, create a virtual environment and run:
```bash
pip install -r requirements.txt
```

Note: The PyTorch packages (torch, torchvision, torchaudio) are installed with CUDA 12.1 support for GPU acceleration.


## Loading data
For this example we will use CamVid dataset. It is a set of:

- train images + segmentation masks
- validation images + segmentation masks
- test images + segmentation masks

All images have 320 pixels height and 480 pixels width. For more inforamtion about dataset visit 
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

# To download Data
```bash
git clone https://github.com/alexgkendall/SegNet-Tutorial ./data
```
