import kagglehub

# Download latest version
path = kagglehub.dataset_download("intelecai/car-segmentation")

print("Path to dataset files:", path)