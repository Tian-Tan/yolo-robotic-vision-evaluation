#  Downloads all the YOLOv8 models and save them in the models/ directory

import os
import urllib.request
from tqdm import tqdm

# List of YOLOv8 model weights
model_urls = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
}

# Create the 'models' folder in the root directory if it doesn't exist
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(root_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Function to download with progress bar
def download_with_progress(url, dest):
    if os.path.exists(dest):
        print(f"✅ {os.path.basename(dest)} already exists. Skipping download.")
        return
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(dest)) as bar:
        urllib.request.urlretrieve(url, dest, reporthook=lambda b, bs, t: bar.update(bs))

# Download all model weights
for model_name, url in model_urls.items():
    model_path = os.path.join(models_dir, model_name)
    download_with_progress(url, model_path)

print(f"All YOLOv8 model weights are successfully downloaded in the '{models_dir}' folder.")