# The original dataset (https://www.kaggle.com/datasets/alincijov/self-driving-cars/data) contains images for
# training and evaluation. Here, we are not using those images for training but for inference. Therefore,
# this script is used to filter out a subset of those images (4241 images for evaluation) for use for inference.

import pandas as pd
import shutil
import os
from tqdm import tqdm

# Paths
csv_file = '../labels_val.csv'
image_source_dir = '../images/' # images folder from the kaggle dataset
image_eval_dir = '../images_subset/'

# Create subset image folder
os.makedirs(image_eval_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file)

# Copy images to the evaluation folder with progress bar
for image_name in tqdm(df['frame'].unique(), desc="Copying Images", unit="file"):
    source_path = os.path.join(image_source_dir, image_name)
    dest_path = os.path.join(image_eval_dir, image_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
    else:
        print(f"Warning: {image_name} not found in {image_source_dir}")

print("Images for evaluation successfully copied!")