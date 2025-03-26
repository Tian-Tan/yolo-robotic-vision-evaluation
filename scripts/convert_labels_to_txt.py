# Convert labels_val_modified.csv to a folder of .txt files in YOLOv8 format

import pandas as pd
import os
from tqdm import tqdm
import cv2

# Paths
script_dir = os.path.dirname(__file__)
csv_file = os.path.join(script_dir, '..', 'labels_val_modified.csv')
images_dir = os.path.join(script_dir, '..', 'images')
labels_dir = os.path.join(script_dir, '..', 'labels')
os.makedirs(labels_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

# Convert each unique image
for image_name in tqdm(df['frame'].unique(), desc="Converting Labels", unit="file"):
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_name} not found. Skipping.")
        continue

    # Read image dimensions using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading {image_name}. Skipping.")
        continue
    img_height, img_width = img.shape[:2]

    # Write label file
    label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        for _, row in df[df['frame'] == image_name].iterrows():
            x_center = ((row['xmin'] + row['xmax']) / 2) / img_width
            y_center = ((row['ymin'] + row['ymax']) / 2) / img_height
            width = (row['xmax'] - row['xmin']) / img_width
            height = (row['ymax'] - row['ymin']) / img_height
            yolo_line = f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            f.write(yolo_line + '\n')

print("Labels successfully converted to YOLOv8 format!")