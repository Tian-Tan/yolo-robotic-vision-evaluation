# Convert labels_val_modified.csv to a folder of .txt files in YOLOv8 format

import pandas as pd
import os
from tqdm import tqdm

# Paths
script_dir = os.path.dirname(__file__)
csv_file = os.path.join(script_dir, '..', 'labels_val_modified.csv')
labels_dir = os.path.join(script_dir, '..', 'labels/')
os.makedirs(labels_dir, exist_ok=True)

# Conversion function for YOLO format
def convert_to_yolo(row, img_width=1280, img_height=720):
    x_center = ((row['xmin'] + row['xmax']) / 2) / img_width
    y_center = ((row['ymin'] + row['ymax']) / 2) / img_height
    width = (row['xmax'] - row['xmin']) / img_width
    height = (row['ymax'] - row['ymin']) / img_height
    return f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Read CSV and create .txt labels with a progress bar
df = pd.read_csv(csv_file)

# Add progress bar to track conversion progress
for image_id in tqdm(df['frame'].unique(), desc="Converting Labels", unit="file"):
    label_path = os.path.join(labels_dir, image_id.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        for _, row in df[df['frame'] == image_id].iterrows():
            yolo_label = convert_to_yolo(row)
            f.write(yolo_label + '\n')

print("Labels successfully converted to YOLOv8 format!")