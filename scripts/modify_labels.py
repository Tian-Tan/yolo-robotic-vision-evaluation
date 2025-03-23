# Maps the labels of the dataset to the labels of COCO which was used to train YOLOv8

import os
import pandas as pd

# Mapping of custom class IDs to COCO class IDs
class_mapping = {
    1: 2,   # car → car (COCO ID: 2)
    2: 7,   # truck → truck (COCO ID: 7)
    3: 0,   # pedestrian → person (COCO ID: 0)
    4: 1,   # bicyclist → bicycle (COCO ID: 1)
    5: 9    # light → traffic light (COCO ID: 9)
}

# Load the labels CSV
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'labels_val.csv')
df = pd.read_csv(csv_path)

# Map custom class IDs to COCO class IDs
df['class_id'] = df['class_id'].map(class_mapping)

# Save the modified labels
df.to_csv(os.path.join(script_dir, '..', 'labels_val_modified.csv'), index=False)
print("Labels successfully converted to COCO format!")