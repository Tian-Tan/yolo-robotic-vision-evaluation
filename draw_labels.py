import cv2
import os
import yaml
import random

# === Configuration ===
images_dir = 'images'
labels_dir = 'labels'
yaml_path = 'self_driving.yaml'
num_images = 3
random_seed = 42  # Set your seed here

# === Load class names from YAML ===
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
    raw_names = data.get('names', {})

# Handle if class names are a list or dict
if isinstance(raw_names, list):
    class_names = {i: name for i, name in enumerate(raw_names)}
else:
    class_names = {int(k): v for k, v in raw_names.items()}

# === Set seed for reproducibility ===
random.seed(random_seed)

# === Filter and sample image files ===
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
selected_files = random.sample(image_files, min(num_images, len(image_files)))

# === Draw boxes and display ===
for img_file in selected_files:
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    if image is None or not os.path.exists(label_path):
        continue

    height, width = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, cx, cy, w, h = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO format to pixel coords
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            label = class_names[class_id] if class_id in class_names else str(class_id)

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Labeled Image', image)
    print(f"Showing: {img_file} — press any key to continue...")
    cv2.waitKey(0)

cv2.destroyAllWindows()
