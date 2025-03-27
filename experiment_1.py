# Experiment 1: Model vs latency (average time taken per image)

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ultralytics import YOLO
from tqdm import tqdm

# === Configuration ===
models = [
    'models/yolov8n.pt',
    'models/yolov8s.pt',
    'models/yolov8m.pt',
    'models/yolov8l.pt',
    'models/yolov8x.pt'
]
images_dir = 'images'  # directory containing test images
imgsz = 640
image_files = sorted([
    os.path.join(images_dir, f) for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# === Main experiment ===
results_data = []
latency_data = []

for model_path in tqdm(models, desc="Evaluating Models", unit="model"):
    model = YOLO(model_path)
    per_image_latencies = []

    for idx, image_path in enumerate(tqdm(image_files, desc=f"Inferencing {os.path.basename(model_path)}", leave=False, unit="img")):
        start = time.time()
        model.predict(source=image_path, imgsz=imgsz, save=False, verbose=False)
        end = time.time()

        latency = (end - start) * 1000  # ms
        per_image_latencies.append(latency)
        latency_data.append({
            'Model': model_path.split('/')[-1],
            'ImageIndex': idx,
            'Latency (ms)': latency
        })

    avg_latency = sum(per_image_latencies) / len(per_image_latencies)
    results_data.append({
        'Model': model_path.split('/')[-1],
        'Average Latency (ms/image)': avg_latency
    })

# === Save results ===
results_df = pd.DataFrame(results_data)
results_df.to_csv('experiment_1_avg_latency.csv', index=False)

latency_df = pd.DataFrame(latency_data)
latency_df.to_csv('experiment_1_per_image_latency.csv', index=False)

# === Plot 1: Average Latency per Image vs Model ===
plt.figure(figsize=(8, 5))
plt.bar(results_df['Model'], results_df['Average Latency (ms/image)'])
plt.title('Average Latency per Image vs Model')
plt.ylabel('Latency (ms/image)')
plt.xlabel('Model')
plt.grid(True)
plt.tight_layout()
plt.savefig('avg_latency_per_model.png')
plt.show()

# === Plot 2: Latency per Image in Inference Order (excluding first 10 images and showing outliers) ===
plt.figure(figsize=(10, 6))
colors = cm.get_cmap('tab10', len(latency_df['Model'].unique()))  # use distinct colors

for i, model_name in enumerate(latency_df['Model'].unique()):
    df_model = latency_df[latency_df['Model'] == model_name]
    df_model = df_model[df_model['ImageIndex'] > 9]  # Skip first 10 images

    avg_latency = df_model['Latency (ms)'].mean()
    threshold = 1.5 * avg_latency

    # Split into normal and outlier points
    normal = df_model[df_model['Latency (ms)'] <= threshold]
    outliers = df_model[df_model['Latency (ms)'] > threshold]

    color = colors(i)

    # Plot line for normal points
    plt.plot(
        normal['ImageIndex'],
        normal['Latency (ms)'],
        label=f'{model_name}',
        linewidth=1,
        color=color
    )

    # Plot dots for outliers on top
    plt.scatter(
        outliers['ImageIndex'],
        outliers['Latency (ms)'],
        color=color,
        s=10,
        label=f'{model_name} outliers',
        zorder=5  # ensure dots are on top
    )

plt.title('Latency per Image vs Inference Order (Excl. First 10 Images)')
plt.xlabel('Image Index')
plt.ylabel('Latency (ms)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('latency_per_image.png')
plt.show()