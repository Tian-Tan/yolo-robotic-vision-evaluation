# Experiment 1: Accuracy (Model size) vs latency (average time taken per image)

import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# List of YOLOv8 models for evaluation
models = [
    'models/yolov8n.pt',
    # 'models/yolov8s.pt',
    # 'models/yolov8m.pt',
    # 'models/yolov8l.pt',
    # 'models/yolov8x.pt'
]

# Dictionary to store evaluation results
results_data = []

# Evaluate each model
for model_path in models:
    model = YOLO(model_path)

    # Measure inference latency
    metrics = model.val(
        data='self_driving.yaml',
        imgsz=640,
        classes=[0, 1, 2, 7, 9]  # Focus only on the 5 relevant classes
    )
    
    # latency is the sum of the times for these metrics: preprocess, inference, loss and postprocess
    # print(f"Speeds for {model_path}:", metrics.speed)
    latency = sum(metrics.speed.values())
    # print(f"Latency (sum of speed values) for {model_path}:", latency)

    # Store results
    results_data.append({
        'Model': model_path.split('/')[-1],
        'Speeds': metrics.speed,
        'Latency': latency,
        'mAP@0.5': metrics.box.map50,           
        'mAP@0.5:0.95': metrics.box.map,        
        'Precision': metrics.box.p,     
        'Recall': metrics.box.r,           
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results_data)
results_df.to_csv('experiment_1_results.csv', index=False)
print("Evaluation results saved to 'experiment_1_results.csv'")

# Plot Accuracy vs Latency
plt.figure(figsize=(8, 5))
plt.plot(results_df['Latency'], results_df['mAP@0.5'], marker='o', label='mAP@0.5')
plt.plot(results_df['Latency'], results_df['mAP@0.5:0.95'], marker='o', label='mAP@0.5:0.95')
plt.title('YOLOv8 Accuracy vs Latency')
plt.xlabel('Latency (ms/image)')
plt.ylabel('Accuracy (mAP)')
plt.grid(True)
plt.legend()
plt.show()