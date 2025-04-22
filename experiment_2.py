# Experiment 2: Accuracy (Model size) vs power usage (net power used by process)

import os
import time
import psutil
import threading
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

TDP = 28  # 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz

class PowerMonitor(threading.Thread):
    def __init__(self, pid=None, interval=0.1):
        super().__init__()
        self.process = psutil.Process(pid) if pid else None
        self.interval = interval
        self.running = True
        self.power_used = 0.0
        self.samples = []
        self.timestamps = []

    def run(self):
        start_time = time.time()
        while self.running:
            try:
                if self.process:
                    cpu_percent = self.process.cpu_percent(interval=None)
                else:
                    cpu_percent = psutil.cpu_percent(interval=None)

                time.sleep(self.interval)
                power = (cpu_percent / 100) * TDP
                self.power_used += power * self.interval
                self.samples.append(power)
                self.timestamps.append(time.time() - start_time)
            except Exception:
                break

    def stop(self):
        self.running = False

    def get_power_usage_wh(self):
        return self.power_used / 3600

    def get_average_wattage(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

# === Idle power measurement ===
def measure_idle_power(duration):
    print(f"Measuring idle power for {duration:.2f} seconds...")
    idle_monitor = PowerMonitor()
    idle_monitor.start()
    time.sleep(duration)
    idle_monitor.stop()
    idle_monitor.join()
    return idle_monitor.get_power_usage_wh()

# === Setup ===
model_paths = [
    'models/yolov8n.pt',
    # 'models/yolov8s.pt',
    # 'models/yolov8m.pt',
    # 'models/yolov8l.pt',
    # 'models/yolov8x.pt'
]

images_dir = 'images'
imgsz = 640
image_files = sorted([
    os.path.join(images_dir, f) for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

results_data = []

# === Evaluation Loop ===
for model_path in tqdm(model_paths, desc="Evaluating Models", unit="model"):
    model_name = os.path.basename(model_path)
    model = YOLO(model_path)
    process_monitor = PowerMonitor(pid=psutil.Process().pid)

    start_time = time.time()
    process_monitor.start()

    for image_path in tqdm(image_files, desc=f"Predicting {model_name}", leave=False, unit="img"):
        model.predict(source=image_path, imgsz=imgsz, save=False, verbose=False)

    process_monitor.stop()
    process_monitor.join()
    end_time = time.time()

    duration = end_time - start_time
    idle_power_wh = measure_idle_power(duration)
    total_power_wh = process_monitor.get_power_usage_wh()
    avg_wattage = process_monitor.get_average_wattage()
    net_power_wh = max(total_power_wh - idle_power_wh, 0)

    results_data.append({
        'Model': model_name,
        'Idle Power (Wh)': round(idle_power_wh, 4),
        'Total Power (Wh)': round(total_power_wh, 4),
        'Net Power (Wh)': round(net_power_wh, 4),
        'Average Wattage (W)': round(avg_wattage, 2)
    })

# === Save Results ===
results_df = pd.DataFrame(results_data)
results_df.to_csv('experiment_2_predict_loop_results.csv', index=False)
print("Saved to 'experiment_2_predict_loop_results.csv'")

# === Plot Net Power Consumption Bar Chart ===
plt.figure(figsize=(8, 5))
plt.bar(results_df['Model'], results_df['Net Power (Wh)'], color='skyblue')
plt.title('YOLOv8 Net Power Usage by Model')
plt.xlabel('Model')
plt.ylabel('Net Power Usage (Wh)')
plt.tight_layout()
plt.savefig('net_power_usage_per_model_predict_loop.png')
plt.show()
