# Experiment 3: YOLOv8n - CPU Usage Over Time During Inference

import os
import time
import psutil
import threading
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# === Configuration ===
images_dir = 'images'
imgsz = 640
model_path = 'models/yolov8n.pt'
interval = 0.1  # sampling interval in seconds

image_files = sorted([
    os.path.join(images_dir, f) for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# === Resource Monitor Class ===
class CPUUsageMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.cpu_samples = []
        self.timestamps = []

    def run(self):
        start_time = time.time()
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=None)
            time.sleep(self.interval)
            self.cpu_samples.append(cpu_percent)
            self.timestamps.append(time.time() - start_time)

    def stop(self):
        self.running = False

    def get_trace(self):
        return self.timestamps, self.cpu_samples

# === Run Inference + Monitor CPU ===
print("Starting inference and CPU monitoring...")
model = YOLO(model_path)
monitor = CPUUsageMonitor(interval=interval)
monitor.start()

for image_path in tqdm(image_files, desc="Running Inference", unit="img"):
    model.predict(source=image_path, imgsz=imgsz, save=False, verbose=False)

monitor.stop()
monitor.join()

timestamps, cpu_usage = monitor.get_trace()

# === Trim first 10 seconds (warm-up) ===
def trim_trace(timestamps, values, start=10.0):
    return [(t, v) for t, v in zip(timestamps, values) if t > start]

filtered = trim_trace(timestamps, cpu_usage)
t_cpu, y_cpu = zip(*filtered)

# === Save to CSV ===
df = pd.DataFrame({
    'Time (s)': t_cpu,
    'CPU Usage (%)': y_cpu
})
df.to_csv('experiment_3.csv', index=False)
print("Saved CPU usage to 'experiment_3.csv'")

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(t_cpu, y_cpu, label='Inference CPU Usage')
plt.title('CPU Usage Over Time During Inference')
plt.xlabel('Time (s)')
plt.ylabel('CPU Usage (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('cpu_usage_inference_only.png')
plt.show()