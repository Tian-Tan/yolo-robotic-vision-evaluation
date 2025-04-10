# Experiment 3: YOLOv8n - Inference vs Idle CPU and Memory Usage Over Time

import os
import time
import psutil
import threading
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

images_dir = 'images'
imgsz = 640
model_path = 'models/yolov8n.pt'

image_files = sorted([
    os.path.join(images_dir, f) for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.cpu_samples = []
        self.mem_samples = []
        self.timestamps = []

    def run(self):
        start_time = time.time()
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent
            time.sleep(self.interval)
            self.cpu_samples.append(cpu_percent)
            self.mem_samples.append(mem_percent)
            self.timestamps.append(time.time() - start_time)

    def stop(self):
        self.running = False

    def get_trace(self):
        return self.timestamps, self.cpu_samples, self.mem_samples

# === Inference Monitoring ===
print("Starting inference and resource monitoring...")
model = YOLO(model_path)
monitor = ResourceMonitor()
monitor.start()

for image_path in tqdm(image_files, desc="Running Inference", unit="img"):
    model.predict(source=image_path, imgsz=imgsz, save=False, verbose=False)

monitor.stop()
monitor.join()
inference_time = monitor.timestamps[-1]

timestamps_inf, cpu_inf, mem_inf = monitor.get_trace()

# === Idle Monitoring ===
print(f"\nMeasuring idle CPU and memory for {inference_time:.2f} seconds...")
idle_monitor = ResourceMonitor()
idle_monitor.start()
time.sleep(inference_time)
idle_monitor.stop()
idle_monitor.join()

timestamps_idle, cpu_idle, mem_idle = idle_monitor.get_trace()

# === Trim first 10 seconds ===
def trim_trace(timestamps, values, start=10.0):
    return [(t, v) for t, v in zip(timestamps, values) if t > start]

cpu_inf_filtered = trim_trace(timestamps_inf, cpu_inf)
mem_inf_filtered = trim_trace(timestamps_inf, mem_inf)
cpu_idle_filtered = trim_trace(timestamps_idle, cpu_idle)
mem_idle_filtered = trim_trace(timestamps_idle, mem_idle)

# Unpack back
t_cpu_inf, y_cpu_inf = zip(*cpu_inf_filtered)
t_mem_inf, y_mem_inf = zip(*mem_inf_filtered)
t_cpu_idle, y_cpu_idle = zip(*cpu_idle_filtered)
t_mem_idle, y_mem_idle = zip(*mem_idle_filtered)

# === Align length ===
min_len = min(len(t_cpu_inf), len(y_cpu_inf), len(y_cpu_idle), len(y_mem_inf), len(y_mem_idle))

# Truncate all to the same length
df = pd.DataFrame({
    'Time (s)': t_cpu_inf[:min_len],
    'CPU Usage Inference (%)': y_cpu_inf[:min_len],
    'CPU Usage Idle (%)': y_cpu_idle[:min_len],
    'Memory Usage Inference (%)': y_mem_inf[:min_len],
    'Memory Usage Idle (%)': y_mem_idle[:min_len],
})
df.to_csv('experiment_3_cpu_memory_vs_idle.csv', index=False)

# === Plot CPU Usage ===
plt.figure(figsize=(10, 5))
plt.plot(t_cpu_inf[:min_len], y_cpu_inf[:min_len], label='Inference CPU Usage')
plt.plot(t_cpu_inf[:min_len], y_cpu_idle[:min_len], label='Idle CPU Usage')
plt.title('CPU Usage Over Time: Inference vs Idle')
plt.xlabel('Time (s)')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cpu_usage_inference_vs_idle.png')
plt.show()

# === Plot Memory Usage ===
plt.figure(figsize=(10, 5))
plt.plot(t_mem_inf[:min_len], y_mem_inf[:min_len], label='Inference Memory Usage')
plt.plot(t_mem_inf[:min_len], y_mem_idle[:min_len], label='Idle Memory Usage')
plt.title('Memory Usage Over Time: Inference vs Idle')
plt.xlabel('Time (s)')
plt.ylabel('Memory Usage (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('memory_usage_inference_vs_idle.png')
plt.show()