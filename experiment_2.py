# Experiment 2: Accuracy (Model size) vs power usage (net power used by process)

import psutil
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

TDP = 28  # 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz

class PowerMonitor(threading.Thread):
    def __init__(self, pid=None, interval=0.1):
        super().__init__()
        self.process = psutil.Process(pid) if pid else None
        self.interval = interval
        self.running = True
        self.power_used = 0.0

    def run(self):
        while self.running:
            try:
                if self.process:
                    cpu_percent = self.process.cpu_percent(interval=None)
                else:
                    cpu_percent = psutil.cpu_percent(interval=None)

                time.sleep(self.interval)
                self.power_used += (cpu_percent / 100) * TDP * self.interval
            except Exception:
                break

    def stop(self):
        self.running = False

    def get_power_usage_wh(self):
        return self.power_used / 3600  # Convert Joules to Wh

def measure_idle_power(duration):
    print(f"Measuring idle power for {duration:.2f} seconds...")
    idle_monitor = PowerMonitor()
    idle_monitor.start()
    time.sleep(duration)
    idle_monitor.stop()
    idle_monitor.join()
    return idle_monitor.get_power_usage_wh()

# List of YOLOv8 model paths
model_paths = [
    'models/yolov8n.pt',
    'models/yolov8s.pt',
    # 'models/yolov8m.pt',
    # 'models/yolov8l.pt',
    # 'models/yolov8x.pt'
]

results_data = []

for model_path in model_paths:
    print(f"Evaluating {model_path}...")

    model = YOLO(model_path)
    process_monitor = PowerMonitor(pid=psutil.Process().pid)
    start_time = time.time()
    process_monitor.start()

    metrics = model.val(
        data='self_driving.yaml',
        imgsz=640,
        classes=[0, 1, 2, 7, 9]
    )

    process_monitor.stop()
    process_monitor.join()
    end_time = time.time()

    eval_duration = end_time - start_time
    idle_power_wh = measure_idle_power(eval_duration)
    total_power_wh = process_monitor.get_power_usage_wh()
    net_power_wh = max(total_power_wh - idle_power_wh, 0)

    results_data.append({
        'Model': model_path.split('/')[-1],
        'Idle Power (Wh)': round(idle_power_wh, 4),
        'Total Power (Wh)': round(total_power_wh, 4),
        'Net Power (Wh)': round(net_power_wh, 4),
        'mAP@0.5': metrics.box.map50,
        'mAP@0.5:0.95': metrics.box.map,
    })

# Save results
results_df = pd.DataFrame(results_data)
results_df.to_csv('experiment_2_results.csv', index=False)
print("Evaluation results saved to 'experiment_2_results.csv'")

# Plot Net Power Usage vs Accuracy
plt.figure(figsize=(8, 5))
plt.plot(results_df['mAP@0.5'], results_df['Net Power (Wh)'], marker='o', label='Net Power vs mAP@0.5')
plt.plot(results_df['mAP@0.5:0.95'], results_df['Net Power (Wh)'], marker='o', label='Net Power vs mAP@0.5:0.95')
plt.title('YOLOv8 Power Usage vs Accuracy')
plt.xlabel('Accuracy (mAP)')
plt.ylabel('Net Power Usage (Wh)')
plt.grid(True)
plt.legend()
plt.show()