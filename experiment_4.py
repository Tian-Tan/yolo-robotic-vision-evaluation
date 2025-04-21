# Experiment 4: Profiling YOLOv8 Inference and Grouping by Functional Category

import os
import time
import cProfile
import pstats
import io
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
model_path = 'models/yolov8n.pt'
images_dir = 'images'
imgsz = 640
profile_txt_output = 'experiment_4_profile_stats.txt'

image_files = sorted([
    os.path.join(images_dir, f) for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# Define model
model = YOLO(model_path)

# Define inference loop
def run_inference_loop():
    for image_path in tqdm(image_files, desc="Profiling YOLOv8 Predict", unit="img"):
        model.predict(source=image_path, imgsz=imgsz, save=False, verbose=False)

# Start profiling
print("Starting profiling...")
profiler = cProfile.Profile()
profiler.enable()
run_inference_loop()
profiler.disable()

# Extract profiling stats
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.strip_dirs()
ps.sort_stats('cumtime')
ps.print_stats()
full_stats_text = s.getvalue()

# Save raw profiling report to text file
with open(profile_txt_output, 'w') as f:
    f.write(full_stats_text)

# Define grouping logic
group_map = {
    "Preprocessing": ["imread", "ultralytics.utils.patches"],
    "Data Loading": ["__next__", "ultralytics.data.loaders"],
    "Forward Pass": ["forward", "_call_impl", "conv2d", "silu", "max_pool2d"],
    "Postprocessing": ["cat", "postprocess", "ultralytics.utils.ops"],
    "Execution Controller": ["stream_inference", "__call__", "generator_context", "predict"],
}

grouped_time = {k: 0.0 for k in group_map}
ungrouped_funcs = []
total_time = 0.0

# Aggregate times by group
for func, stat in ps.stats.items():
    file_path, line_no, func_name = func
    cum_time = stat[3]
    total_time += cum_time
    matched = False
    for group, substrings in group_map.items():
        if any(sub in func_name or sub in file_path for sub in substrings):
            grouped_time[group] += cum_time
            matched = True
            break
    if not matched:
        ungrouped_funcs.append((f"{func_name} ({os.path.basename(file_path)})", cum_time))

# Only include unmatched time in "Other"
grouped_time["Other"] = sum(t for _, t in ungrouped_funcs)

# Save grouped summary to CSV
group_df = pd.DataFrame({
    'Stage': list(grouped_time.keys()),
    'Cumulative Time (s)': list(grouped_time.values())
})
group_df['Percent (%)'] = (group_df['Cumulative Time (s)'] / total_time * 100).round(2)
group_df = group_df.sort_values(by='Cumulative Time (s)', ascending=False)
group_df.to_csv('experiment_4_profile_summary.csv', index=False)

# Append top 30 ungrouped functions to profile report
ungrouped_funcs.sort(key=lambda x: x[1], reverse=True)
with open(profile_txt_output, 'a') as f:
    f.write("\n\nTop 30 Ungrouped (Other) Functions by Time:\n")
    for name, time_spent in ungrouped_funcs[:30]:
        f.write(f"{name:60s} {time_spent:.4f} s\n")

# Plot grouped time as a bar chart
plt.figure(figsize=(8, 5))
plt.bar(group_df['Stage'], group_df['Cumulative Time (s)'], color='cornflowerblue')
plt.title('Cumulative Inference Time by Stage (YOLOv8n)')
plt.xlabel('Inference Stage')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('experiment_4_profile_grouped_bar.png')
plt.show()