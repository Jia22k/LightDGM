import os
import json
import numpy as np
import pandas as pd

metrics_path = "../methods/StableDiffusion/outputs/metrics.json"
npy_dir = "../methods/StableDiffusion/outputs/metrics"

# ------------------------
# Load metrics.json
# ------------------------

with open(metrics_path, "r") as f:
    metrics_data = json.load(f)

rows = []

# ------------------------
# Process each entry
# ------------------------

for entry in metrics_data:
    name = entry["image"]   # e.g. "10_bright"

    # Split name
    parts = name.split("_")
    image_id = parts[0]
    condition = parts[1]

    # Load brightness evolution
    brightness_path = os.path.join(npy_dir, f"{name}_brightness.npy")

    if os.path.exists(brightness_path):
        brightness_curve = np.load(brightness_path)
        final_brightness = brightness_curve[-1]
    else:
        final_brightness = None

    row = {
        "image": image_id,
        "condition": condition,
        "ssim": entry["ssim"],
        "lpips": entry["lpips"],
        "brightness_diff": entry["brightness_diff"],
        "hist_diff": entry["hist_diff"],
        "final_brightness": final_brightness
    }

    rows.append(row)

# ------------------------
# Create DataFrame
# ------------------------

df = pd.DataFrame(rows)

# Save full table
df.to_csv("../methods/StableDiffusion/outputs/aggregated_results.csv", index=False)

print("Saved aggregated_results.csv")

# ------------------------
# Pivot table (clean view)
# ------------------------

pivot = df.pivot(index="image", columns="condition")

pivot.to_csv("../methods/StableDiffusion/outputs/pivot_results.csv")

print("Saved pivot_results.csv")