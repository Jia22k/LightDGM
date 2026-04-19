import os
import json
import numpy as np
import pandas as pd

metrics_path = "../methods/ControlNet/outputs/metrics.json"
npy_dir = "../methods/ControlNet/outputs/metrics"

# ------------------------
# Load JSON
# ------------------------

with open(metrics_path, "r") as f:
    metrics_data = json.load(f)

rows = []

# ------------------------
# Process entries
# ------------------------

for entry in metrics_data:
    image_id = entry["image"]
    condition = entry["condition"]
    strength = entry["strength"]

    name = f"{image_id}_{condition}_s{int(strength*100)}"

    brightness_path = os.path.join(npy_dir, f"{name}_brightness.npy")

    if os.path.exists(brightness_path):
        brightness_curve = np.load(brightness_path)
        final_brightness = brightness_curve[-1]
    else:
        final_brightness = None

    row = {
        "image": image_id,
        "condition": condition,
        "strength": strength,
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

# Save flat table
df.to_csv("../methods/ControlNet/outputs/aggregated_results.csv", index=False)

print("Saved aggregated_results.csv")

# ------------------------
# Pivot (multi-index)
# ------------------------

pivot = df.pivot_table(
    index="image",
    columns=["condition", "strength"],
    values=["ssim", "lpips", "brightness_diff"]
)

pivot.to_csv("../methods/ControlNet/outputs/pivot_results_multistrength.csv")

print("Saved pivot_results_multistrength.csv")