import os
import json
import numpy as np
import pandas as pd

metrics_path = "../methods/ZeroDCE/outputs/metrics.json"
npy_dir = "../methods/ZeroDCE/outputs/metrics"

# ------------------------
# Load JSON
# ------------------------

with open(metrics_path, "r") as f:
    metrics_data = json.load(f)

rows = []

# ------------------------
# Process entries (GENERALIZED)
# ------------------------

for entry in metrics_data:
    image_id = entry.get("image")

    # Optional fields
    condition = entry.get("condition", "none")
    strength = entry.get("strength", "none")

    # Build name ONLY if fields exist
    if condition != "none" and strength != "none":
        name = f"{image_id}_{condition}_s{int(float(strength)*100)}"
    else:
        name = str(image_id)

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
        "ssim": entry.get("ssim"),
        "lpips": entry.get("lpips"),
        "brightness_diff": entry.get("brightness_diff"),
        "hist_diff": entry.get("hist_diff"),
        "final_brightness": final_brightness
    }

    rows.append(row)

# ------------------------
# Create DataFrame
# ------------------------

df = pd.DataFrame(rows)

# Save flat table
output_dir = "../methods/ZeroDCE/outputs"
df.to_csv(os.path.join(output_dir, "aggregated_results.csv"), index=False)

print("Saved aggregated_results.csv")

# ------------------------
# Pivot (ONLY if condition exists)
# ------------------------

if "condition" in df.columns and df["condition"].nunique() > 1:

    pivot_cols = ["condition"]

    # Add strength only if meaningful
    if df["strength"].nunique() > 1:
        pivot_cols.append("strength")

    pivot = df.pivot_table(
        index="image",
        columns=pivot_cols,
        values=["ssim", "lpips", "brightness_diff"]
    )

    pivot.to_csv(os.path.join(output_dir, "pivot_results.csv"))
    print("Saved pivot_results.csv")

else:
    print("Skipping pivot (no condition/strength variation)")