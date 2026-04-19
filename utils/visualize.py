import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------
# Setup
# ------------------------

csv_path = "../methods/ZeroDCE/outputs/aggregated_results.csv"
save_dir = "../methods/ZeroDCE/outputs/plots"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Normalize columns
df["condition"] = df.get("condition", "none")
df["strength"] = df.get("strength", "none")
df["strength_numeric"] = pd.to_numeric(df["strength"], errors="coerce")

has_strength = df["strength_numeric"].notna().any()
has_condition = df["condition"].nunique() > 1

# ------------------------
# 1. SSIM vs Brightness
# ------------------------

plt.figure(figsize=(8,6))

if has_strength:
    scatter = plt.scatter(
        df["brightness_diff"],
        df["ssim"],
        c=df["strength_numeric"],
        cmap="viridis",
        alpha=0.8
    )
    plt.colorbar(scatter, label="Strength")
else:
    plt.scatter(df["brightness_diff"], df["ssim"], alpha=0.8)

plt.xlabel("Brightness Change")
plt.ylabel("SSIM")
plt.title("SSIM vs Brightness")
plt.grid()

plt.savefig(os.path.join(save_dir, "tradeoff.png"))
plt.close()

# ------------------------
# 2. Brightness Histogram (ALWAYS)
# ------------------------

plt.figure()
df["brightness_diff"].hist(bins=20)
plt.xlabel("Brightness Change")
plt.title("Brightness Distribution")
plt.grid()

plt.savefig(os.path.join(save_dir, "brightness_hist.png"))
plt.close()

# ------------------------
# 3. Brightness Consistency
# ------------------------

plt.figure()
plt.plot(df["brightness_diff"].values)
plt.xlabel("Image Index")
plt.ylabel("Brightness Change")
plt.title("Brightness Consistency")
plt.grid()

plt.savefig(os.path.join(save_dir, "brightness_trend.png"))
plt.close()

# ------------------------
# 4. SSIM Distribution
# ------------------------

plt.figure()
df["ssim"].hist(bins=20)
plt.title("SSIM Distribution")
plt.grid()

plt.savefig(os.path.join(save_dir, "ssim_hist.png"))
plt.close()

# ------------------------
# 5. LPIPS Distribution
# ------------------------

plt.figure()
df["lpips"].hist(bins=20)
plt.title("LPIPS Distribution")
plt.grid()

plt.savefig(os.path.join(save_dir, "lpips_hist.png"))
plt.close()

# ------------------------
# 6. Strength-based plots (if available)
# ------------------------

if has_strength:
    plt.figure(figsize=(8,6))

    if has_condition:
        for condition in df["condition"].unique():
            subset = df[df["condition"] == condition]
            grouped = subset.groupby("strength_numeric").mean(numeric_only=True)

            plt.plot(grouped.index, grouped["brightness_diff"], marker='o', label=condition)
        plt.legend()
    else:
        grouped = df.groupby("strength_numeric").mean(numeric_only=True)
        plt.plot(grouped.index, grouped["brightness_diff"], marker='o')

    plt.xlabel("Strength")
    plt.ylabel("Brightness Change")
    plt.title("Brightness vs Strength")
    plt.grid()

    plt.savefig(os.path.join(save_dir, "brightness_vs_strength.png"))
    plt.close()

print(f"All plots saved to: {save_dir}")