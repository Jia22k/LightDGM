import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Setup
# ------------------------

csv_path = "../methods/StableDiffusion/outputs/aggregated_results.csv"
save_dir = "../methods/StableDiffusion/outputs/plots"

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# ------------------------
# 1. SSIM vs Brightness (MAIN FIGURE)
# ------------------------

plt.figure(figsize=(8,6))

for condition in df["condition"].unique():
    subset = df[df["condition"] == condition]
    plt.scatter(
        subset["brightness_diff"],
        subset["ssim"],
        label=condition,
        alpha=0.7
    )

plt.xlabel("Brightness Change")
plt.ylabel("SSIM (Structure Preservation)")
plt.title("Brightness vs Structure Tradeoff")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "ssim_vs_brightness.png"))
plt.close()

# ------------------------
# 2. Colored Scatter (LPIPS)
# ------------------------

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    df["brightness_diff"],
    df["ssim"],
    c=df["lpips"],
    cmap="coolwarm",
    alpha=0.8
)

plt.colorbar(scatter, label="LPIPS")
plt.xlabel("Brightness Change")
plt.ylabel("SSIM")
plt.title("Tradeoff with Perceptual Change (LPIPS)")
plt.grid()

plt.savefig(os.path.join(save_dir, "ssim_vs_brightness_lpips.png"))
plt.close()

# ------------------------
# 3. Average Metrics per Condition
# ------------------------

grouped = df.groupby("condition").mean(numeric_only=True)

# Brightness
grouped["brightness_diff"].plot(kind="bar", figsize=(6,4))
plt.title("Average Brightness Change per Condition")
plt.ylabel("Brightness Change")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "avg_brightness.png"))
plt.close()

# SSIM
grouped["ssim"].plot(kind="bar", figsize=(6,4))
plt.title("Average SSIM per Condition")
plt.ylabel("SSIM")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "avg_ssim.png"))
plt.close()

# LPIPS
grouped["lpips"].plot(kind="bar", figsize=(6,4))
plt.title("Average LPIPS per Condition")
plt.ylabel("LPIPS")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "avg_lpips.png"))
plt.close()

# ------------------------
# 4. Variance (Stability)
# ------------------------

grouped_std = df.groupby("condition").std(numeric_only=True)

grouped_std["brightness_diff"].plot(kind="bar", figsize=(6,4))
plt.title("Brightness Variability per Condition")
plt.ylabel("Std Dev")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "brightness_variance.png"))
plt.close()

# ------------------------
# 5. Heatmap (Optional)
# ------------------------

pivot = df.pivot(index="image", columns="condition", values="ssim")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis", annot=False)
plt.title("SSIM Heatmap Across Images and Conditions")
plt.tight_layout()

plt.savefig(os.path.join(save_dir, "ssim_heatmap.png"))
plt.close()

# ------------------------
# Done
# ------------------------

print(f"All plots saved to: {save_dir}")