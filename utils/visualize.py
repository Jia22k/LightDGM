import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Setup
# ------------------------

csv_path = "../methods/ControlNet/outputs/aggregated_results.csv"
save_dir = "../methods/ControlNet/outputs/plots"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# ------------------------
# 1. SSIM vs Brightness (colored by strength)
# ------------------------

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    df["brightness_diff"],
    df["ssim"],
    c=df["strength"],
    cmap="viridis",
    alpha=0.8
)

plt.colorbar(scatter, label="Strength")
plt.xlabel("Brightness Change")
plt.ylabel("SSIM (Structure)")
plt.title("SSIM vs Brightness (Colored by Strength)")
plt.grid()

plt.savefig(os.path.join(save_dir, "tradeoff_strength.png"))
plt.close()

# ------------------------
# 2. Separate tradeoff per strength
# ------------------------

for strength in sorted(df["strength"].unique()):
    subset = df[df["strength"] == strength]

    plt.figure(figsize=(8,6))

    for condition in subset["condition"].unique():
        cond_subset = subset[subset["condition"] == condition]

        plt.scatter(
            cond_subset["brightness_diff"],
            cond_subset["ssim"],
            label=condition,
            alpha=0.7
        )

    plt.xlabel("Brightness Change")
    plt.ylabel("SSIM")
    plt.title(f"Tradeoff at Strength = {strength}")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, f"tradeoff_strength_{int(strength*100)}.png"))
    plt.close()

# ------------------------
# 3. Brightness vs Strength (VERY IMPORTANT)
# ------------------------

plt.figure(figsize=(8,6))

for condition in df["condition"].unique():
    subset = df[df["condition"] == condition]
    grouped = subset.groupby("strength").mean(numeric_only=True)

    plt.plot(
        grouped.index,
        grouped["brightness_diff"],
        marker='o',
        label=condition
    )

plt.xlabel("Strength")
plt.ylabel("Brightness Change")
plt.title("Brightness vs Strength")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "brightness_vs_strength.png"))
plt.close()

# ------------------------
# 4. SSIM vs Strength
# ------------------------

plt.figure(figsize=(8,6))

for condition in df["condition"].unique():
    subset = df[df["condition"] == condition]
    grouped = subset.groupby("strength").mean(numeric_only=True)

    plt.plot(
        grouped.index,
        grouped["ssim"],
        marker='o',
        label=condition
    )

plt.xlabel("Strength")
plt.ylabel("SSIM")
plt.title("Structure Preservation vs Strength")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "ssim_vs_strength.png"))
plt.close()

# ------------------------
# 5. Variance vs Strength (STABILITY)
# ------------------------

plt.figure(figsize=(8,6))

for condition in df["condition"].unique():
    subset = df[df["condition"] == condition]
    grouped_std = subset.groupby("strength").std(numeric_only=True)

    plt.plot(
        grouped_std.index,
        grouped_std["brightness_diff"],
        marker='o',
        label=condition
    )

plt.xlabel("Strength")
plt.ylabel("Brightness Std Dev")
plt.title("Brightness Variability vs Strength")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "variance_vs_strength.png"))
plt.close()

# ------------------------
# 6. Condition comparison (bar chart at each strength)
# ------------------------

for strength in sorted(df["strength"].unique()):
    subset = df[df["strength"] == strength]
    grouped = subset.groupby("condition").mean(numeric_only=True)

    grouped["brightness_diff"].plot(kind="bar", figsize=(6,4))
    plt.title(f"Brightness per Condition (Strength={strength})")
    plt.ylabel("Brightness Change")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"brightness_condition_s{int(strength*100)}.png"))
    plt.close()

# ------------------------
# Done
# ------------------------

print(f"All multi-strength plots saved to: {save_dir}")