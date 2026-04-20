import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------
# Paths (CHANGE IF NEEDED)
# ------------------------

retinex_path = "../methods/ZeroDCE/outputs/aggregated_results.csv"
gan_path = "../methods/EnlightenGAN/outputs/aggregated_results.csv"

save_dir = "../comparison_results"
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# Load data
# ------------------------

ret_df = pd.read_csv(retinex_path)
gan_df = pd.read_csv(gan_path)

ret_df["image"] = ret_df["image"].astype(str).str.replace(r"\.png$|\.jpg$", "", regex=True)
gan_df["image"] = gan_df["image"].astype(str).str.replace(r"\.png$|\.jpg$", "", regex=True)

# ------------------------
# Align by image (important)
# ------------------------

merged = pd.merge(ret_df, gan_df, on="image", suffixes=("_ret", "_gan"))

print("Total paired samples:", len(merged))

# ------------------------
# Summary stats
# ------------------------

def summarize(df, prefix):
    return {
        "SSIM": df[f"ssim_{prefix}"].mean(),
        "LPIPS": df[f"lpips_{prefix}"].mean(),
        "Brightness": df[f"brightness_diff_{prefix}"].mean(),
        "Histogram": df[f"hist_diff_{prefix}"].mean()
    }

ret_summary = summarize(merged, "ret")
gan_summary = summarize(merged, "gan")

summary_df = pd.DataFrame([ret_summary, gan_summary],
                          index=["RetinexNet", "EnlightenGAN"])

print("\n=== SUMMARY ===")
print(summary_df)

summary_df.to_csv(os.path.join(save_dir, "summary.csv"))

# ------------------------
# Paired difference (VERY IMPORTANT)
# ------------------------

merged["ssim_diff"] = merged["ssim_gan"] - merged["ssim_ret"]
merged["lpips_diff"] = merged["lpips_ret"] - merged["lpips_gan"]  # lower is better
merged["brightness_diff_delta"] = merged["brightness_diff_gan"] - merged["brightness_diff_ret"]

print("\n=== MEAN DIFFERENCES ===")
print("SSIM (GAN - Retinex):", merged["ssim_diff"].mean())
print("LPIPS (Retinex - GAN):", merged["lpips_diff"].mean())
print("Brightness (GAN - Retinex):", merged["brightness_diff_delta"].mean())

# ------------------------
# Plot 1: SSIM comparison
# ------------------------

plt.figure()
plt.scatter(merged["ssim_ret"], merged["ssim_gan"], alpha=0.6)
plt.xlabel("Retinex SSIM")
plt.ylabel("EnlightenGAN SSIM")
plt.title("SSIM Comparison")
plt.plot([0,1],[0,1],'r--')  # diagonal
plt.savefig(os.path.join(save_dir, "ssim_compare.png"))
plt.close()

# ------------------------
# Plot 2: LPIPS comparison
# ------------------------

plt.figure()
plt.scatter(merged["lpips_ret"], merged["lpips_gan"], alpha=0.6)
plt.xlabel("Retinex LPIPS")
plt.ylabel("EnlightenGAN LPIPS")
plt.title("LPIPS Comparison")
plt.plot([0,1],[0,1],'r--')
plt.savefig(os.path.join(save_dir, "lpips_compare.png"))
plt.close()

# ------------------------
# Plot 3: Brightness comparison
# ------------------------

plt.figure()
plt.scatter(merged["brightness_diff_ret"], merged["brightness_diff_gan"], alpha=0.6)
plt.xlabel("Retinex Brightness")
plt.ylabel("EnlightenGAN Brightness")
plt.title("Brightness Comparison")
plt.plot([-50,50],[-50,50],'r--')
plt.savefig(os.path.join(save_dir, "brightness_compare.png"))
plt.close()

# ------------------------
# Plot 4: Difference histograms
# ------------------------

plt.figure()
merged["ssim_diff"].hist(bins=20)
plt.title("SSIM Difference (GAN - Retinex)")
plt.savefig(os.path.join(save_dir, "ssim_diff_hist.png"))
plt.close()

plt.figure()
merged["lpips_diff"].hist(bins=20)
plt.title("LPIPS Difference (Retinex - GAN)")
plt.savefig(os.path.join(save_dir, "lpips_diff_hist.png"))
plt.close()

# ------------------------
# Save merged table
# ------------------------

merged.to_csv(os.path.join(save_dir, "paired_results.csv"), index=False)

print("\nSaved all comparison results to:", save_dir)