import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("evaluation_results.csv")

# Create output folder if needed
import os
os.makedirs("report_figures", exist_ok=True)

# Brightness Difference Plot
brightness_avg = df.groupby("model")["brightness_diff"].mean()

plt.figure(figsize=(8, 5))
brightness_avg.plot(kind="bar")
plt.title("Average Brightness Difference by Model")
plt.ylabel("Brightness Difference")
plt.xlabel("Model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("report_figures/brightness_difference.png")
plt.show()

# Histogram Difference Plot
hist_avg = df.groupby("model")["histogram_diff"].mean()

plt.figure(figsize=(8, 5))
hist_avg.plot(kind="bar")
plt.title("Average Histogram Difference by Model")
plt.ylabel("Histogram Difference")
plt.xlabel("Model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("report_figures/histogram_difference.png")
plt.show()

# SSIM Plot
ssim_avg = df.groupby("model")["ssim"].mean()

plt.figure(figsize=(8, 5))
ssim_avg.plot(kind="bar")
plt.title("Average SSIM by Model")
plt.ylabel("SSIM")
plt.xlabel("Model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("report_figures/ssim.png")
plt.show()

print("Saved plots to report_figures/")