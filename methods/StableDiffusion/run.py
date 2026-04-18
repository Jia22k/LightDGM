import os
import json
import numpy as np
from data_loader import LoLDataset
from model import StableDiffusionRunner
from eval import compute_metrics

# ------------------------
# Setup
# ------------------------

dataset = LoLDataset("../../data/LoL/eval15/low/")
runner = StableDiffusionRunner()

save_root = "outputs"
os.makedirs(save_root, exist_ok=True)

# ------------------------
# Lighting prompts
# ------------------------

prompts = {
    "bright": "same scene, strong bright lighting, high exposure, well illuminated",
    "dark": "same scene, low light, dim lighting, shadows",
    "warm": "same scene, warm lighting, orange tones",
    "cool": "same scene, cool lighting, blue tones"
}

negative_prompt = "dark, underexposed, dim lighting, shadows"

# ------------------------
# Loop
# ------------------------

strengths = [0.15, 0.3, 0.5]
all_metrics = []

for idx in range(len(dataset)):
    img, path = dataset[idx]
    img_np = np.array(img)

    base_name = os.path.splitext(os.path.basename(path))[0]

    for condition, prompt in prompts.items():
        for strength in strengths:

            name = f"{base_name}_{condition}_s{int(strength*100)}"

            print(f"Running: {name}")

            output_np = runner.run(
                img,
                prompt,
                negative_prompt,
                save_root,
                name,
                strength=strength   # NEW
            )

            metrics = compute_metrics(img_np, output_np)

            metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}

            metrics["image"] = base_name
            metrics["condition"] = condition
            metrics["strength"] = strength

            all_metrics.append(metrics)

            # ------------------------
            # Checkpoints
            # ------------------------
            if len(all_metrics) % 10 == 0:
                with open("outputs/metrics_partial.json", "w") as f:
                    json.dump(all_metrics, f, indent=4)

                print(f"Checkpoint saved at {len(all_metrics)} entries")

# ------------------------
# Save metrics
# ------------------------
with open(os.path.join(save_root, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=4)