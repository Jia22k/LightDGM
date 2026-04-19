import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import json
import numpy as np
from data.data_loader import LoLDataset
from model import ControlNetRunner
from eval.eval import compute_metrics

dataset = LoLDataset("datasets/LoL/eval15/low/")
runner = ControlNetRunner()

save_root = "outputs_controlnet"
os.makedirs(save_root, exist_ok=True)

prompts = {
    "bright": "same scene, strong bright lighting, high exposure, well illuminated",
    "dark": "same scene, low light, dim lighting, shadows",
    "warm": "same scene, warm lighting, orange tones",
    "cool": "same scene, cool lighting, blue tones"
}

negative_prompt = "dark, underexposed, dim lighting, shadows"

strengths = [0.15, 0.3, 0.5]

all_metrics = []

for idx in range(len(dataset)):
    img, path = dataset[idx]
    img_np = np.array(img)

    base_name = os.path.splitext(os.path.basename(path))[0]

    for condition, prompt in prompts.items():
        for strength in strengths:

            name = f"{base_name}_{condition}_s{int(strength*100)}"

            print(f"Running ControlNet: {name}")

            output_np = runner.run(
                img,
                prompt,
                negative_prompt,
                save_root,
                name,
                strength
            )

            metrics = compute_metrics(img_np, output_np)

            metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}

            metrics["image"] = base_name
            metrics["condition"] = condition
            metrics["strength"] = strength

            all_metrics.append(metrics)

            if len(all_metrics) % 10 == 0:
                with open(os.path.join(save_root, "metrics_partial.json"), "w") as f:
                    json.dump(all_metrics, f, indent=4)

# final save
with open(os.path.join(save_root, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=4)