from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import json

input_dir = Path("../data/processed/ficus/resized_images")
output_dir = Path("../data/processed/ficus/lighting_variants")
lighting_code_path = Path("../data/processed/ficus/lighting_codes/lighting_codes.json")

output_dir.mkdir(parents=True, exist_ok=True)
lighting_code_path.parent.mkdir(parents=True, exist_ok=True)

lighting_codes = {}

def apply_warm_cool(img, warmth):
    arr = np.array(img).astype(np.float32) / 255.0

    if warmth > 0:
        arr[..., 0] *= (1.0 + 0.3 * warmth)
        arr[..., 2] *= (1.0 - 0.2 * warmth)
    else:
        arr[..., 2] *= (1.0 + 0.3 * abs(warmth))
        arr[..., 0] *= (1.0 - 0.2 * abs(warmth))

    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

variants = [
    ("orig", 1.0, 0.0),
    ("dark", 0.7, 0.0),
    ("bright", 1.3, 0.0),
    ("warm", 1.0, 0.8),
    ("cool", 1.0, -0.8),
]

image_paths = list(input_dir.glob("*.png"))

for image_path in image_paths:
    img = Image.open(image_path).convert("RGB")
    stem = image_path.stem

    for variant_name, brightness, warmth in variants:
        modified = ImageEnhance.Brightness(img).enhance(brightness)
        modified = apply_warm_cool(modified, warmth)

        output_name = f"{stem}_{variant_name}.png"
        modified.save(output_dir / output_name)

        lighting_codes[output_name] = {
            "brightness": brightness,
            "warmth": warmth
        }

with open(lighting_code_path, "w") as f:
    json.dump(lighting_codes, f, indent=2)

print(f"Created {len(lighting_codes)} lighting variant images")
print(f"Saved lighting codes to {lighting_code_path}")