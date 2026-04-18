from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_low = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "our485" / "low"
train_high = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "our485" / "high"

test_low = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "eval15" / "low"
test_high = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "eval15" / "high"

out_train = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix" / "datasets" / "lol_pix2pix" / "train"
out_test = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix" / "datasets" / "lol_pix2pix" / "test"

out_train.mkdir(parents=True, exist_ok=True)
out_test.mkdir(parents=True, exist_ok=True)


def make_combined(low_dir: Path, high_dir: Path, out_dir: Path):
    for low_path in sorted(low_dir.glob("*")):
        if not low_path.is_file():
            continue

        high_path = high_dir / low_path.name

        if not high_path.exists():
            print(f"Skipping {low_path.name} - no matching high image")
            continue

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if low_img.size != high_img.size:
            high_img = high_img.resize(low_img.size)

        width, height = low_img.size

        combined = Image.new("RGB", (width * 2, height))
        combined.paste(low_img, (0, 0))
        combined.paste(high_img, (width, 0))

        save_path = out_dir / low_path.name
        combined.save(save_path)

        print(f"Saved: {save_path}")


print("Creating pix2pix training dataset...")
make_combined(train_low, train_high, out_train)

print("Creating pix2pix test dataset...")
make_combined(test_low, test_high, out_test)

print("Done creating pix2pix dataset.")