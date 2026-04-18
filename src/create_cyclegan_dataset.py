from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_low = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "our485" / "low"
train_high = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "our485" / "high"

test_low = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "eval15" / "low"
test_high = PROJECT_ROOT / "processed" / "lol" / "raw" / "lol_dataset" / "eval15" / "high"

base_out = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix" / "datasets" / "lol_cyclegan"

trainA = base_out / "trainA"
trainB = base_out / "trainB"
testA = base_out / "testA"
testB = base_out / "testB"

for folder in [trainA, trainB, testA, testB]:
    folder.mkdir(parents=True, exist_ok=True)


def copy_images(src_dir: Path, dst_dir: Path):
    for img_path in sorted(src_dir.glob("*")):
        if img_path.is_file():
            shutil.copy2(img_path, dst_dir / img_path.name)
            print(f"Copied {img_path.name} -> {dst_dir.name}")


print("Copying training low-light images...")
copy_images(train_low, trainA)

print("Copying training high-light images...")
copy_images(train_high, trainB)

print("Copying test low-light images...")
copy_images(test_low, testA)

print("Copying test high-light images...")
copy_images(test_high, testB)

print("Done creating CycleGAN dataset.")