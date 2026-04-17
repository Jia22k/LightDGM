
''' This is a quick debugging file.

It checks whether dataset.py is working correctly.

For example, it prints:

image shapes
tensor shapes
lighting tensor values

This helps catch errors early.'''
from dataset import LightingDataset

dataset = LightingDataset()

original_image, lighting_tensor, target_image = dataset[0]

print("Dataset size:", len(dataset))
print("Original image shape:", original_image.shape)
print("Target image shape:", target_image.shape)
print("Lighting tensor:", lighting_tensor)