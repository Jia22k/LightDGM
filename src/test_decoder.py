import torch
from decoder import LightingDecoder

model = LightingDecoder()

dummy_image = torch.randn(1, 3, 128, 128)
dummy_lighting = torch.tensor([[1.0, 0.0]])

output = model(dummy_image, dummy_lighting)

print("Output shape:", output.shape)