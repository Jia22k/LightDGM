''' This checks whether the neural network can run without crashing.

It usually:

creates fake input data
passes it through the model
prints the output shape

Example:

input image shape = [1, 3, 128, 128]
output image shape = [1, 3, 128, 128]

That confirms the network architecture is valid.'''

import torch
from decoder import LightingDecoder

model = LightingDecoder()

dummy_image = torch.randn(1, 3, 128, 128)
dummy_lighting = torch.tensor([[1.0, 0.0]])

output = model(dummy_image, dummy_lighting)

print("Output shape:", output.shape)