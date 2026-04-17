''' file actually trains the model.

It:

loads the dataset
loads the neural network
loops through the data many times
calculates loss
updates the model weights
saves checkpoints.'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LightingDataset
from decoder import LightingDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LightingDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = LightingDecoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 500

for epoch in range(epochs):
    total_loss = 0

    for original_image, lighting_tensor, target_image in dataloader:
        original_image = original_image.to(device)
        lighting_tensor = lighting_tensor.to(device)
        target_image = target_image.to(device)

        output = model(original_image, lighting_tensor)

        loss = criterion(output, target_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "../outputs/checkpoints/lighting_decoder.pth")
print("Saved model checkpoint.")