import torch
import torch.nn as nn


class LightingDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 256 * 8 * 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(),

            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, lighting):
        image_features = self.image_encoder(image)
        combined = torch.cat([image_features, lighting], dim=1)

        x = self.fc(combined)
        x = x.view(-1, 256, 8, 8)

        output = self.decoder(x)
        return output