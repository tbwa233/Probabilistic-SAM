import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=6, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_sigma = nn.Linear(64, latent_dim)

    def forward(self, image, mask=None):
        if self.use_mask:
            x = torch.cat([image, mask], dim=1)
        else:
            x = image

        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)

        mu = self.fc_mu(feat)
        log_sigma = torch.clamp(self.fc_log_sigma(feat), min=-5, max=5)
        sigma = torch.exp(log_sigma)

        return Independent(Normal(mu, sigma), 1)