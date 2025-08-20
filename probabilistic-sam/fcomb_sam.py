import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMFcomb(nn.Module):
    def __init__(self, latent_dim, sam_mask_decoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = sam_mask_decoder
        self.z_projector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, image_embedding, image_pe, sparse, dense, z):
        B, _, H, W = image_embedding.shape
        z_embed = self.z_projector(z).unsqueeze(1).expand(B, sparse.shape[1], -1)

        new_sparse = sparse + z_embed

        pred, _ = self.decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,  # pass positional encoding
            sparse_prompt_embeddings=new_sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        return pred