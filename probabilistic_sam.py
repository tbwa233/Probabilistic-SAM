import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from segment_anything.modeling.sam import Sam
from fcomb_sam import SAMFcomb
from posterior_encoder import PosteriorEncoder
import numpy as np
import torch.nn.functional as F


def generate_boxes(masks, fallback_ratio=0.25):
    if masks.dim() == 4:
        masks = masks.squeeze(1)

    boxes = []
    for mask in masks:
        mask_np = mask.cpu().numpy()
        y, x = np.where(mask_np > 0.5)
        if len(x) == 0 or len(y) == 0:
            h, w = mask_np.shape
            x0, y0 = int(w * (0.5 - fallback_ratio)), int(h * (0.5 - fallback_ratio))
            x1, y1 = int(w * (0.5 + fallback_ratio)), int(h * (0.5 + fallback_ratio))
        else:
            x0, y0 = x.min(), y.min()
            x1, y1 = x.max(), y.max()
        boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.float32))

    return torch.stack(boxes).to(masks.device)

class ProbabilisticSAM(nn.Module):
    def __init__(self, sam_model: Sam, latent_dim=6, beta=10.0):
        super().__init__()
        self.sam = sam_model
        self.latent_dim = latent_dim
        self.beta = beta

        # Prior: P(z | x)
        self.prior_net = PosteriorEncoder(in_channels=3, latent_dim=latent_dim, use_mask=False)

        # Posterior: Q(z | x, y)
        self.posterior_net = PosteriorEncoder(in_channels=4, latent_dim=latent_dim, use_mask=True)

        # Fcomb: z + SAM decoder fusion
        self.fcomb = SAMFcomb(latent_dim=latent_dim, sam_mask_decoder=self.sam.mask_decoder)

    def forward(self, image, mask_gt=None, training=True):
        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image)

        # Latent distributions
        prior_dist = self.prior_net(image)
        z_prior = prior_dist.rsample()

        if training:
            posterior_dist = self.posterior_net(image, mask_gt)
            z_post = posterior_dist.rsample()
        else:
            posterior_dist = None
            z_post = z_prior

        # Prepare prompt from GT mask
        boxes = generate_boxes(mask_gt)
        sparse, dense = self.sam.prompt_encoder(None, boxes, None)
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # Get positional encoding from SAM
        pred_mask = self.fcomb(image_embedding, image_pe, sparse, dense, z_post)  # Pass it in

        return pred_mask, prior_dist, posterior_dist

    def loss(self, pred, mask_gt, prior_dist, posterior_dist):
        # Upsample prediction to match ground truth size
        pred_upsampled = F.interpolate(pred, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)

        bce = nn.BCEWithLogitsLoss()(pred_upsampled, mask_gt)

        pred_sigmoid = torch.sigmoid(pred_upsampled)
        dice = 1 - (2 * (pred_sigmoid * mask_gt).sum() + 1) / (pred_sigmoid.sum() + mask_gt.sum() + 1)

        recon_loss = bce + dice
        kl_loss = kl.kl_divergence(posterior_dist, prior_dist).mean()
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss