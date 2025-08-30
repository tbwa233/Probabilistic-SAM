from probabilistic_sam import ProbabilisticSAM
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

def train(model, loaders, device, epochs=10):
    model.to(device)
    opt = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loaders['train']):
            image = batch['image_sam'].to(device)
            mask = batch['mask'].to(device)  # shape [B, 1, H, W]

            pred, prior, posterior = model(image, mask, training=True)
            loss, recon, kl = model.loss(pred, mask, prior, posterior)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loaders['train']):.4f}")