import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import random

from segment_anything import sam_model_registry
from probabilistic_sam import ProbabilisticSAM
from lidc_dataset import LIDC_IDRI
from train import train

# Config
DATA_DIR = "/content/drive/MyDrive/Data/"  # Update as needed
SAM_CKPT = "/content/drive/MyDrive/sam_vit_b_01ec64.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 10
LATENT_DIM = 6
BETA = 10.0

def create_dataloaders():
    dataset = LIDC_IDRI(dataset_location=DATA_DIR)
    size = len(dataset)
    indices = list(range(size))
    random.shuffle(indices)
    split = int(0.1 * size)
    train_indices, val_indices = indices[split:], indices[:split]
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices))
    return {"train": train_loader, "val": val_loader}

def main():
    # Load SAM backbone
    sam_model = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)

    # Initialize Probabilistic SAM
    model = ProbabilisticSAM(sam_model=sam_model, latent_dim=LATENT_DIM, beta=BETA)

    # Dataloaders
    loaders = create_dataloaders()

    # Train
    train(model, loaders, device=DEVICE, epochs=EPOCHS)

    # Save model
    torch.save(model.state_dict(), "probabilistic_sam_lidc.pth")

if __name__ == "__main__":
    main()