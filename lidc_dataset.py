import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random
import cv2

class LIDC_IDRI(Dataset):
    def __init__(self, dataset_location, image_size=1024):
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.series_uid = []

        # Load pickled dataset
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            if file.endswith('.pickle'):
                print(f"Loading file: {file}")
                file_path = os.path.join(dataset_location, file)
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        # Parse dataset content
        for key, value in data.items():
            self.images.append(value['image'].astype(np.float32))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert len(self.images) == len(self.labels) == len(self.series_uid)
        print(f"Loaded {len(self.images)} samples.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and randomly selected mask (1 of 4 annotations)
        image = self.images[idx]
        mask = self.labels[idx][random.randint(0, 3)].astype(np.float32)

        # Resize
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        image_resized = (image_resized - np.min(image_resized)) / (np.max(image_resized) - np.min(image_resized) + 1e-8)

        # SAM expects 3-channel RGB input
        image_rgb = np.stack([image_resized] * 3, axis=-1)  # (H, W, 3)
        image_rgb = torch.FloatTensor(image_rgb.transpose(2, 0, 1))  # (3, H, W)

        mask_tensor = torch.FloatTensor(mask_resized).unsqueeze(0)  # shape: (1, H, W)

        return {
            'image_sam': image_rgb,
            'mask': mask_tensor,
            'uid': self.series_uid[idx]
        }