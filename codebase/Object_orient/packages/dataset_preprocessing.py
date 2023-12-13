import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda

class DatasetPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            ToTensor(),
            Lambda(lambda x: torch.tensor(self.snake_scan(x.numpy())))
        ])

    def snake_scan(self, img):
        if len(img.shape) != 3:
            raise ValueError('Input image must be a 3D array')

        channels, rows, cols = img.shape
        snake = np.zeros((rows, cols * channels), dtype=img.dtype)
        for r in range(rows):
            row_data = img[:, r, :].flatten()
            if r % 2 == 1:
                row_data = row_data[::-1]
            snake[r] = row_data
        return snake

    def load_data(self):
        train_data = datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=self.transform
        )

        test_data = datasets.CIFAR10(
            root='data',
            train=False,
            download=False,
            transform=self.transform
        )

        loaders = {
            'train': torch.utils.data.DataLoader(train_data,
                                                 batch_size=100,
                                                 shuffle=True,
                                                 num_workers=0),

            'test': torch.utils.data.DataLoader(test_data,
                                                batch_size=100,
                                                shuffle=False,
                                                num_workers=0),
        }

        return loaders

# Usage:
preprocessor = DatasetPreprocessor()
loaders = preprocessor.load_data()
