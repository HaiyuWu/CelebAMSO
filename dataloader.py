from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
from glob import glob


class AttributesTestDataset(Dataset):
    def __init__(self, image_path, label_path, transforms=None):
        self.images = glob(image_path + "/*")
        self.labels = np.array(pd.read_csv(label_path))[:, 1:]
        self.labels[self.labels == -1] = 0
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        label = self.labels[item][[21]]
        if self.transforms:
            image = self.transforms(image)
        return image, label.astype(np.float32)
