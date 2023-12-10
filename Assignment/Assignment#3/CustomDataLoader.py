from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFilter

# Custom Dataset
class CustomDataset(Dataset):
    """Custom Dataset Class"""
    def __init__(self, 
                 root_dir, 
                 train: bool = True,
                 convert: str = 'RGB',
                 transform=None):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.convert = convert
        if train:
            self.csv = pd.read_csv(root_dir.strip('/')+'/train.csv', dtype={'file_name' : str, 'label' : int})
        else:
            self.csv = pd.read_csv(root_dir.strip('/')+'/test.csv', dtype={'file_name' : str})
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if self.train:
            dpath = 'train'
        else:
            dpath = 'test'
        img_name = os.path.join(self.root_dir, dpath, self.csv.iloc[idx, 0].zfill(3)+'.jpg')
        image = Image.open(img_name).convert(self.convert)
        
        if self.transform:
            image = self.transform(image)
        
        if self.train:
            label = self.csv.iloc[idx, 1]
            return image, label
        else:
            return image