import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FundusDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.label_dir:
            label_path = os.path.join(self.label_dir, img_name)
            mask = np.array(Image.open(label_path).convert('L'))
            mask[mask > 0] = 255
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            return image, mask, img_name
        else:
            if self.transform:
                image = self.transform(image=image)['image']
            return image, img_name