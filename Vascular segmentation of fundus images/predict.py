import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet
from dataset import FundusDataset

def predict(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            for i, pred in enumerate(preds):
                pred = pred.squeeze().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                img = Image.fromarray(pred)
                img.save(os.path.join(output_dir, img_names[i]))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    data_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_dataset = FundusDataset(
        image_dir='test/image',
        transform=data_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    predict(model, test_loader, device, output_dir='predictions')
    print('Predictions saved to predictions/')

if __name__ == '__main__':
    main()