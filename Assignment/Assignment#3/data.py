import os
import pyheif
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = self._find_heic_images(folder_path)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        heic_path = self.image_paths[idx]
        jpg_path = self._get_output_path(heic_path)
        self._convert_heic_to_jpg(heic_path, jpg_path)
        return jpg_path
    
    def _find_heic_images(self, folder_path):
        heic_images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.heic'):
                    heic_images.append(os.path.join(root, file))
        return heic_images
    
    def _get_output_path(self, heic_path):
        return os.path.splitext(heic_path)[0] + '.jpg'
    
    def _convert_heic_to_jpg(self, heic_path, output_jpg_path):
        heif_file = pyheif.read(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image.save(output_jpg_path, 'JPEG')
