from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pyheif

class CustomHEICImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [file for file in os.listdir(root_dir) if file.endswith('.heic')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        heic_path = os.path.join(self.root_dir, self.image_paths[idx])

        # HEIC 이미지를 JPEG로 변환
        output_jpg_path = os.path.splitext(heic_path)[0] + '.jpg'
        self.convert_heic_to_jpg(heic_path, output_jpg_path)

        # JPEG 이미지를 PIL 이미지로 로드
        image = Image.open(output_jpg_path)

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        return image

    def convert_heic_to_jpg(self, heic_path, output_jpg_path):
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
