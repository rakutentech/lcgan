import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import ImageOps, Image
import torchvision.transforms as transforms
import albumentations as A

class Dataset_(Dataset):
    def __init__(self, data_dir, resized_size, is_train):
        super(Dataset_, self).__init__()
        self.data_dir = data_dir
        self.resized_size = resized_size
        self.is_train = is_train
        self.rescaler = transforms.Resize((resized_size, resized_size), Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
        self.h_flip = transforms.RandomHorizontalFlip()
        self.color_jitter = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True),
        ])
        self.perspective_transform1 = A.Perspective(scale=(0.05, 0.1), keep_size=True, fit_output=True, always_apply=True)
        self.perspective_transform2 = A.Perspective(scale=(0.05, 0.1), keep_size=True, fit_output=False, always_apply=True)
        self.drop_out = A.CoarseDropout(max_holes=1, max_height=0.5, max_width=0.5, min_holes=1, min_height=0.3, min_width=0.3, always_apply=True)
        self.load_dataset()

    def random_geometry_transform(self, image):
        p = torch.rand(1)
        if p < 0.5:
            image = self.perspective_transform1(image=np.array(image))
        else:
            image = self.perspective_transform2(image=np.array(image))
        return image
    
    def random_appearance_transform(self, image):
        p = torch.rand(1)
        if p < 0.5:
            image = self.random_drop_out(image)
        else:
            image = self.random_color_transform(image)
        return image
    
    def random_drop_out(self, image):
        image = self.drop_out(image=np.array(image))
        return image
    
    def random_color_transform(self, image):
        image = self.color_jitter(image=np.array(image))
        return image

    def load_dataset(self):
        mode = "train"
        root = os.path.join(self.data_dir, mode)
        self.data = ImageFolder(root=root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.is_train:
            image, label = self.data[index]
            width, height = image.size
            if width == self.resized_size:
                resized_image = image
            else:
                resized_image = self.rescaler(image)
            
            resized_image = self.h_flip(resized_image)
            geometry_change = resized_image.copy()
            appearance_change = resized_image.copy()
            resized_image = self.to_tensor(resized_image)
            
            geometry_change = self.random_geometry_transform(geometry_change)
            appearance_change = self.random_appearance_transform(appearance_change)
            
            geometry_change = Image.fromarray(geometry_change['image'])
            appearance_change = Image.fromarray(appearance_change['image'])
            geometry_change = self.to_tensor(geometry_change)
            appearance_change = self.to_tensor(appearance_change)

            resized_image = resized_image * 2.0 - 1.0
            geometry_change = geometry_change * 2.0 - 1.0
            appearance_change = appearance_change * 2.0 - 1.0
            resized_image = torch.clamp(resized_image, min=-1.0, max=1.0)
            geometry_change = torch.clamp(geometry_change, min=-1.0, max=1.0)
            appearance_change = torch.clamp(appearance_change, min=-1.0, max=1.0)

            return resized_image, geometry_change, appearance_change
        else:
            image, label = self.data[index]
            width, height = image.size
            if width == self.resized_size:
                resized_image = image
            else:
                resized_image = self.rescaler(image)
            resized_image = self.to_tensor(resized_image)
            resized_image = resized_image * 2.0 - 1.0
            resized_image = torch.clamp(resized_image, min=-1.0, max=1.0)
            
            return resized_image, int(label)
