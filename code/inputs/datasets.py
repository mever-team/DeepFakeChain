import cv2 
import torch
import albumentations as A
import numpy as np


from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def crop(im, margin):
    w, h, _ = im.shape
    shrink_factor = margin / 2.
    bb = (int((1 - shrink_factor) * w / 2),
          int((1 - shrink_factor) * h / 2),
          int((1 + shrink_factor) * w / 2),
          int((1 + shrink_factor) * h / 2))
    
    return im[bb[0]: bb[2], bb[1]: bb[3], :]


class ImagePathDataset(Dataset):

    def __init__(self,
                 data, 
                 target_size, 
                 augmentation=None, 
                 normalization=None, 
                 preprocessing=None, 
                 margin=1.3, 
                 label_transform=None,
                 num_labels=None):

        super().__init__()

        self.data = data
        self.margin = margin     
        
        preprocessing = preprocessing or A.Compose([])
        augmentation = augmentation or A.Compose([])
        normalization = normalization or A.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))       
        self.transform = A.Compose([
            preprocessing,
            A.Resize(*target_size),
            augmentation,
            normalization,
            ToTensorV2(),
        ])

        self.label_transform = label_transform or (lambda x:x)
        self.num_labels = num_labels or self.estimated_num_labels 
        # num_labels depends on the source, the task, and the considered manipulation categories
        # is consistent with original category labels even if some manipulations are missing
        
    def set_target_size(self, target_size):
        self.transform[1].height = target_size[0]
        self.transform[1].width = target_size[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path, label = self.data[idx]
        im = cv2.imread(str(path))
        if self.margin != 1.3 and self.margin != 2 and self.margin is not None:
            im = crop(im, self.margin)

        return self.transform(image=im)['image'].float(), torch.tensor(self.label_transform(label))

    def image(self, idx): 
        im = self[idx][0].numpy()
        _, d1, d2 = im.shape
        mean = np.repeat(np.array([0.485, 0.456, 0.406]), d1 * d2).reshape((3, d1, d2))
        std = np.repeat(np.array([0.229, 0.224, 0.225]), d1 * d2).reshape((3, d1, d2))
        im = np.uint8((mean + im * std) * 256).transpose(1,2,0)
        return Image.fromarray(im)
    
    @property
    def labels(self):
        return [self.label_transform(label) for _, label in self.data]
   
    @property
    def estimated_num_labels(self):
        return max(self.labels) + 1
    
    @property
    def balanced_sampler(self):
        labels = self.labels
        c = Counter(labels)
        weights = [1 / c[l] for l in labels]
        return torch.utils.data.WeightedRandomSampler(weights, len(self.data))

    @property
    def balanced_weights(self):
        labels = self.labels
        c = Counter(labels)
        return torch.tensor([1 / c[l] for l in labels])
