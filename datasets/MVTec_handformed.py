from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets import ImageFolder
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torchvision.transforms as transforms

class MVTec_Dataset_handformed(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, normalize: bool = False,
                 blur_outliers: bool = True, blur_std: float = 1.0):
        super().__init__(root)
        
        classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_class = normal_class

        # MVTec pre-processing (Resize, CenterCrop, feature normalization and scaling)
        if normalize:
            self.feature_min = -1
            self.feature_max = 1
        else:
            self.feature_min = 0
            self.feature_max = 1
            
        mvtec_min = -2.118
        mvtec_max = 2.640
        img_size = 224

        train_transform = [
            transforms.Resize(img_size),
            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size),
            transforms.RandomAffine(180, translate=(0.05, 0.05)),
            transforms.ToTensor()
        ]
        test_transform = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ]

        if normalize:
            train_transform += [
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.Normalize(mean=(mvtec_min, mvtec_min, mvtec_min),
                                     std=((mvtec_max - mvtec_min), (mvtec_max - mvtec_min), (mvtec_max - mvtec_min))),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
            test_transform += [
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.Normalize(mean=(mvtec_min, mvtec_min, mvtec_min),
                                     std=((mvtec_max - mvtec_min), (mvtec_max - mvtec_min), (mvtec_max - mvtec_min))),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        self.train_set = MVTec(root=self.root + 'handformed',
                               transform=train_transform)
        
                

class MVTec(ImageFolder):
    """MVTec Torchvision ImageFolder class with __getitem__ method patch for anomaly detection setting."""
    
    def __init__(self, blur_outliers: bool = False, blur_std: float = 1.0, *args, **kwargs):
        super(MVTec, self).__init__(*args, **kwargs)

        self.blur_outliers = blur_outliers
        self.blur_std = blur_std
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
                        
        target = int(self.classes[target] != 'good')  # 0: normal, 1: outlier

        if (target == 1) and self.blur_outliers:
            sample = torch.from_numpy(np.stack((
                gaussian_filter(sample[0, ...].numpy(), sigma=self.blur_std),
                gaussian_filter(sample[1, ...].numpy(), sigma=self.blur_std),
                gaussian_filter(sample[2, ...].numpy(), sigma=self.blur_std)
            )))
            
        return sample, 1, -1, index


class MVTec_Masks(ImageFolder):
    """
    MVTec ground-truth masks Torchvision ImageFolder class with __getitem__ method patch for anomaly detection setting.
    """
    
    def __init__(self, *args, **kwargs):
        super(MVTec_Masks, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
            
        target = int(self.classes[target] != 'good')  # 0: normal, 1: outlier

        return sample, 1, -1, index
