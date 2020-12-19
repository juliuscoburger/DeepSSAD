import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets.vision import VisionDataset
from skimage.filters import gaussian as gblur


class Noise_Dataset(TorchvisionDataset):

    def __init__(self, noise=('gaussian',), size=1000, image_size=(3, 224, 224), offset=0,
                 data_augmentation: bool = False):
        super().__init__(None)

        self.n_classes = 1  # only class 1: outlier since noise is used for outlier exposure
        self.shuffle = False
        transform = []
        if data_augmentation:
            transform += [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(image_size[1], padding=4)]
        transform += [transforms.ToTensor()]
        if data_augmentation:
            transform += [transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))]
        transform = transforms.Compose(transform)

        # Get dataset
        self.train_set = Noise(noise, size, image_size, transform, offset)
        self.test_set = None
        
        
class Noise(VisionDataset):
    """A fake dataset that returns artificial random noise images.
    Args:
        noise (tuple, optional): Tuple of strings that specify the different types of noise to generate. Can be
            ('gaussian', 'uniform_0_1', 'uniform_-1_1', 'bernoulli', 'rademacher', 'blobs').
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        offset (int): Offsets the index-based random seed used to generate each image. Default: 0
    """

    def __init__(self, noise=('gaussian',), size=1000, image_size=(3, 224, 224), transform=None, offset=0):
        super(Noise, self).__init__(None, transform=transform)
        self.noise = noise
        self.size = size
        self.image_size = image_size
        self.offset = offset
        self.shuffle_idxs = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, semi_target, index)
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.offset)
        random.seed(index + self.offset)
        index = (index + self.offset) % self.size
        
        rand_noise = random.sample(self.noise, 1)[0]  # draw random type of noise

        if rand_noise == 'gaussian':
            img = math.sqrt(0.5) * torch.randn(*self.image_size, dtype=torch.float32)  # Draw from N(0, 0.5)
            img = img.clamp(-1, 1)  # clip to [-1, 1] range
        if rand_noise == 'uniform_0_1':
            img = torch.rand(*self.image_size, dtype=torch.float32)  # Draw from U(0, 1)
        if rand_noise == 'uniform_-1_1':
            img = 2 * torch.rand(*self.image_size, dtype=torch.float32) - 1  # Draw from U(-1, 1)
        if rand_noise == 'bernoulli':
            img = torch.empty(*self.image_size, dtype=torch.float32).fill_(0.5)
            img = torch.bernoulli(img)
        if rand_noise == 'rademacher':
            img = torch.empty(*self.image_size, dtype=torch.float32).fill_(0.5)
            img = 2 * torch.bernoulli(img) - 1
        if rand_noise == 'blobs':
            img = np.float32(np.random.binomial(n=1, p=0.7, size=self.image_size[::-1]))
            img = gblur(img, sigma=1.5, multichannel=False)
            img[img < 0.75] = 0.0
            img = 2 * torch.from_numpy(img.transpose((2, 0, 1))) - 1
        if rand_noise == 'textures':
            pass
            # TODO: Implement textures dataset
        torch.set_rng_state(rng_state)
        
        
        # convert to PIL Image
        img = img.to(torch.float32)
        
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        

        return img, torch.tensor(1), torch.tensor(-1), torch.tensor(index)
        #return img, 1, -1, index

    def __len__(self):
        return self.size
