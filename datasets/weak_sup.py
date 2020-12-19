from torch.utils.data import Subset, ConcatDataset
from PIL import Image
from torchvision.datasets import MNIST, EMNIST, FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from base.base_dataset import BaseADDataset
from .preprocessing import create_semisupervised_setting
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.vision import VisionDataset

from .imagenet22k import ImageNet22K_Dataset


from .noise import Noise_Dataset
from .MVTec import MVTec_Dataset

import torch
import torchvision.transforms as transforms
import random
import numpy as np
from kornia import gaussian_blur2d

from skimage.transform import rotate as im_rotate
from datasets.MVTec import MVTec, MVTec_Masks


def ceil(x: float):
        return int(np.ceil(x))


def floor(x: float):
    return int(np.floor(x))


def confetti_noise(size: torch.Size, p: float = 0.01, blobshaperange: ((int, int), (int, int)) = ((3, 3), (5, 5)), fillval: int = 255, backval: int = 0, ensureblob: bool = True, awgn: float = 0.0, clamp: bool = False, onlysquared: bool = True, rotation: int = 0, colorrange: (int, int) = None) -> torch.Tensor:
    """
    Generates "confetti" noise, as seen in the paper.
    The noise is based on sampling randomly many rectangles (in the following called blobs) at random positions.
    Additionally, all blobs are of random size (within some range), of random rotation, and of random color.
    The color is randomly chosen per blob, thus consistent within one blob.
    :param size: size of the overall noise image(s), should be (n x h x w) or (n x c x h x w), i.e.
        number of samples, channels, height, width. Blobs are grayscaled for (n x h x w) or c == 1.
    :param p: the probability of inserting a blob per pixel.
        The average number of blobs in the image is p * h * w.
    :param blobshaperange: limits the random size of the blobs. For ((h0, h1), (w0, w1)), all blobs' width
        is ensured to be in {w0, ..., w1}, and height to be in {h0, ..., h1}.
    :param fillval: if the color is not randomly chosen (see colored parameter), this sets the color of all blobs.
        This is also the maximum value used for clamping (see clamp parameter). Can be negative.
    :param backval: the background pixel value, i.e. the color of pixels in the noise image that are not part
        of a blob. Also used for clamping.
    :param ensureblob: whether to ensure that there is at least one blob per noise image.
    :param awgn: amount of additive white gaussian noise added to all blobs.
    :param clamp: whether to clamp all noise image to the pixel value range (backval, fillval).
    :param onlysquared: whether to restrict the blobs to be squares only.
    :param rotation: the maximum amount of rotation (in degrees)
    :param colorrange: the range of possible color values for each blob and channel.
        Defaults to None, where the blobs are not colored, but instead parameter fillval is used.
        First value can be negative.
    :return: torch tensor containing n noise images. Either (n x c x h x w) or (n x h x w), depending on size.
    """
    
    assert len(size) == 4 or len(size) == 3, 'size must be n x c x h x w'
    if isinstance(blobshaperange[0], int) and isinstance(blobshaperange[1], int):
        blobshaperange = (blobshaperange, blobshaperange)
    assert len(blobshaperange) == 2
    assert len(blobshaperange[0]) == 2 and len(blobshaperange[1]) == 2
    assert colorrange is None or len(size) == 4 and size[1] == 3
    out_size = size
    colors = []
    if len(size) == 3:
        size = (size[0], 1, size[1], size[2])  # add channel dimension
    else:
        size = tuple(size)  # Tensor(torch.size) -> tensor of shape size, Tensor((x, y)) -> Tensor with 2 elements x & y
    mask = (torch.rand((size[0], size[2], size[3])) < p).unsqueeze(1)  # mask[i, j, k] == 1 for center of blob
    while ensureblob and (mask.view(mask.size(0), -1).sum(1).min() == 0):
        idx = (mask.view(mask.size(0), -1).sum(1) == 0).nonzero().squeeze()
        s = idx.size(0) if len(idx.shape) > 0 else 1
        mask[idx] = (torch.rand((s, 1, size[2], size[3])) < p)
    res = torch.empty(size).fill_(backval).int()
    idx = mask.nonzero()  # [(idn, idz, idy, idx), ...] = indices of blob centers
    if idx.reshape(-1).size(0) == 0:
        return torch.zeros(out_size).int()

    all_shps = [
        (x, y) for x in range(blobshaperange[0][0], blobshaperange[1][0] + 1)
        for y in range(blobshaperange[0][1], blobshaperange[1][1] + 1) if not onlysquared or x == y
    ]
    picks = torch.FloatTensor(idx.size(0)).uniform_(0, len(all_shps)).int()  # for each blob center pick a shape
    nidx = []
    for n, blobshape in enumerate(all_shps):
        if (picks == n).sum() < 1:
            continue
        bhs = range(-(blobshape[0] // 2) if blobshape[0] % 2 != 0 else -(blobshape[0] // 2) + 1, blobshape[0] // 2 + 1)
        bws = range(-(blobshape[1] // 2) if blobshape[1] % 2 != 0 else -(blobshape[1] // 2) + 1, blobshape[1] // 2 + 1)
        extends = torch.stack([
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.arange(bhs.start, bhs.stop).repeat(len(bws)),
            torch.arange(bws.start, bws.stop).unsqueeze(1).repeat(1, len(bhs)).reshape(-1)
        ]).transpose(0, 1)
        nid = idx[picks == n].unsqueeze(1) + extends.unsqueeze(0)
        if colorrange is not None:
            col = torch.randint(
                colorrange[0], colorrange[1], (3, )
            )[:, None].repeat(1, nid.reshape(-1, nid.size(-1)).size(0)).int()
            colors.append(col)
        nid = nid.reshape(-1, extends.size(1))
        nid = torch.max(torch.min(nid, torch.LongTensor(size) - 1), torch.LongTensor([0, 0, 0, 0]))
        nidx.append(nid)
    idx = torch.cat(nidx)  # all pixel indices that blobs cover, not only center indices
    shp = res[idx.transpose(0, 1).numpy()].shape
    if colorrange is not None:
        colors = torch.cat(colors, dim=1)
        gnoise = (torch.randn(3, *shp) * awgn).int() if awgn != 0 else (0, 0, 0)
        res[idx.transpose(0, 1).numpy()] = colors[0] + gnoise[0]
        res[(idx + torch.LongTensor((0, 1, 0, 0))).transpose(0, 1).numpy()] = colors[1] + gnoise[1]
        res[(idx + torch.LongTensor((0, 2, 0, 0))).transpose(0, 1).numpy()] = colors[2] + gnoise[2]
    else:
        gnoise = (torch.randn(shp) * awgn).int() if awgn != 0 else 0
        res[idx.transpose(0, 1).numpy()] = torch.ones(shp).int() * fillval + gnoise
        res = res[:, 0, :, :]
        if len(out_size) == 4:
            res = res.unsqueeze(1).repeat(1, out_size[1], 1, 1)
    if clamp:
        res = res.clamp(backval, fillval) if backval < fillval else res.clamp(fillval, backval)
    mask = mask[:, 0, :, :]
    if rotation > 0:
        idx = mask.nonzero()
        res = res.unsqueeze(1) if res.dim() != 4 else res
        res = res.transpose(1, 3).transpose(1, 2)
        for pick, blbctr in zip(picks, mask.nonzero()):
            rot = np.random.uniform(-rotation, rotation)
            p1, p2 = all_shps[pick]
            dims = (
                blbctr[0],
                slice(max(blbctr[1] - floor(0.75 * p1), 0), min(blbctr[1] + ceil(0.75 * p1), res.size(1) - 1)),
                slice(max(blbctr[2] - floor(0.75 * p2), 0), min(blbctr[2] + ceil(0.75 * p2), res.size(2) - 1)),
                ...
            )
            res[dims] = torch.from_numpy(
                im_rotate(
                    res[dims].float(), rot, order=0, cval=0, center=(blbctr[1]-dims[1].start, blbctr[2]-dims[2].start),
                    clip=False
                )
            ).int()
        res = res.transpose(1, 2).transpose(1, 3)
        res = res.squeeze() if len(out_size) != 4 else res
    return res

def smooth_noise(img: torch.Tensor, ksize: int, std: float, p: float = 1.0, inplace: bool = True) -> torch.Tensor:
    """
    Smoothens (blurs) the given noise images with a Gaussian kernel.
    :param img: torch tensor (n x c x h x w).
    :param ksize: the kernel size used for the Gaussian kernel.
    :param std: the standard deviation used for the Gaussian kernel.
    :param p: the chance smoothen an image, on average smoothens p * n images.
    :param inplace: whether to apply the operation inplace.
    """
    if not inplace:
        img = img.clone()
    ksize = ksize if ksize % 2 == 1 else ksize - 1
    picks = torch.from_numpy(np.random.binomial(1, p, size=img.size(0))).bool()
    if picks.sum() > 0:
        img[picks] = gaussian_blur2d(img[picks].float(), (ksize, ) * 2, (std, ) * 2).int()
    return img

def confetti_image(image):
    image = image * 255
    size = ((1,3,224,224))
    
    generated_noise_rgb = confetti_noise(size, 0.000018, ((8, 8), (54, 54)), fillval=255, clamp=False, awgn=0, rotation=45, colorrange=(-256, 0))
    generated_noise = confetti_noise(size, 0.000012, ((8, 8), (54, 54)), fillval=-255, clamp=False, awgn=0, rotation=45)
    generated_noise = generated_noise_rgb + generated_noise
    generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
    
            
    generated_noise = generated_noise.int()
    
    anom = (image.int() + generated_noise).clamp(0,255).byte()
    anom = anom[0] * (1/255)
    
    return anom



class weak_supervision_Dataset(TorchvisionDataset):
    

    def __init__(self, root: str, size: int = 1, dataset: str = "emnist", nc=1):
        super().__init__(root)
        
        self.size = size
        random.seed(size)

        transform = transforms.ToTensor()
        
        if dataset == "emnist":
            outlier_set = MyEMNIST(root=self.root, split="letters", transform=transform, download=True, train=True)
            
        if dataset == "mnistc":
            f = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
            outlier_set = MyMNISTC_Train(root=self.root, transform=transform, folders = f)
            
        if dataset == "fmnist":
            outlier_set = MyFashionMNIST(root=self.root, transform=transform, target_transform=target_transform, download=True, train=True)
        
        
        if dataset == "mvtec_noise":
            noise1 = Noise_Dataset(noise=("gaussian",), size=5000)
            noise2 = Noise_Dataset(noise=("blobs",), size=5000)
            noise3 = Noise_Dataset(noise=("bernoulli",), size=5000)
            outlier_set = ConcatDataset([noise1.train_set, noise2.train_set, noise3.train_set])
            
            
        
            
                
            
        
        if dataset == "mvtec_imagenet":
            print("imagenet")
            self.train_set = ImageNet22K_Dataset(root=self.root, size=size, seed=size).train_set
            self.test_set = None
            print(len(self.train_set))
        elif dataset == "mvtec_confetti":
            print("confetti")
            dataset = MVTec_Dataset(root=self.root, normal_class=nc)
            print(type(dataset.train_set))
            
            ind = [(random.randint(0,len(dataset.train_set)-1))%len(dataset.train_set) for i in range(self.size)]
            outlier_set = Subset(dataset.train_set, ind)
            outlier = []
            for i in range(len(outlier_set)):
                noise_img = confetti_image(outlier_set[i][0])
                outlier.append(noise_img)
                
            outlier = torch.stack(outlier)
            print(outlier.shape)
            f1 = torch.full((self.size,), 1, dtype=int)
            f2 = torch.full((self.size,), -1, dtype=int)
            idx = torch.tensor(np.array(range(self.size)))
            
            self.train_set = TensorDataset(outlier, f1, f2, idx)
            print(len(self.train_set))
            
        elif dataset == "mvtec_all":
            print("mvtec_all")
            #imgnet_set = ImageNet22K_Dataset(root=self.root, size=size, seed=size).train_set
            
            dataset = MVTec_Dataset(root=self.root, normal_class=nc)
            print(type(dataset.train_set))
            
            ind = [(random.randint(0,len(dataset.train_set)-1))%len(dataset.train_set) for i in range(self.size)]
            outlier_set = Subset(dataset.train_set, ind)
            outlier = []
            for i in range(len(outlier_set)):
                noise_img = confetti_image(outlier_set[i][0])
                outlier.append(noise_img)
                
            outlier = torch.stack(outlier)
            print(outlier.shape)
            f1 = torch.full((self.size,), 1, dtype=int)
            f2 = torch.full((self.size,), -1, dtype=int)
            idx = torch.tensor(np.array(range(self.size)))
            
            confet_set = TensorDataset(outlier, f1, f2, idx)
            
            noise1 = Noise_Dataset(noise=("gaussian",), size=int(self.size/3))
            noise2 = Noise_Dataset(noise=("blobs",), size=int(self.size/3))
            noise3 = Noise_Dataset(noise=("bernoulli",), size=int(self.size/3))
            noise_set = ConcatDataset([noise1.train_set, noise2.train_set, noise3.train_set])
            
            
            self.train_set = ConcatDataset([noise_set, confet_set])#,imgnet_set])
            print(len(self.train_set))
            
        else:
            ind = [random.randint(0,len(outlier_set)-1) for i in range(self.size)]
            outlier = Subset(outlier_set, ind)
            self.train_set = outlier
            self.test_set = None
                        

class MyEMNIST(EMNIST):
    """
    Torchvision EMNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample. -> used for outlier exposure
    """

    def __init__(self, *args, **kwargs):
        super(MyEMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        img.to(torch.float32)

        return img, 1, -1, index
    
    
class MyFashionMNIST(FashionMNIST):
    """
    Torchvision FashionMNIST class with additional targets for the semi-supervised setting and patch of __getitem__
    method to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MyFashionMNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            
        img.to(torch.float32)


        return img, 1, -1, index



class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        img.to(torch.float32)

        return img, target, semi_target, index
    
    
class MyMNISTC_Test(VisionDataset):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, root="", transform=None, target_transform=None, folders=None):
        super(MyMNISTC_Test, self).__init__(root)
        self.transform=transform
        self.target_transform=target_transform
        self.folders=folders

        #self.semi_targets = torch.zeros_like(self.targets)
        
        #folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
        self.folders= folders
        mnistC = []
        targets = []
        
        self.data = torch.Tensor()
        self.targets = torch.Tensor()
        
        for i in range(len(self.folders)):
            a,b = divmod(i, 5)
            set_images = np.load("./datasets/MNIST-C/mnist_c/{}/test_images.npy".format(self.folders[i]))
            set_images = np.squeeze(set_images)/255
            l = set_images.shape[0]
        
            set_images = torch.Tensor(set_images).float()
            tensor = torch.ones((2,), dtype=torch.float64)
            target = tensor.new_full((l,), i).float()
            mnistC.append(set_images)
            targets.append(target)
            
        torch.cat(mnistC, dim=0, out=self.data)
        torch.cat(targets, dim=0, out=self.targets)
        self.size = len(self.folders)*10000
        

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        img.to(torch.float32)

        return img, 1, -1, index
        #return img, target, -1, index
    
    def __len__(self):
        return self.size
    
class MyMNISTC_Train(VisionDataset):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, root="", transform=None, target_transform=None, folders=None):
        super(MyMNISTC_Train, self).__init__(root)
        self.transform=transform
        self.target_transform=target_transform
        self.folders=folders

        #self.semi_targets = torch.zeros_like(self.targets)
        
        #folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
        self.folders= folders
        mnistC = []
        targets = []
        
        self.data = torch.Tensor()
        self.targets = torch.Tensor()
        
        for i in range(len(self.folders)):
            a,b = divmod(i, 5)
            set_images = np.load("./datasets/MNIST-C/mnist_c/{}/train_images.npy".format(self.folders[i]))
            set_images = np.squeeze(set_images)/255
            l = set_images.shape[0]
        
            set_images = torch.Tensor(set_images).float()
            tensor = torch.ones((2,), dtype=torch.float64)
            target = tensor.new_full((l,), i).float()
            mnistC.append(set_images)
            targets.append(target)
            
        torch.cat(mnistC, dim=0, out=self.data)
        torch.cat(targets, dim=0, out=self.targets)
        self.size = len(self.folders)*60000
        

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        img.to(torch.float32)

        return img, 1, -1, index
        #return img, target, -1, index
    
    def __len__(self):
        return self.size
