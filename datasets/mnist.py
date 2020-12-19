from torch.utils.data import Subset, ConcatDataset
from PIL import Image
from torchvision.datasets import MNIST, EMNIST, FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from base.base_dataset import BaseADDataset
from .preprocessing import create_semisupervised_setting
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.vision import VisionDataset


from .noise import Noise_Dataset

import torch
import torchvision.transforms as transforms
import random
import numpy as np


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.1, noise="noise"):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        if normal_class!=-1:
            self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                            download=True)


        if normal_class != -1:
            # Create semi-supervised setting
            idx, ol, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
            
            # Subset train_set to semi-supervised setup
            self.train_set = Subset(train_set, idx)
        else:
            train_set.semi_targets = torch.zeros(len(train_set))
            self.train_set = train_set
            '''
            folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
            
            mnistC = []
            
            for i in range(15):
                a,b = divmod(i, 5)
                set_images = np.load("./datasets/MNIST-C/mnist_c/{}/test_images.npy".format(folders[i]))
                set_images = np.squeeze(set_images)
                l = set_images.shape[0]
            
                set_images = torch.Tensor(set_images)
                tensor = torch.ones((2,), dtype=torch.float64)
                f1 = tensor.new_full((l,), i)
                f2 = tensor.new_full((l,), -1)
                idx = torch.tensor(np.array(range(l)))
                
                print(set_images.shape, f1.shape, f2.shape, idx.shape)
                
                mnistC.append(TensorDataset(set_images, f1, f2, idx))
            
            self.train_set = ConcatDataset([self.train_set]+mnistC)
            
            '''
            target_transform = transforms.Lambda(lambda x: 0)
            self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)
            f = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]

            mnistc_test = MyMNISTC_Test(root=self.root, transform=transform, target_transform=None, folders = f)

            #print(len(mnistc_test.data))
            
            self.test_set = ConcatDataset([self.test_set, mnistc_test])
            
        
        
        #Draw random noise samples and concat-datasets -> DSVDD case
        if noise=="random":  
            if normal_class==-1:
                noise1 = Noise_Dataset(noise=("gaussian",), size=2000, image_size=(1,28,28))
                noise2 = Noise_Dataset(noise=("blobs",), size=2000, image_size=(1,28,28))
                noise3 = Noise_Dataset(noise=("bernoulli",), size=2000, image_size=(1,28,28))
            else:
                noise1 = Noise_Dataset(noise=("gaussian",), size=1000, image_size=(1,28,28))
                noise2 = Noise_Dataset(noise=("blobs",), size=1000, image_size=(1,28,28))
                noise3 = Noise_Dataset(noise=("bernoulli",), size=1000, image_size=(1,28,28))

            self.train_set = ConcatDataset([self.train_set, noise1.train_set, noise2.train_set, noise3.train_set])

        elif noise=="oe_emnist":
            outlier_set = MyEMNIST(root=self.root, split="letters", transform=transform, target_transform=target_transform, download=True, train=True)
            if normal_class==-1:
                ind = [random.randint(0,len(outlier_set)-1) for i in range(20000)]
            else:
                ind = [random.randint(0,len(outlier_set)-1) for i in range(2000)]
            outlier = Subset(outlier_set, ind)
            self.train_set = ConcatDataset([self.train_set, outlier])
            
        elif noise=="oe_fmnist":
            outlier_set = MyFashionMNIST(root=self.root, transform=transform, target_transform=target_transform, download=True, train=True)
            ind = [random.randint(0,len(outlier_set)-1) for i in range(2000)]
            outlier = Subset(outlier_set, ind)
            self.train_set = ConcatDataset([self.train_set, outlier])
            
        elif noise=="supervised_mnistc":
            f = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
            mnistc_train = MyMNISTC_Train(root=self.root, transform=transform, target_transform=None, folders = f)
            ind = [random.randint(0,len(mnistc_train)-1) for i in range(10000)]
            outlier_mnistc = Subset(mnistc_train, ind)
            
            
            outlier_set = MyEMNIST(root=self.root, split="letters", transform=transform, target_transform=target_transform, download=True, train=True)
            ind = [random.randint(0,len(outlier_set)-1) for i in range(10000)]
            outlier_emnist = Subset(outlier_set, ind)
            
            noise1 = Noise_Dataset(noise=("gaussian",), size=5000, image_size=(1,28,28))
            noise2 = Noise_Dataset(noise=("blobs",), size=10000, image_size=(1,28,28))
            noise3 = Noise_Dataset(noise=("bernoulli",), size=5000, image_size=(1,28,28))
            
            
            self.train_set = ConcatDataset([self.train_set, outlier_mnistc, outlier_emnist,  noise1.train_set, noise2.train_set, noise3.train_set])
            print("nice-supervised")



        

        # Get test set
        if normal_class != -1:
            self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                    download=True)
            
    def load_new_test_set(self, mnistc_partition):
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: 0)
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform, download=True)
        mnistc_test = MyMNISTC_Test(root=self.root, transform=transform, target_transform=None, folders=mnistc_partition)
        self.test_set = ConcatDataset([self.test_set, mnistc_test])
        return
            

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
