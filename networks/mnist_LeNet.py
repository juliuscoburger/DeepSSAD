import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=32, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        
        B = loss=="hypersphere"
        B = True
        
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, bias=B, padding=2),
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(8, 4, 5, bias=B, padding=2),
            nn.BatchNorm2d(4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        
    
        self.l3 = nn.Sequential(nn.Linear(4 * 7 * 7, self.rep_dim, bias=B))
        self.l4 = nn.Sequential(nn.ReLU(),nn.Linear(rep_dim, 1, bias=B))
    
        


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.l1(x)
        x = self.l2(x)
        
        x = x.view(int(x.size(0)), -1)
        x = self.l3(x)
        return x


class MNIST_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=32, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MNIST_LeNet(rep_dim=rep_dim, loss=loss)
        self.decoder = MNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
