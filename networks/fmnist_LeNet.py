import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class FashionMNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=64, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        
        B = loss=="hypersphere"
            
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, bias=B, padding=2),
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, bias=B, padding=2),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            )
        
   
        self.l3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128, bias=B),
            nn.BatchNorm1d(128, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.Linear(128, self.rep_dim, bias=B)
            )



    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.l1(x)
        x = self.l2(x)
        x = x.view(int(x.size(0)), -1)
        x = self.l3(x)
        return x


class FashionMNIST_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc3 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(8, 32, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(128 / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class FashionMNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=64, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = FashionMNIST_LeNet(rep_dim=rep_dim, loss=loss)
        self.decoder = FashionMNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
