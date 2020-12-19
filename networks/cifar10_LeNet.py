import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        
        B = loss=="hypersphere"
        
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, bias=B, padding=2),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, bias=B, padding=2),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, bias=B, padding=2),
            nn.BatchNorm2d(128, eps=1e-04, affine=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            )
        
        self.l4 = nn.Sequential(nn.Linear(128 * 4 * 4, self.rep_dim, bias=B))


        

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x.view(int(x.size(0)), -1)
        x = self.l4(x)
        return x


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128, loss="normal"):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim, loss=loss)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
