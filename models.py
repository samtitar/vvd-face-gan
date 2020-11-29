import torch
import torch.nn as nn

NF = 256

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size, NF * 8, kernel_size=4, 
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(NF * 8),
            nn.ReLU(True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(NF * 8, NF * 4, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.ReLU(True))

        self.layer3 = nn.Sequential(    
            nn.ConvTranspose2d(NF * 4, NF * 2, kernel_size=4, 
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF * 2),
            nn.ReLU(True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(NF * 2, NF, kernel_size=4, 
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF),
            nn.ReLU(True))
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(NF, 3, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh())

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y) 
        y = self.layer3(y)  
        y = self.layer4(y)  

        return self.layer5(y)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, NF, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(NF, NF * 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer3 = nn.Sequential(    
            nn.Conv2d(NF * 2, NF * 4, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(NF * 4, NF * 8, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NF * 8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(NF * 8, 1, kernel_size=4,
                      stride=1, padding=0, bias=False),
            nn.Sigmoid())
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y) 
        y = self.layer3(y)  
        y = self.layer4(y)  

        return self.layer5(y)