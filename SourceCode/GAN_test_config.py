numRandomLatent = 8*8*128
modelFileName = "DCGAN_Torch_1_epoch_100"

# Define model
from torch import nn
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(8*8*128, 8*8*64), nn.LeakyReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(  in_channels=64,
                                                out_channels=128,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(  in_channels=128,
                                                out_channels=256,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(  in_channels=256,
                                                out_channels=512,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer5 = nn.Sequential(nn.Conv2d(  in_channels=512,
                                                out_channels=3,
                                                kernel_size=5, stride=1, padding='same'), 
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1,64, 8, 8)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x