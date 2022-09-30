# %%
# import libraries

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io

import numpy as np
import os, subprocess
import time
from matplotlib import pyplot as plt

# %%
# define parameter
saveFolder = "DCGAN"
modelName = "DCGAN_Torch_1"
epochs = 400 # total training epochs
continueWith = 28 # Continue training at number of epoch, 0 will restart training
batchSize = 50

# %%
# 
def progressBar(name, value, endvalue, comment="", bar_length = 50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent*bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f"\r{name} : [{arrow + spaces}]{round(percent*100):4}%  {comment}", flush=True, end='')
    if value == endvalue:     
        print("\n", flush=True, end='')

# %%
# create folder for save datas
print("Creating folder for saving datas")
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
if not os.path.exists( os.path.join(saveFolder, "resultInEpoch") ):
    os.mkdir( os.path.join(saveFolder, "resultInEpoch") )
if not os.path.exists( os.path.join(saveFolder, "models") ):
    os.mkdir( os.path.join(saveFolder, "models") )

# %%
# get devices
gpuIdx = 0
deviceGPU = torch.device(f"cuda:{gpuIdx}")
deviceCPU = torch.device("cpu:0")

device = deviceCPU
if torch.cuda.is_available():
    device = deviceGPU
    print(f"Using cuda with device-{gpuIdx} ({torch.cuda.get_device_name(gpuIdx)})")
else:
    cpuName = subprocess.check_output(["wmic","cpu","get", "name"]).decode("utf8").strip().split("\n")[1]
    print(f"Using CPU ({cpuName})")

# %%
# show states functions
def pltImages(images, numRows, numCols):
    for i,image in enumerate(images):
        plt.subplot(numRows, numCols, i+1)
        plt.imshow(image)

# %%
# define dataset
class AnimeFaceDataset(Dataset):
    def __init__(self, path, count=None, transform=None):
        files = os.listdir(path)

        # copy list
        files_temp = files

        # remove file which not image
        for file in files_temp:
            if(file.split('.')[-1] != 'jpg'):
                files.remove(file)

        # num of images to read
        if(count is not None and count<len(files)):
            files = files[:count]
    
        # save
        self.path = path
        self.files = np.array(files)
        self.transform = transform

    def __getitem__(self,index):
        img = io.read_image(os.path.join(self.path, self.files[index])).float()
        if(self.transform is not None):
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)

class AnimeFaceDataset_preload(Dataset):
    def __init__(self, path, count=None, transform=None):
        files = os.listdir(path)

        # copy list
        files_temp = files

        # remove file which not image
        for file in files_temp:
            if(file.split('.')[-1] != 'jpg'):
                files.remove(file)

        # num of images to read
        if(count is not None and count<len(files)):
            files = files[:count]

        # read images
        numFiles = len(files)
        self.images = None
        for i,file in enumerate(files):
            if(i%10==0)or((i+1) == numFiles):
                progressBar("Loadinn dataset", i+1, numFiles)
            # read image
            img = io.read_image(os.path.join(path, file)).float()
            # transform image
            if(transform is not None):
                img = transform(img)
            # save image
            if(self.images is not None):
                self.images = torch.cat((self.images, img.unsqueeze(0)), 0)
            else:
                self.images = img.unsqueeze(0)


    def __getitem__(self,index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

# %%
# create dataset
trans = transforms.Compose([transforms.Resize((64,64)), 
                            transforms.Normalize((0, 0, 0), (255, 255, 255))])
trainSet = AnimeFaceDataset("datas", count=None, transform = trans)

# %%
## debug test dataset
# print(len(trainSet))
# print(trainSet[0].shape)
# print(trainSet[0].min())
# realImages = np.array([trainSet[i].numpy() for i in range(4*4)])
# realImages = np.rollaxis(realImages, 1, 4)
# pltImages(realImages, 4, 4)
# plt.show()

# %%
# Define model
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(  in_channels=3,
                                                out_channels=64,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer2 = nn.Sequential(nn.Conv2d(  in_channels=64,
                                                out_channels=128,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer3 = nn.Sequential(nn.Conv2d(  in_channels=128,
                                                out_channels=256,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer4 = nn.Sequential(nn.Conv2d(  in_channels=256,
                                                out_channels=512,
                                                kernel_size=4, stride=2, padding=1), 
                                    nn.LeakyReLU(0.02, True))
        self.layer5 = nn.Sequential(nn.Dropout(0.2), 
                                    nn.Linear(4*4*512, 1), 
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=-3)
        x = self.layer5(x)

        return x

# %%
# Create model
genModel = Generator().to(deviceGPU)
print(genModel)
disModel = Discriminator().to(deviceGPU)
print(disModel)


# %%
# Define training parameter
loss_fns = {  "generator"     :nn.BCELoss(),
              "discriminator" :nn.BCELoss() }
optimizers = {"generator"     :torch.optim.Adam(genModel.parameters(), lr = 2e-4, betas=(0.5, 0.999) ),
              "discriminator" :torch.optim.Adam(disModel.parameters(), lr = 2e-4, betas=(0.5, 0.999) ) }
schedulers = {"generator"     :torch.optim.lr_scheduler.StepLR(optimizers["generator"], step_size=5, gamma=0.9),
              "discriminator" :torch.optim.lr_scheduler.StepLR(optimizers["discriminator"], step_size=5, gamma=0.9) }

# %%
# debug test code

# random_latent_vectors = torch.rand(size=(16,16)).to(deviceGPU)
# print("rand latent shape :", random_latent_vectors.shape)
# fackImgs = genModel(random_latent_vectors)
# fackImgs = fackImgs.detach().to(deviceCPU).numpy()
# print(fackImgs.shape)
# fackImgs = np.rollaxis(fackImgs, 1, 4)
# print(fackImgs.shape)
# pltImages(fackImgs, 4, 4)
# plt.show()

# print("fake images shape :", fackImgs.shape)
# pred = disModel(fackImgs)
# print("pred shape :", pred.shape)

# print("trainSet len :", len(trainSet))
# dataloader = DataLoader(trainSet, batch_size=100, shuffle=True)
# realImages = iter(dataloader).next()
# realImages = realImages.to(deviceGPU)
# discriminatorSet = DiscriminatorDataset(realImages, fackImgs)
# print("disSet len :", len(discriminatorSet))

# %% 
# define training process
def train(dataloader, generator, discriminator, epochs, lossFns, optimizers, schedulers=None, startWith = 0, loadOptimizer=True, loadScheduler=True):
    numLatent = 8*8*128
    
    # generate Random_Latent_Vectors for showing result
    # pltRandomLatentVectors = torch.rand(size=(4*4,numLatent), device=deviceGPU)
    pltRandomLatentVectors = torch.empty(size=(4*4,numLatent), device=deviceGPU).normal_(mean=0,std=1.0)

    # load or create files
    RandomLatentFileName = os.path.join(saveFolder, "resultInEpoch", 'Random_Latent_Vectors.pt')
    lossesLogFileName = "loss_log.txt"

    if(startWith == 0):# new training
        torch.save(pltRandomLatentVectors, RandomLatentFileName)

        # create loss_log file
        with open(os.path.join(saveFolder, lossesLogFileName), mode="w") as logfile:
            logfile.write(f"losses of {modelName}\n")
    else:# continue training, loading previes paramater
        print(f"Continue previes training with epoch-{startWith}")
        # load test RandomLatent
        pltRandomLatentVectors = torch.load(RandomLatentFileName)

        # load model
        print("Loading model")
        checkpoint = torch.load( os.path.join(saveFolder, "models", f"{modelName}_epoch_{startWith}") )

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        if(loadOptimizer):
        # load optimizers
            optimizers["generator"].load_state_dict(checkpoint['optimizers_generator_state_dict'])
            optimizers["discriminator"].load_state_dict(checkpoint['optimizers_discriminator_state_dict'])
           

        # load schedulers
        if(schedulers is not None)and loadScheduler:
            schedulers["generator"].load_state_dict(checkpoint['schedulers_generator_state_dict'])
            schedulers["discriminator"].load_state_dict(checkpoint['schedulers_discriminator_state_dict'])

    print("Training start")
    numBatch = len(dataloader)
    startTime= time.time()
    for epoch in range(startWith, epochs):
        epochStartTime= time.time()
        totalDisLoss, totalGenLoss = 0.0, 0.0

        generator.train()
        discriminator.train()
        for batch, realImages in enumerate(dataloader):
            ## print training states
            progressBar(f"epochs {epoch+1:4}/{epochs}", batch+1, numBatch)

            ## preprocess
            realImages = realImages.to(deviceGPU)

            ## training discriminator

            # generate fake images
            # random_latent_vectors = torch.rand(size=(len(realImages),numLatent), device=deviceGPU) # uniform(0, 1)
            random_latent_vectors = torch.empty(size=(len(realImages),numLatent), device=deviceGPU).normal_(mean=0,std=1.0) # normal(0, 1)
            fakeImages = generator(random_latent_vectors)

            # update discriminator
            optimizers["discriminator"].zero_grad()
            realPred = discriminator(realImages)
            readTarget = torch.ones((len(realImages),1), device=deviceGPU)
            realLoss = lossFns["discriminator"](realPred, readTarget)
            fakePred = discriminator(fakeImages.detach())
            fakeTarget = torch.zeros((len(realImages),1), device=deviceGPU)
            fakeLoss = lossFns["discriminator"](fakePred, fakeTarget)
            disLoss = (realLoss + fakeLoss) / 2
            currDisLoss = disLoss.item()
            totalDisLoss += currDisLoss
            disLoss.backward()
            optimizers["discriminator"].step()

            ## training generator
            fakeImages = generator(random_latent_vectors)
            # update generator
            optimizers["generator"].zero_grad()
            genLoss = lossFns["generator"](discriminator(fakeImages), torch.ones((len(fakeImages),1)).to(deviceGPU))
            currGenLoss = genLoss.item()
            totalGenLoss += currGenLoss
            genLoss.backward()
            optimizers["generator"].step()
            
        # update scheduler
        if(schedulers is not None):
            schedulers["generator"].step()
            schedulers["discriminator"].step()

        print(f"    Losses[G:{totalGenLoss/numBatch:9.4f} | D:{totalDisLoss/numBatch:9.4f}] || ", end='')
        print(f"lr[G:{optimizers['generator'].param_groups[0]['lr']:7.2e} | D:{optimizers['discriminator'].param_groups[0]['lr']:7.2e}] || ", end='')
        print(f"taked {time.time()-epochStartTime:7.2f}s")


        ## save model
        print("    saving models", end="")
        checkpointName = os.path.join(saveFolder, "models", f"{modelName}_epoch_{epoch+1}")
        checkpointData ={   "generator_state_dict" : generator.state_dict(),
                            "discriminator_state_dict" : discriminator.state_dict(),
                            "optimizers_generator_state_dict" : optimizers["generator"].state_dict(),
                            "optimizers_discriminator_state_dict" : optimizers["discriminator"].state_dict()
                        }
        if(schedulers is not None):
            checkpointData["schedulers_generator_state_dict"] = schedulers["generator"].state_dict()
            checkpointData["schedulers_discriminator_state_dict"] = schedulers["discriminator"].state_dict()
        torch.save(checkpointData, checkpointName)

        # delete part of model to save space
        if ((epoch>1) and ((epoch+1)%10)!=1):
            print("    removing prev model", end="")
            os.remove(os.path.join(saveFolder, "models", f"{modelName}_epoch_{epoch}"))

        generator.eval()
        discriminator.eval()

        ## save fake images
        print("    saving fake images", end="")
        fakeImages = generator(pltRandomLatentVectors)
        fakeImages = fakeImages.detach().to(deviceCPU).numpy()
        fakeImages = np.rollaxis(fakeImages, 1, 4)
        pltImages(fakeImages, 4, 4)
        imagesFileName = f"epoch_{epoch+1}.png"
        plt.savefig(os.path.join(saveFolder, "resultInEpoch", imagesFileName), dpi=200)

        ## save loss
        print("    saving loss", end="\n\r")
        with open(os.path.join(saveFolder, lossesLogFileName), mode="a") as logfile:
            logfile.write(f"G-Loss:{totalGenLoss/numBatch:15.9f} | D-Loss:{totalDisLoss/numBatch:15.9f}\n")
        
    print(f"training taked {time.time()-startTime:7.2f}s")

# %%
# training model
dataloader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
train(dataloader, genModel, disModel, epochs, loss_fns, optimizers, schedulers=schedulers, startWith=continueWith)