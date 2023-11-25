"""
Reference for portions of this code:
PyTorch DCGAN Tutorial - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
This code includes modifications and adaptations for our specific use-case,
but the foundational concepts and some code segments are inspired by or derived from
the above PyTorch tutorial.
"""

"""
Author: Ei Tanaka
Date: Nov 25, 2023
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project
Purpose: This script trains a DCGAN model on the Stanford Dogs dataset.

Comment:
I tried to train the DC-GAN model with 90 epoch so far.
"""

# =================================== Import ===================================
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from stanford_dogs_data import DogImages

# =================================== Setup ===================================
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Using CPU')

# =================================== Constants ===================================
# Root directory for dataset
OS_PATH = os.getcwd()
os.chdir('..')
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Models')
os.chdir(OS_PATH)

# Inputs
WORKERS = 2   # Number of workers for dataloader
BATCH_SIZE = 64
IMAGE_SIZE = 64
NC = 3    # Number of channels in the training images. For color images this is 3
NZ = 100    # Size of z latent vector (i.e. size of generator input)
NGF = 64    # Size of feature maps in generator
NDF = 64    # Size of feature maps in discriminator
NUM_EPOCHS = 30    # Number of training epochs
LR = 0.0001    # Learning rate for optimizers
BETA1 = 0.5    # Beta1 hyperparam for Adam optimizers
ngpu = 1    # Number of GPUs available. Use 0 for CPU mode.

# pretrained
pretrained = True

# =================================== Data ===================================
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
])

dataset = DogImages(root=DATA_DIR, train=True, download=False, cropped=True, transform=transforms)

# Test the dataset
for i in range(10):
    image, label = dataset[i]
    print(image.shape, label)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()


# =================================== Models ===================================
# Weight initialization
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:    # Convolutional layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:    # BatchNorm layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    # Bias terms

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),
            # State size: (NGF*8) x 4 x 4
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),
            # State size: (NGF*4) x 8 x 8
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),
            # State size: (NGF*2) x 16 x 16
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # State size: NGF x 32 x 32
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: NC x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2
if pretrained:
    netG.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'DCGAN_netG.pth')))
else:
    netG.apply(weights_init)

print(netG)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main= nn.Sequential(
            # Input is (NC) x 64 x 64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: NDF x 32 x 32
            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*2) x 16 x 16
            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*4) x 8 x 8
            nn.Conv2d(NDF*4, NDF*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*8) x 4 x 4
            nn.Conv2d(NDF*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2
if pretrained:
    netD.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'DCGAN_netD.pth')))
else:
    netD.apply(weights_init)

# Print the model
print(netD)

# =================================== Loss Function and Optimizer ===================================
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

# =================================== Training ===================================
# Function to save the models
def save_best_model(netG, netD, epoch):
    torch.save(netG.state_dict(), os.path.join(MODEL_DIR, 'DCGAN_netG.pth'))
    torch.save(netD.state_dict(), os.path.join(MODEL_DIR, 'DCGAN_netD.pth'))

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
best_loss = 1000000.0
iters = 0

print('Starting Training Loop...')
# For each epoch
for epoch in range(NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_data).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, NZ, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, NUM_EPOCHS, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save the best model
        if errG.item() < best_loss:
            best_loss = errG.item()
            save_best_model(netG, netD, epoch)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# =================================== Results ===================================
# Plot the training losses
plt.figure(figsize=(10, 5))
plt.title('Generator and Discriminator Loss During Training')
plt.plot(G_losses, label='Generator')
plt.plot(D_losses, label='Discriminator')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, 'DCGAN_loss.png'))
plt.show()

# Visualize the fake images
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1,2,0)))
plt.savefig(os.path.join(MODEL_DIR, 'DCGAN_fake_images.png'))
plt.show()

# Real Images vs. Fake Images
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1,2,0)))
plt.savefig(os.path.join(MODEL_DIR, 'DCGAN_real_vs_fake.png'))
plt.show()
