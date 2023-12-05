"""

Author: Kismat Khatri
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project
Purpose: This script trains a Traditional GAN model on the Stanford Dogs dataset.
"""

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from stanford_dogs_data import DogImages  # Ensure this module is available

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Constants
DATA_DIR = "/home/ubuntu/Deep-Learning/Data/"
MODEL_DIR = "/home/ubuntu/Deep-Learning/Models/"
BATCH_SIZE = 64
IMAGE_SIZE = 64
NC = 3 
NZ = 100  # Size of z latent vector (i.e. size of generator input)
NGF = 256  # Size of feature maps in generator
NDF = 256  # Size of feature maps in discriminator
NUM_EPOCHS = 30  # Number of training epochs
LR = 0.0002  # Learning rate for optimizers
BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers

# Load the dataset
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = DogImages(root=DATA_DIR, train=True, download=False, cropped=True, transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =================================== Generator ===================================
class Generator(nn.Module):
    def __init__(self, NZ, NGF, NC):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(NZ, NGF),
            nn.BatchNorm1d(NGF),
            nn.ReLU(True),
            nn.Linear(NGF, NGF * 2),
            nn.BatchNorm1d(NGF * 2),
            nn.ReLU(True),
            nn.Linear(NGF * 2, NGF * 4),
            nn.BatchNorm1d(NGF * 4),
            nn.ReLU(True),
            nn.Linear(NGF * 4, NC * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, NC, IMAGE_SIZE, IMAGE_SIZE)

# =================================== Discriminator ===================================
class Discriminator(nn.Module):
    def __init__(self, NDF, NC):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(NC * IMAGE_SIZE * IMAGE_SIZE, NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(NDF * 4, NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(NDF * 2, NDF),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(NDF, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = input.view(-1, NC * IMAGE_SIZE * IMAGE_SIZE)
        return self.main(output)

# Create the generator and discriminator
netG = Generator(NZ, NGF, NC).to(device)
netD = Discriminator(NDF, NC).to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

netG.apply(weights_init)
netD.apply(weights_init)

# Print the models
print(netG)
print(netD)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

real_label = 1
fake_label = 0

# Training loop
print('Starting Training Loop...')
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        # Generate batch of latent vectors
        noise = torch.randn(b_size, NZ, device=device)
        fake = netG(noise)

        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, NUM_EPOCHS, i, len(dataloader), errD.item(), errG.item(), output.mean().item(), errG.item()))

# Save models
torch.save(netG.state_dict(), os.path.join(MODEL_DIR, 'GAN_netG.pth'))
torch.save(netD.state_dict(), os.path.join(MODEL_DIR, 'GAN_netD.pth'))

# Generate and save images
with torch.no_grad():
    fake = netG(torch.randn(64, NZ, device=device)).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Generated Images')
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
plt.savefig(os.path.join(MODEL_DIR, 'GAN_generated_images.png'))
plt.show()
