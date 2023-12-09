# Imports
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from stanford_dogs_data_CycleGAN import DogImages
import itertools

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


# Root directory for dataset
OS_PATH = os.getcwd()
os.chdir('..')
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Models')
os.chdir(OS_PATH)



# Hyperparameters
IMAGE_SIZE = 64
CHANNELS_IMG = 3
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETAS = (BETA1, 0.999)
BATCH_SIZE = 64
WORKERS = 2
NUM_EPOCHS = 30

# =================================== Data ===================================
# Transformations (assuming both domains use the same transformations)
transforms_color = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_gray = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Correctly use the defined transformations
dataset_A = DogImages(root=DATA_DIR, exclude_breed='German Shepherd', include_only_breed=None, train=True, download=False, cropped=True, transform=transforms_color)
dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

dataset_B = DogImages(root=DATA_DIR, exclude_breed=None, include_only_breed='German Shepherd', train=True, download=False, cropped=True, transform=transforms_color)
dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)



# Plot some training images from Domain A
real_batch_A = next(iter(dataloader_A))
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Training Images from Domain A')
plt.imshow(np.transpose(vutils.make_grid(real_batch_A[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()

# Optionally, plot some training images from Domain B
real_batch_B = next(iter(dataloader_B))
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Training Images from Domain B')
plt.imshow(np.transpose(vutils.make_grid(real_batch_B[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()

# Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 64, normalize=False),  # Random noise dimension
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, IMAGE_SIZE * IMAGE_SIZE * CHANNELS_IMG),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)
        return img


# Discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE * CHANNELS_IMG, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


gen_AtoB = Generator().to(device)
gen_BtoA = Generator().to(device)
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)

# Set up optimizers for both generators and discriminators
optimizer_gen = torch.optim.Adam(itertools.chain(gen_AtoB.parameters(), gen_BtoA.parameters()), lr=LEARNING_RATE,
                                 betas=BETAS)
optimizer_disc = torch.optim.Adam(itertools.chain(disc_A.parameters(), disc_B.parameters()), lr=LEARNING_RATE,
                                  betas=BETAS)

# Define Loss functions
criterion_GAN = nn.MSELoss()  # For GAN Loss
criterion_cycle = nn.L1Loss()  # For Cycle-consistency Loss
lambda_cycle = 10.0  # Weight for cycle loss
G_losses_AtoB = []
G_losses_BtoA = []
D_losses_A = []
D_losses_B = []


# Training Loop
for epoch in range(NUM_EPOCHS):
    for data_A, data_B in zip(dataloader_A, dataloader_B):
        # Unpack data
        real_A = data_A[0].to(device)
        real_B = data_B[0].to(device)

        # Adversarial ground truths
        valid = torch.ones(real_A.size(0), 1).to(device)
        fake = torch.zeros(real_A.size(0), 1).to(device)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_gen.zero_grad()

        # Translate images to opposite domain
        fake_B = gen_AtoB(real_A)
        fake_A = gen_BtoA(real_B)

        # Calculate the loss for generators
        loss_GAN_AtoB = criterion_GAN(disc_B(fake_B), valid)
        loss_GAN_BtoA = criterion_GAN(disc_A(fake_A), valid)

        # Cycle loss
        recovered_A = gen_BtoA(fake_B)
        recovered_B = gen_AtoB(fake_A)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * lambda_cycle
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * lambda_cycle

        # Total generator loss
        total_gen_loss = loss_GAN_AtoB + loss_GAN_BtoA + loss_cycle_ABA + loss_cycle_BAB
        total_gen_loss.backward()
        optimizer_gen.step()

        # -----------------------
        #  Train Discriminators
        # -----------------------
        optimizer_disc.zero_grad()

        # Discriminator A loss
        loss_real_A = criterion_GAN(disc_A(real_A), valid)
        loss_fake_A = criterion_GAN(disc_A(fake_A.detach()), fake)
        total_disc_A_loss = (loss_real_A + loss_fake_A) / 2

        # Discriminator B loss
        loss_real_B = criterion_GAN(disc_B(real_B), valid)
        loss_fake_B = criterion_GAN(disc_B(fake_B.detach()), fake)
        total_disc_B_loss = (loss_real_B + loss_fake_B) / 2

        # Total disc loss
        total_disc_loss = total_disc_A_loss + total_disc_B_loss
        total_disc_loss.backward()
        optimizer_disc.step()

        # Print log info
        if i % 50 == 0:
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}] [D loss: {total_disc_loss.item()}] [G loss: {total_gen_loss.item()}]")


G_losses_AtoB.append(loss_GAN_AtoB.item())
G_losses_BtoA.append(loss_GAN_BtoA.item())
D_losses_A.append(total_disc_A_loss.item())
D_losses_B.append(total_disc_B_loss.item())

# Save models
torch.save(gen_AtoB.state_dict(), os.path.join(MODEL_DIR, 'CycleGAN_gen_AtoB.pth'))
torch.save(gen_BtoA.state_dict(), os.path.join(MODEL_DIR, 'CycleGAN_gen_BtoA.pth'))
torch.save(disc_A.state_dict(), os.path.join(MODEL_DIR, 'CycleGAN_disc_A.pth'))
torch.save(disc_B.state_dict(), os.path.join(MODEL_DIR, 'CycleGAN_disc_B.pth'))

# =================================== Results ===================================
# Plot the training losses
plt.figure(figsize=(10, 5))
plt.title('CycleGAN Training Losses')
plt.plot(G_losses_AtoB, label='Generator A to B')
plt.plot(G_losses_BtoA, label='Generator B to A')
plt.plot(D_losses_A, label='Discriminator A')
plt.plot(D_losses_B, label='Discriminator B')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, 'CycleGAN_loss.png'))
plt.show()

# Visualize the translation results
# Grab a batch of real images from each dataloader
real_batch_A = next(iter(dataloader_A))
real_batch_B = next(iter(dataloader_B))

# Generate translated images
with torch.no_grad():
    fake_B = gen_AtoB(real_batch_A[0].to(device)).detach().cpu()
    fake_A = gen_BtoA(real_batch_B[0].to(device)).detach().cpu()

# Plot the real images from Domain A and their translated versions in Domain B
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images from Domain A")
plt.imshow(np.transpose(vutils.make_grid(real_batch_A[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Translated Images in Domain B")
plt.imshow(np.transpose(vutils.make_grid(fake_B[:64], padding=5, normalize=True), (1, 2, 0)))
plt.savefig(os.path.join(MODEL_DIR, 'CycleGAN_A_to_B.png'))
plt.show()

# Plot the real images from Domain B and their translated versions in Domain A
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images from Domain B")
plt.imshow(np.transpose(vutils.make_grid(real_batch_B[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Translated Images in Domain A")
plt.imshow(np.transpose(vutils.make_grid(fake_A[:64], padding=5, normalize=True), (1, 2, 0)))
plt.savefig(os.path.join(MODEL_DIR, 'CycleGAN_B_to_A.png'))
plt.show()