"""
Author: Kismat Khatri
Date: Nov 28, 2023
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project


"""


import os
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from train_DCGAN import Generator, Discriminator, weights_init  # Ensure this is correctly imported from your script
from stanford_dogs_data import DogImages  # Ensure this is correctly imported from your script

# Constants (you can make these configurable via the Streamlit interface as well)
WORKERS = 2
NC = 3
NZ = 100
NGF = 64
NDF = 64
ngpu = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

def train_dcgan(epochs, batch_size, lr, beta1, DATA_DIR, MODEL_DIR):
    # Initialize models
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Apply weight initialization or load pretrained model
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Create the dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = DogImages(root=DATA_DIR, train=True, download=False, cropped=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=WORKERS)


    # Training Loop
    for epoch in range(epochs):
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

            # Output training stats to Streamlit (or console)
            if i % 50 == 0:
                st.write(f'[{epoch}/{epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    # Save the model at the end of training
    torch.save(netG.state_dict(), os.path.join(MODEL_DIR, 'DCGAN_netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(MODEL_DIR, 'DCGAN_netD_final.pth'))
    st.success('Training completed and models saved.')




# Streamlit Interface
st.title('DCGAN for Stanford Dogs Dataset')

# Sidebar for parameter settings
st.sidebar.header('Training Parameters')
epochs = st.sidebar.number_input('Epochs', min_value=1, value=30)
batch_size = st.sidebar.number_input('Batch Size', min_value=1, value=64)
lr = st.sidebar.number_input('Learning Rate', min_value=0.00001, max_value=0.1, value=0.0002, format="%.5f")
beta1 = st.sidebar.slider('Beta1 for Adam Optimizer', min_value=0.0, max_value=1.0, value=0.5)

DATA_DIR = st.sidebar.text_input('Data Directory', "/home/ubuntu/Deep-Learning/Data/")
MODEL_DIR = st.sidebar.text_input('Model Directory', "/home/ubuntu/Deep-Learning/Models/")

# Button to start training
if st.sidebar.button('Start Training'):
    with st.spinner('Training in progress...'):
        train_dcgan(epochs, batch_size, lr, beta1, DATA_DIR, MODEL_DIR)
        st.success('Training completed!')

# Add additional Streamlit components as needed to display results, images, etc.
