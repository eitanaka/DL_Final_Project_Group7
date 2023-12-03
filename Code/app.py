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
from torchvision.utils import make_grid
from PIL import Image
<<<<<<< HEAD
from torchvision.transforms import ToPILImage
import numpy as np
=======
>>>>>>> 5b48a10f22b65abc7d57674e639e4fb9b5340835
import torchvision.transforms as transforms
from train_DCGAN import Generator, Discriminator, weights_init  # Ensure this is correctly imported from your script
from stanford_dogs_data import DogImages  # Ensure this is correctly imported from your script# Ensure this is correctly imported from your script

# Constants (you can make these configurable via the Streamlit interface as well)
WORKERS = 2
NC = 3
NZ = 100
NGF = 64
NDF = 64
ngpu = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Function to load a model
def load_model(model_path):
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()
    return netG

# Function to generate images
def generate_images(netG, NZ, device):
    # Generate a batch of latent vectors (random noise)
    latent_vector = torch.randn(1, NZ, 1, 1, device=device)
    # Generate fake image batch with G
    fake_image = netG(latent_vector)
    return fake_image

# Streamlit Interface
st.title('DCGAN for Stanford Dogs Dataset')

# Sidebar for model directory
MODEL_DIR = st.sidebar.text_input('Model Directory', "/home/ubuntu/Deep-Learning/Models/")

# Dropdown menu for pre-selected images (optional, for display purposes)
# Dropdown menu for pre-selected images
st.sidebar.header('Select an Input Image')
image_options = {
    "Chihuahua": "/home/ubuntu/Deep-Learning/Data/StanfordDogs/Images/n02085620-Chihuahua/n02085620_10074.jpg",
    "Japanese Spaniel": "/home/ubuntu/Deep-Learning/Data/StanfordDogs/Images/n02085782-Japanese_spaniel/n02085782_1058.jpg",
    "Maltese Dog": "/home/ubuntu/Deep-Learning/Data/StanfordDogs/Images/n02085936-Maltese_dog/n02085936_10297.jpg",
    "African Hunting Dog": "/home/ubuntu/Deep-Learning/Data/StanfordDogs/Images/n02116738-African_hunting_dog/n02116738_10476.jpg"
}
selected_image_name = st.sidebar.selectbox('Select an Input Image', list(image_options.keys()))
selected_image_path = image_options[selected_image_name]

<<<<<<< HEAD
# Display the selected image (optional)
input_image = Image.open(selected_image_path)
st.image(input_image, caption='Selected Input Image', use_column_width=True)

# Load model
model_files = os.listdir(MODEL_DIR)
selected_model = st.sidebar.selectbox('Select a Model', model_files)
model_path = os.path.join(MODEL_DIR, selected_model)
netG = load_model(model_path)

# Button to generate images
if st.sidebar.button('Generate Image'):
    with st.spinner('Generating Image...'):
        output_tensor = generate_images(netG, NZ, device).squeeze(0).detach().cpu()
        
        # Normalize to [0, 1], then convert to [0, 255] and change to uint8
        output_tensor = (output_tensor + 1) / 2  # Normalize to [0, 1]
        output_tensor = output_tensor.clamp(0, 1)
        output_tensor = output_tensor.numpy()
        output_tensor = np.transpose(output_tensor, (1, 2, 0))  # Convert from CHW to HWC format
        output_tensor = (output_tensor * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

        output_image = ToPILImage()(output_tensor)

        st.image(output_image, caption='Generated Image', use_column_width=True)

        
=======
>>>>>>> 5b48a10f22b65abc7d57674e639e4fb9b5340835

# Function to load and display generated images
def display_generated_images(model_dir, ngpu, nz, device):
    # Initialize and load the generator model
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(os.path.join(model_dir, 'DCGAN_netG_final.pth')))
    netG.eval()  # Set the model to evaluation mode

    # Generate a batch of images
<<<<<<< HEAD
    fixed_noise = torch.randn(64, NZ , 1, 1, device=device)  # 64 images
=======
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # 64 images
>>>>>>> 5b48a10f22b65abc7d57674e639e4fb9b5340835
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()

    # Convert tensor to grid of images and display
    grid = make_grid(fake_images, padding=2, normalize=True)
    st.image(grid.permute(1, 2, 0).numpy(), caption='Generated Images')


# Display generated images section
st.header("Generated Images")
if st.button('Show Generated Images'):
    display_generated_images(MODEL_DIR, ngpu, NZ, device)


# Display Training Loss
st.header("Training Loss")
loss_image_path = os.path.join(MODEL_DIR, 'DCGAN_loss.png')
if os.path.exists(loss_image_path):
    loss_image = Image.open(loss_image_path)
    st.image(loss_image, caption='Training Loss')

# Display Fake Images
st.header("Generated Fake Images")
fake_images_path = os.path.join(MODEL_DIR, 'DCGAN_fake_images.png')
if os.path.exists(fake_images_path):
    fake_images = Image.open(fake_images_path)
    st.image(fake_images, caption='Fake Images')

# Display Real vs Fake Images Comparison
st.header("Real vs Fake Images")
real_vs_fake_path = os.path.join(MODEL_DIR, 'DCGAN_real_vs_fake.png')
if os.path.exists(real_vs_fake_path):
    real_vs_fake_images = Image.open(real_vs_fake_path)
<<<<<<< HEAD
    st.image(real_vs_fake_images, caption='Real vs Fake Images')
=======
    st.image(real_vs_fake_images, caption='Real vs Fake Images')
    
# Add additional Streamlit components as needed to display results, images, etc.
>>>>>>> 5b48a10f22b65abc7d57674e639e4fb9b5340835
