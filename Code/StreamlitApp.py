"""
Author: Kismat Khatri
Date: Dec 4, 2023
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project


"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import streamlit as st
from train_DCGAN import Generator  # Ensure this is correctly imported from your script
from stanford_dogs_data import DogImages  # Ensure this is correctly imported from your script

# Constants for DCGAN
WORKERS = 2
NC = 3
NZ = 100
NGF = 64
NDF = 64
ngpu = 1

# Set device for DCGAN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for Dataset Analysis
DATA_DIR = "/home/ubuntu/Deep-Learning/Data/"
CROPPED = True # or False, based on your preference
TRAIN = True # or False, if you want to use the test dataset

# Initialize the Stanford Dogs dataset
@st.cache_data 
def load_data():
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()
    ])
    dataset = DogImages(root=DATA_DIR, train=TRAIN, cropped=CROPPED, transform=transform)
    return dataset

# Load the dataset
dataset = load_data()

# Function to load a DCGAN model
def load_model(model_path):
    try:
        file_extension = os.path.splitext(model_path)[1]
        
        if file_extension in ['.pth', '.pt']:
            # Loading for DCGAN model
            netG = Generator(ngpu).to(device)
            netG.load_state_dict(torch.load(model_path, map_location=device))
            netG.eval()
            return netG
        elif file_extension == '.safetensors':
            # Loading for UNet2 Diffuser model
            unet2_diffuser = load_unet2_diffuser_model(model_path)
            unet2_diffuser.eval()
            return unet2_diffuser
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_unet2_diffuser_model(model_path):
    # Replace this with the actual loading procedure for your UNet2 Diffuser model
    # The procedure will depend on how the .safetensors file was saved
    # Example:
    # unet2_diffuser = CustomUNet2DiffuserClass()
    # unet2_diffuser.load_state_dict(custom_load_function(model_path))
    # return unet2_diffuser
    pass


# Function to generate images using DCGAN
def generate_images(netG, NZ, device):
    latent_vector = torch.randn(1, NZ, 1, 1, device=device)
    fake_image = netG(latent_vector)
    return fake_image

# Streamlit Interface
st.title('DCGAN, Unet2 and Dataset Analysis for Stanford Dogs')

# Sidebar for settings
st.sidebar.title("Settings")

# Sidebar for model directory
MODEL_DIR = st.sidebar.text_input('Model Directory', "/home/ubuntu/Deep-Learning/Models/")

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

# Display the selected image
input_image = Image.open(selected_image_path)
st.image(input_image, caption='Selected Input Image', use_column_width=True)

# Load model for DCGAN
model_files = os.listdir(MODEL_DIR)
selected_model = st.sidebar.selectbox('Select a DCGAN Model', model_files)
model_path = os.path.join(MODEL_DIR, selected_model)
netG = load_model(model_path)

# Button to generate images using DCGAN
if st.sidebar.button('Generate Image'):
    with st.spinner('Generating Image...'):
        output_tensor = generate_images(netG, NZ, device).squeeze(0).detach().cpu()
        output_image = transforms.ToPILImage()(output_tensor)
        st.image(output_image, caption='Generated Image', use_column_width=True)

# Display random samples from the dataset
if st.button('Show Random Images from Dataset'):
    fig = dataset.show_samples(rows=3, cols=8)
    st.pyplot(fig)

# Show breed distribution
if st.button('Show Breed Distribution'):
    fig = dataset.plot_breed_distribution()
    st.pyplot(fig)

# Show color distribution
if st.button('Show Color Distribution'):
    fig = dataset.plot_color_distribution()
    st.pyplot(fig)

# Show augmented samples
if st.button('Show Augmented Samples'):
    fig = dataset.show_augmented_samples()
    st.pyplot(fig)

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
    st.image(real_vs_fake_images, caption='Real vs Fake Images')

# Main
if __name__ == '__main__':
    pass
