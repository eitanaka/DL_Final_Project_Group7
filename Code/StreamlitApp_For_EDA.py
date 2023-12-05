"""
Author: Kismat Khatri
Date: Nov 30, 2023
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project


"""

import streamlit as st
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from stanford_dogs_data import DogImages

# Constants
DATA_DIR = "/home/ubuntu/Deep-Learning/Data/"
CROPPED = True # or False, based on your preference
TRAIN = True # or False, if you want to use the test dataset

# Initialize the dataset
@st.cache_data  # Updated cache decorator
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

# Streamlit Interface
st.title('Dog Images Dataset Analysis')

# Display random samples
if st.button('Show Random Images'):
    fig = dataset.show_samples(rows=3, cols=8)

    st.pyplot(fig)


# Display breed distribution

if st.button('Show Breed Distribution'):
    fig = dataset. plot_breed_distribution()

    st.pyplot(fig)


# Show color distribution
if st.button('Show Color Distribution'):
    fig = dataset.plot_color_distribution()
    st.pyplot(fig)

# Show augmented samples
if st.button('Show Augmented Samples'):
    fig = dataset.show_augmented_samples()
    st.pyplot(fig)

# Main
if __name__ == '__main__':
    st.sidebar.title("Settings")
    # Any additional settings or features you want to add.
