# =================================== Imports / Setup ===================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# =================================== Constants ===================================
IMAGE_SIZE = 224

# =================================== Loading Data (PyTorch) ===================================
class DogImagesTorch():
    def __inti__(self):

        url = {
            'images': 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
            'annotations': 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar',
            'lists': 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar',
        }


# =================================== Loading Data (Tensorflow) ===================================
# https://www.tensorflow.org/datasets/catalog/stanford_dogs
# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/splits
# https://www.tensorflow.org/datasets/keras_example
# https://www.tensorflow.org/datasets/keras_example#using_the_dataset_with_keras

def preprocess_image(image, label):
    """
    Preprocesses the image and label
    :param image: float32 tensor of shape [height, width, 3] with values in [0, 1]
    :param label: int32 tensor of shape [] with values in [0, NUM_CLASSES)
    :return: float32 tensor of shape [height, width, 3] with values in [-1, 1]
    """
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255.0

    # data augmentation
    # To Do

    return image, label

def load_data_TensorFlow():
    """
    Loads the data using TensorFlow
    :return: train_dataset, test_dataset
    """
    dataset, info = tfds.load('stanford_dogs', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = train_dataset.map(preprocess_image).batch(32)
    test_dataset = test_dataset.map(preprocess_image).batch(32)

    return train_dataset, test_dataset

# =================================== EDA ===================================

def main():
    pass

if __name__ == '__main__':
    main()