"""
Author: Ei Tanaka,Cody Yu
Date: Nov 10, 2023
Purpose: To load the Stanford Dogs dataset and EDA analysis
Reference: https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py
"""

# =================================== Imports ===================================
from PIL import Image
from os.path import join
import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets.utils import download_url, list_dir, list_files
import random
import torch

# =================================== Constants ===================================
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + "Data" + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)
# =================================== Loading Data (PyTorch) ===================================
class DogImages(data.Dataset):
    """ Stanford Dogs Dataset <http://vision.stanford.edu/aditya86/ImageNetDogs/>.
    Args:
        root (string): Root directory of dataset where directory `_train.py` exists.
        cropped (bool, optional): If True, returns cropped images from bounding boxes.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the target
            and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, train=True, cropped=False, transform=None, target_transform=None, download=True):
        self.root = os.path.join(os.path.expanduser(root), 'StanfordDogs')
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.lists_folder = os.path.join(self.root, 'Lists')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                       for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation + '.jpg', idx) for annotation, box, idx in
                                       self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[idx]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[idx][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + os.path.sep + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree as ET
        e = ET.parse(path).getroot()
        boxes = []
        for obj in e.iter('object'):
            boxes.append([int(obj.find('bndbox').find('xmin').text),
                          int(obj.find('bndbox').find('ymin').text),
                          int(obj.find('bndbox').find('xmax').text),
                          int(obj.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [s[0][0] for s in split]
        labels = [l[0]-1 for l in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for idx in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[idx]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images)) / float(len(counts.keys()))))

        return counts

    def show_samples(self, rows=3, cols=8):
        num_images = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i in range(num_images):
            ax = axes[i // cols, i % cols]
            img, label = self[np.random.randint(len(self))]
            img = img.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
            ax.imshow(np.asarray(img))
            ax.set_title(self.classes[label], fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_breed_distribution(self):
        breed_counts = self.stats()
        breeds = [self.classes[breed] for breed in breed_counts.keys()]
        counts = list(breed_counts.values())

        plt.figure(figsize=(20, 10))
        plt.bar(breeds, counts, color='skyblue')
        plt.xlabel('Breeds')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=90)
        plt.title('Distribution of Dog Breeds in Dataset')
        plt.show()

    def plot_color_distribution(self, sample_size=100):
        # Initialize lists to store RGB values
        r_pixels, g_pixels, b_pixels = [], [], []

        for _ in range(sample_size):
            img, _ = self[np.random.randint(len(self))]
            # Convert image to numpy array and normalize pixel values to [0, 1]
            img_np = np.asarray(img.permute(1, 2, 0)) / 255.0
            r_pixels.extend(img_np[:, :, 0].flatten())
            g_pixels.extend(img_np[:, :, 1].flatten())
            b_pixels.extend(img_np[:, :, 2].flatten())

        # Plotting the color distributions
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(r_pixels, bins=50, color='red', alpha=0.7)
        plt.title('Red Channel Distribution')
        plt.subplot(1, 3, 2)
        plt.hist(g_pixels, bins=50, color='green', alpha=0.7)
        plt.title('Green Channel Distribution')
        plt.subplot(1, 3, 3)
        plt.hist(b_pixels, bins=50, color='blue', alpha=0.7)
        plt.title('Blue Channel Distribution')
        plt.tight_layout()
        plt.show()

    def show_augmented_samples(self, num_samples=3, seed=0):
        torch.manual_seed(seed)  # For reproducibility

        # Define the augmentation transformations
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ])

        # Choose a few images at random
        indices = torch.randperm(len(self))[:num_samples]
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            img, label = self[idx]
            img = transforms.ToPILImage()(img)
            augmented_img = augmentation_transforms(img)

            # Display original images
            axes[i].imshow(np.asarray(img))
            axes[i].set_title('Original - ' + self.classes[label])
            axes[i].axis('off')

            # Display augmented images
            axes[i + num_samples].imshow(np.asarray(augmented_img))
            axes[i + num_samples].set_title('Augmented - ' + self.classes[label])
            axes[i + num_samples].axis('off')

        plt.tight_layout()
        plt.show()

def main():
    dataset = DogImages(root=DATA_DIR, train=True, download=False, cropped=True, transform=transforms.ToTensor())
    dataset.show_samples()
    dataset.plot_breed_distribution()
    dataset.plot_color_distribution()
    dataset.show_augmented_samples()
    image, label = dataset[0]
    print(f'Image shape: {image.shape}, Label: {label}')


if __name__ == '__main__':
    main()