# Generating Photorealistic Images of Dogs using Generative Adversarial Networks (GANs)
## DATS 6303 Final Project - Group 7
### George Washington University, Columbian College of Arts & Sciences

## Overview
This project aims to generate realistic dog images using GANs, with a focus on DCGANs and Unit2D diffusers.

## Project Structure
- `Code`: Contains all code files including data preprocessing and model training scripts. (Details in the Code folder)
- `Group-Project-Proposal`: Initial project proposal outlining the objectives and methodology.
- `Final-Group-Project-Presentation`: Presentation slides detailing the project findings and results.
- `Final-Group-Project-Report`: Comprehensive report including detailed analysis, results, and discussions.
- `Individual-Project-Reports`: Contributions and reports from each team member.

## Installation and Usage

1. **Clone the Repository**:
   To start working with the project, clone it from GitHub using the following command:
   ```bash
   git clone https://github.com/eitanaka/DL_Final_Project_Group7.git

2. **Run on GPU**:
Ensure your system has a GPU for efficient model training.

3. **Data Loading**:
Use the `StanfordDogImage` class for loading the Stanford Dogs dataset.

4. **Train DCGAN**:
- Modify the constants and `main` function in the `train_DCGAN` script for hyperparameter tuning.
- Update the dataset path as per your local setup.

5. Install the dependencies, and log into huggingface with "huggingface-cli login" with your access token (read or write for uploading to hf)

#sudo apt -qq install git-lfs
#pip install diffusers[training]

6. In the code, replace the hub_model_id and the output_dir in the class based off of what repository you want to save your model to.
Also, change the hub_private_repo depending on what settings your hf is, and set push_to_hub to false if you don't want to push your model.


## Contributors
- Cody Yu
- Kismat Khatri
- Jeffrey Hu
- Ei Tanaka

## License
(Information about the license, if applicable)

## Acknowledgments
(Acknowledgments to people, resources, or institutions)

For more details, please refer to individual files and folders.
