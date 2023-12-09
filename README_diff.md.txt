# My Project

This README file is a reference for the diffusion trainer code, UNet2Ddiffuser_train

## Installation

Install the dependencies, and log into huggingface with "huggingface-cli login" with your access token (read or write for uploading to hf)

#sudo apt -qq install git-lfs
#pip install diffusers[training]

## Usage

In the code, replace the hub_model_id and the output_dir in the class based off of what repository you want to save your model to.
Also, change the hub_private_repo depending on what settings your hf is, and set push_to_hub to false if you don't want to push your model.

## Contact

Jeffrey Hu - jeffrey.hu1@gwu.edu