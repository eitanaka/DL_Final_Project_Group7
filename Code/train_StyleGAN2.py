""" Training script for StyleGAN2-ADA.
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
"""
"""
Reference:
https://www.youtube.com/watch?v=-314Kbgtq9w
"""

"""
Author: Ei Tanaka
Date: Dec 4, 2023
Class: DATS 6303 Deep Learning
Instructor: Dr. Amir Jafari
Assignment: Final Project
"""

# =================================== Import ===================================
import os
import sys

import numpy as np
import torch

# =================================== Constants ===================================


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')
# =================================== Functions ===================================

# =================================== Main ===================================
def main():
    pass

if __name__ == '__main__':
    main()