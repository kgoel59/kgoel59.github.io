# CSCI446/946 Big Data Analytics - Week 9
# Lab 8 – Image Processing with Python


# Task1: Image Data Analysis Using Python

# import packages
import imageio
import matplotlib.pyplot as plt
import numpy as np 
import random

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")





# Task2: CNN for Image Representation

# import packages
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")





pic = imageio.imread("data/logic_on_pic.jpg") 
plt.figure(figsize = (5,5)) 

