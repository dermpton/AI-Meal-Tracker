import numpy as np 
import pandas as pd

import os

#------------------

import os 
import torch 
import pandas as pd
import numpy as np 
from sklearn.utils import shuffle 
from PIL import Image
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import requests as reqs
import torch.nn as nn 
from torchvision import models
from tqdm as tqdm 
from torch.utils.checkpoint from checkpoint # For gradient checkpointing
from torch.cuda.amp import autocast, Gradscaler # For mixed-precisions training 
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
from albumentations.pytorch import ToTensorV2
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim 
import zipfile 
import urllin.request
import shutil
import random 
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Setting the device for PyTorch; use CUDA if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#-----------------
#output:
#cuda 
#
# /opt/conda/lib/python3.10/site-packages/albumentations/__init__.py:
# UserWarning: A new version of Albumentations is available: 1.4.21
# (you have 1.4.17). Upgrade using: pip install -U albumentations. To
# disable automatic update checks, set the enviroment variable
# NO_ALBUMENTATIONS_UPDATE to 1. 
# check_for_updates()

#----------------

classes = open("/kaggle/input/food-101/food-101/meta/classes.txt", 'r').read().splitlines()
classes_21 = class[:20] + ['other'] 
classes_21, len(classes_21)

# Defining a custom label encoder for the classes 
class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(label)}
    
    def get_label(self, idx):
        return list(self.label.keys())[idx]

    def get_idx(self, label):
        return self.label.get(label)



# Initializing label encoder with 21 classes and testing its functionality
encoder_21 = Label_encoder(classes_21)
encoder_21.get_label(0), encoder_21.get_idx( encoder_21.get_label(0).get_label(0) )

# Printing each class with it corresponding index 
for i in range(21):
    print(encoder_21.get_label(i), encoder_21.get_idx(encoder_21.get_label(i) ))


# Defining a custom dataset class for handling image data
class Food21(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
# transform is a optional called a keyword argument that can be passed
# or not, to the __init__ constructor hence optional
# the ones we set such as dataframe we can conclude that they are
# positional as the are REQUIRED for the constructor

# Any method that has double underscores are called dunder or magic
# methods like the ones below


    def __len__(self):
        return self.dataframe.shape[0]

# __len__ a dunder method that enables an instance to be counted
# in java terms this would be extending the len() into Len() so that it is # able to work on Objects

# Dataframe? A 2-dimensional data structure, it comes from the pandas
# library, yet to find out more...

# .shape[]? An attribute/property of the dataframe class that returns a
# tuple, i.e., recall that a dataframe is in fact a 2-Dimensional data
# Structure hence, it is only fitting that the shape represents the 
# number of rows and the number of columns in the dataframe 
# .shape in turn returns a tuple in the form 
# (number_of_rows,number_of_columns)
# So shape[0] returns the rows, for what puporse we shall find out

# Reason of the number, to count the number of rows (the samples from the
# dataset (The one currently being constructed)

# This is to inform PyTorch of how many data points there are

    def __getitem__(self, idx):
        img_name = self.dataframe.path.iloc[idx]
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image) # Convert to NumPy array for Albumentations

        label = encoder_21.get_idx(self.dataframe.label.iloc[idx])

        # Apply Albumentations transforms if available
        if self.transform:
            augmented = self.transform(image=image) # Pass image as named
            # argument 
            image = augmented["image"]

            return image, label



# __getitem__
# Enables objects to behave like collections, such as lists or
# dictionaries

# General Purpose in Python:
# - Enabling indexing:
# + The __getitem__ method allows you to use the square bracket notation
# ([]) to access elments within an object.
# + When you write my_object[key], Python internally calls
# my_object.__getitem__(key)
# - Creating Iterable Objects:
# It's fundamental for creating objects that can be iterated over, meaning
# you can use them in for loops.

# Purpose in PyTorch:
# - Datasets:
# + In PyTorch, the Dataset class, which is the base class for many data
# loading utilities, relies heavily on __getitem__.
# + When you create a custom dataset, you must implement __getitem__ to 
# define how individual data samples are accessed.
# + This method is reponsible for:
# + Retrieving a specific data sample and its corresponding label based on 
# the provided index.
# + Applying any necessary transformations to the data.
# - Data Loading:
# + The DataLoader in PyTorch uses __getitem__ to efficiently fetch
# batches of data from your dataset.
# + This allows PyTorch to load data in parallel and in a controlled
# manner, which is essential for training neural networks. 
# - Custom Data Handling:
# + By implementing __getitem__, you can create data custom data
# structures that seamlessly integrate with PyTorch's data loading pipeline


# .path Accesses a column named path in your DataFrame.
# Assumes that the DataFrame has a column called path contains file paths t# to the images. Yet to find out 

# .iloc[idx]? is a Pandas indexer that performs integer-location based
# indexing. Hence IntegerLocation (iloc), gets the element at the position # specified by the index from the path column
# SUMMARY: gets the file path in relation to an index passed.
# Reason why its a bit peculiar considering the path of which I couldn't
# find the path on the data set for food 101 is because the context of the # dataset and dataframe is custom i.e., we are building our own dataset. 


# PIL - Python Imaging Library
# Pillow a fork of PIL, metadata/attributes mode,size,format,pixel data
# .mode - RGB are standard colors, L for Grayscale


# Albumentations (an image augmentation library)
# Work with numpy arrays 
# Allow for efficient numerical operations and easy integration with other
# data processing tools.

# It is a popular image augmentation library used in computer vision.
# It provides a wide range of fast, flexible, and easy-to-use
# transformations (like rotations, flips, color adjustments, and more)
# that help improve the robustness and generalization of machine models

# Role in ML Pipelines:
# Augmentations are applied to training images to artificially expand the
# dataset, introduce variability, and help prevent overfitting by
# exposing the model to different variations of the same image


# TRANSFORMS IN TORCHVISION:
# A set of tools for manipulating and modifying data, particularly images, # before they are fed into a neural network.
# Purpose: 
# - Transforms are used to preprocess data, making it suitable, for 
# training a machine learning model.
# - They can perform tasks like resizing, cropping, normalizing and
# augmenting images.
# - Data augmentation, in particular, is crucial for improving the 
# generalization ability of a model by exposing it to a wider variety of
# data.

# torchvision.transforms:
# - The torchvision.transforms module provides a collection of common
# image transformations.
# - These transformation can be chained together using transforms.Compose
# to create a pipeline of operations.

# Key Concepts:
# - Composition: transforms.Compose allows you to combine multiple
# transforms into a single, executable transformation.
# - Data Augmentation: Transform like RandomCrop, RandomHorizontalFlip,
# and RandomRotation introduce variations into the training data, helping
# the model learn robust features.
# - Normalization: Transform like transforms.Normalize standardize the
# pixel values of images, which can improve training stability.
# Tensor Conversion: transforms.ToTensor converts PIL images or NumPy
# arrays into PyTorch tensors, which are the required input format for
# PyTorch models.


# NEURAL NETWORKS
# A core component of artificial intelligence, particulary in the field of
# machine learning.

# Core Concept:
# - Inspiration from the Brain:
# + Neural Networks are inspired by the structure and function of the
# human brain, specifically how neurons connect and transmit signals.

# - Artificial Neurons:
# + They consist of interconnected "artificial nuerons" (or nodes) that 
# process and transmit information.

# - Layers:
# These neurons are typically organized into layers: 
# + Input Layer: Receives the initial data.
# + Hidden Layers: Process the data through multiple levels of abstraction.
# + Output Layer: Produces the final result.

# - Connections and Weights:
# + The connections between neurons have "weights" that determine the
# strength of the singal being passed. 
# + During training, these weights are adjusted to improve the network's 
# accuracy. (?)

# How They Work:
# - Information Flow:
# + Data flows through the network, with each nueron performing a
# calculation and passing the result to the next layer.

# - Activation Functions:
# + Neurons use "activation functions" to determine whether to activate
# and pass on a signal.

# - Learning:
# + Neural networks learn by adjusting the weights of their connections
# based on the difference between their predictions and the actual results
# This is often called "backpropagation"


# DATASETS
# Dataset class is a fundamental abstraction for representing and working
# with datasets. It's a key component of PyTorch's data loading pipeline,
# designed to make it easy to manage and access your data for machine
# learning tasks.

# Purpose:
# - Abstraction of Data:
# + The Dataset class provides a standardized way to represent any dataset
# - Data Access:
# + It defines how individual data samples are accessed, enabling
# efficient retrieval of data during training or evaluation.
# - Integration with DataLoader:
# + It's designed to work seamlessly with the DataLoader class, which
# handles batching, shuffling, and parallel loading of data.

# Key Characteristics:
# - Abstract Class:
# + torch.utils.data.Dataset is an abstract class, meaning you typically
# create custom subclasses to define your specific dataset.
# - Required Methods:
# + Custom Dataset subclasses must implement two essential methods:
# + __len__(self): Returns the total number of samples in the dataset
# + __getitem__(self,idx): Retrieves the data sample and its
# corresponding label(or target) at the given index idx.
# - Customization:
# + You can customize the Dataset class to handle various formats, perform
# data preprocessing, and apply transformations.


# DATALOADERS
# Simplifies the process of loading and managing datasets during the
# training and evaluation of neural networks.

# Purpose:
# - Efficient Data Loading:
# + DataLoader automates the process of fetching data from a Dataset object
# making it easier to iterate through your data.
# - Batching:
# + It groups data samples into batches, which are then fed into the model
# This is essential for efficient training, especially on GPUs.
# - Shuffling:
# + It can randomly shuffle the data, which is crucial for preventing the
# model from learning the order of the data and improving its
# its generalization.
# - Parallel Loading:
# + It can load data in parallel using multiple worker processes, 
# significantly speeding up data loading, particularly for large datasets.

# Key Functionalities:
# - Batching:
# + DataLoader organizes your data into mini-batches, allowing your model
# to process multiple samples at once. This improves computational 
# efficiency.
# - Shuffling:
# + By shuffling the data, DataLoader ensures that the model sees data 
# in a random order during training. This helps to prevent bias and 
# improve generalization.
# - Parallel Data Loading:
# + The num_workers parameters allows you to specify the number of
# subprocesses to use for data loading. This enables parallel loading,
# which can significantly reduce data loading times.

# RESUMING THE CODE:
# expect the classes to be enumerated i.e., apple_pie 0 ...other 20

def prep_df(path: str) -> pd.DataFrame:
    array = open(path,'r').read().splitlines()
    # Getting the full path for the images
    img_path = '/kaggle/input/food-101/food-101/food-101/images/'
    full_path = [img_path + img + ".jpg" for img in array]
    # Splitting the image index from the label
    imgs = []
    for img in array:
        img = img.split('/')
        imgs.append(img)
    imgs = np.array(img)
    for idx, img in enumerate(imgs):
        if encoder_21.get_idx(img[0]) is None:
            imgs[idx,0] = "other"
    # Converting the array to a data frame
    imgs = pd.DataFrame(imgs[:,0], imgs[:,0], columns=['label'])
