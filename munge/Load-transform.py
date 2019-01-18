#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:30:22 2019

@author: MT
"""

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# Assign project template data directory
data_dir = "/Users/MT/Desktop/DS_Projects/CSC8635_ML_Project/data"
# Import metadata df
meta = pd.read_csv(os.path.join(data_dir,'HAM10000_metadata.csv'))
# Convert diagnoses (ground truth) to factor
meta['dx'] = meta['dx'].astype('category')

# Iterate through data_dir looking for jpg files and append to images_ls
images_ls = []
for dir,_,_ in os.walk(data_dir):
    images_ls.extend(glob(os.path.join(dir,"*.jpg"))) 
    
# Convert images_ls to dataframe and assign variable name
images_df = pd.DataFrame(images_ls)
images_df.columns = ['path']
# Extract image id from path for join with meta df
images_df['image_id'] = images_df['path'].str[-16:-4]

# Join image_df with meta on image id
meta = pd.merge(meta, images_df, how='left', on=['image_id'])




meta['image'] = meta['path'].map(lambda x: np.asarray(Image.open(images_df[0][x]).resize((100,75))))
meta['foo'] = meta.apply(lambda x: np.asarray(Image.open(images_df['path'][x]).resize((100,75))), axis=1)

# Create new empty column for image 
meta['image'] = ""
# Iterate through images df, downsampling to 100x75 pixels, converting to array and inserting into meta df
for i in range(0,len(images_df)-1):
    meta['image'][i] = np.asarray(Image.open(images_df['path'][i]).resize((100,75)))
    




        
        
        
        
        
        
        
        
        
        