#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:30:22 2019

@author: MT
"""

%matplotlib inline
import matplotlib
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imread import imread, imsave
from keras.utils.np_utils import to_categorical


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

## Create new empty column for image 
#meta['image'] = ""
# Iterate through images df, downsampling to 100x75 pixels, converting to array and inserting into meta df
#for i in range(0,len(images_df)-1):
##    meta.image[i] = np.asarray(Image.open(meta.path[i]).resize((100,75)))
#     meta.iloc[i,9] = np.asarray(Image.open(meta.iloc[i,7]).resize((100,75)))

meta['image'] = meta['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
meta.to_csv(os.path.join(data_dir,"meta_asarray.csv"), sep='\t')

# Extract predictor variable (images) and labels as seperate vectors
X=meta['image']
Y=meta['dx']

# Iterate through images vector and normalise image array (divide by max RGB value = 255)
X = np.asarray(meta['image'].tolist())
X = X.astype('float32')
X /= 255

# Split test/train set for predictor and label variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=10)

# Perform one-hot encoding on the labels

# Integer encode each category
label_encoder = LabelEncoder()
Y_train_enc = label_encoder.fit_transform(Y_train)
Y_test_enc = label_encoder.fit_transform(Y_test)
# One-hot encode integers
Y_train = to_categorical(Y_train_enc, num_classes = 7)
Y_test = to_categorical(Y_test_enc, num_classes = 7)



Y_train.shape

X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 10) 

# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = X_train.reshape(X_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
X_validate = X_validate.reshape(X_validate.shape[0], *(75, 100, 3))


X_train.shape
x_train.shape


X_validate.shape








# Verify array/image pipleline integrity by converting back to image
plt.imshow(y)

# invert encoding
inverted = argmax(encoded[0])
print(inverted)




































        