# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:37:20 2019

@author: DSML_Admin
"""

!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:30:22 2019

@author: MT
"""

# Assign project template directory
project_dir = "/Users/MT/Desktop/DS_Projects/CSC8635_ML_Project"
# Assign test number
test_n = '1822-05'

# Load standard data processing libraries
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Import metadata df
meta = pd.read_csv(os.path.join(project_dir,"data","HAM10000_metadata.csv"))
# Iterate through data_dir looking for jpg files and append to images_ls
images_ls = []
for dir,_,_ in os.walk(project_dir):
    images_ls.extend(glob(os.path.join(dir,"*.jpg"))) 

# Convert images_ls to dataframe and assign variable name
images_df = pd.DataFrame(images_ls)
images_df.columns = ['path']

# Extract image id from path for join with meta df
images_df['image_id'] = images_df['path'].str[-16:-4]

# Join image_df with meta on image id
meta = pd.merge(meta, images_df, how='left', on=['image_id'])

# Iterate through images, resizing down to to 100x75 pixels, converting to array and inserting into new column
# (uncomment to use)
#meta['image'] = meta['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

# Cache result (uncomment to use)
#meta.to_pickle(os.path.join(project_dir,"cache/meta_cache.p"))

# Import cached metadata df (uncomment to implement)
meta = pd.read_pickle(os.path.join(project_dir,"cache","meta_cache.p"))

# Save frequency of each class (used for ROC analysis)
class_df = pd.DataFrame(meta['dx'].value_counts())
class_df.columns = ['n']
class_df['dx'] = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_df = class_df.sort_values(by=['dx'])

# Extract predictor variable (images) and labels as seperate vectors
X=meta['image']
Y=meta['dx']

# Integer encode each response category and then one-hot encode
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
# List encoded classes (0 through 6) and store
class_list = list(label_encoder.classes_)
# One-hot encode class integers
Y = to_categorical(Y, num_classes = 7)

# Iterate through images vector and normalise image array (divide by max RGB value = 255)
X = np.asarray(meta['image'].tolist())
X = X.astype('float32')
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Split test/train set for predictor and label variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,random_state=10)

# Split training set further for cross validation (NOT used for talos optimisation)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 10) 
















        