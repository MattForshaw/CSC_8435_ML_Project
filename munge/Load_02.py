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

from keras.utils.np_utils import to_categorical


# Assign project template data directory
data_dir =  "C:\\Users\DSML_Admin\Desktop\CSC_8435_ML_Project-master"
# Import metadata df
meta = pd.read_csv(os.path.join(data_dir,'data\HAM10000_metadata.csv'))
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

meta['image'] = meta['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
# Cache dataframe (commented to prevent accidental use)
#meta.to_csv(os.path.join(data_dir,"cache\meta_cache.p"))





##### PART 2 (starting from post-munge)

# Assign project template data directory
data_dir =  "C:\\Users\DSML_Admin\Desktop\CSC_8435_ML_Project-master"
# Import cached metadata df
meta = pd.read_pickle(os.path.join(data_dir,"cache\meta_cache.p"))

# Extract predictor variable (images) and labels as seperate vectors
X=meta['image']
Y=meta['dx']

# Iterate through images vector and normalise image array (divide by max RGB value = 255)
X = np.asarray(meta['image'].tolist())
X = X.astype('float32')
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Integer encode each response category and then one-hot encode
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
Y = to_categorical(Y, num_classes = 7)

# Split test/train set for predictor and label variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,random_state=10)

# Split train/validation set for predictor and label variables (not for Talos)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 10) 




























        