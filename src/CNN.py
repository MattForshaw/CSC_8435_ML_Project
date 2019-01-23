# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:38:31 2019

@author: DSML_Admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 23:14:53 2019

@author: MT
"""

# Import keras utilities
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = X_train.shape[1:]
act = 'relu'
n_classes = 7

# Construct model
model = Sequential()
#model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation=act,padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation=act,padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), activation=act,padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation=act,padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation=act))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()

# Define the optimizer
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# or 5 e-4

# Start with no LR decay. Then optimise in subsequent models
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# With data augmentation to prevent overfitting 
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# Set random seed to ensure reproducability
np.random.seed(10)      

# Set training parameters and fit model
epochs = 150 
batch_size = 8
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])




# Save model (uncomment to use)
#model.save(os.path.join(project_dir,"tests","cnn"+test_n+".hdf5"))

# Convert model training history dictionary to dataframe for plotting
hist_df = pd.DataFrame(history.history['acc'])
hist_df.columns = ['train_acc']
hist_df['train_loss'] = pd.DataFrame(history.history['loss'])
hist_df['val_acc'] = pd.DataFrame(history.history['val_acc'])
hist_df['val_loss'] = pd.DataFrame(history.history['val_loss'])
hist_df['lr'] = pd.DataFrame(history.history['lr'])
hist_df['epoch'] = epochs

# Cache history df (uncomment to use)
#hist_df.to_pickle(os.path.join(project_dir,"cache","hist_df_"+test_n+".p"))




