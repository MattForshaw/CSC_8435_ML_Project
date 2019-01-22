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




import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Set random seed to ensure reproducability
np.random.seed(10)      

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = X_train.shape[1:]
act = 'relu'

model = Sequential()
#model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation=act,padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation=act,padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

#model.add(BatchNormalization(input_shape=X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation=act,padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation=act,padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation=act))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()

# Define the optimizer
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Data augmentation to prevent overfitting 
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


# Fit the model
epochs = 150 
batch_size = 8
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
# Save model
model.save(os.path.join(data_dir,"cnn_78TEST_22JAN.hdf5"))





#Load saved model
from keras.models import load_model
model = load_model(os.path.join(data_dir,"cnn_78TEST_22JAN.hdf5"))


loss, acc = model.evaluate(X_test, Y_test, verbose=1)
loss_val, acc_val = model.evaluate(X_val, Y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))
print("Test: accuracy = %f  ;  loss = %f" % (acc, loss))



# Save model
model.save(os.path.join(data_dir,"cnn_1822-05_78.1TEST_22JAN.hdf5"))





# Convert model training history dictionary to dataframe for plotting
hist_df = pd.DataFrame(history.history['acc'])
hist_df.columns = ['acc']
hist_df['loss'] = pd.DataFrame(history.history['loss'])
hist_df['val_acc'] = pd.DataFrame(history.history['val_acc'])
hist_df['val_loss'] = pd.DataFrame(history.history['val_loss'])
hist_df['lr'] = pd.DataFrame(history.history['lr'])
hist_df['epoch'] = range(150)

# Cache history df
hist_df.to_pickle(os.path.join(data_dir,"cache\hist_df_1822-05.p"))


# Make sliced copies of hist_df for plotting
hist_df_acc = hist_df.drop({'loss','val_loss','lr'}, axis=1)
hist_df_loss = hist_df.drop({'acc','val_acc','lr'}, axis=1)

from ggplot import *

# Plot Training Accuracy
ggplot(pd.melt(hist_df_acc, id_vars=['epoch']), aes(x='epoch', y='value', color='variable')) +\
    geom_line() +\
    xlab("Epoch") + ylab("Accuracy (%)") + ggtitle("Training Accuracy")

# Plot Training Loss
ggplot(pd.melt(hist_df_loss, id_vars=['epoch']), aes(x='epoch', y='value', color='variable')) +\
    geom_line() +\
    xlab("Epoch") + ylab("Loss (%)") + ggtitle("Training Loss")


# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7))


label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(7), label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction classified incorrectly')






