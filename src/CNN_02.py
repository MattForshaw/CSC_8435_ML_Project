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

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, softmax

# import talos
import talos as ta


# Set random seed to ensure reproducability
np.random.seed(10)      




p = {'lr': (2, 10, 30),
     'first_neuron': [4, 8, 16, 32, 64, 128],
     'hidden_layers': [2, 3, 4, 5, 6],
     'batch_size': [2, 3, 4],
     'epochs': [300],
     'dropout': (0, 0.40, 10),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': ['adam', 'nadam'],
     'losses': [categorical_crossentropy, logcosh],
     'activation': [relu, elu],
     'last_activation': [softmax]}


def fashion_mnist_fn(x_train, y_train, x_val, y_val, params):
    conv_dropout = float(params['conv_dropout'])
    dense1_neuron = int(params['dense1_neuron'])
    model = Sequential()
    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(128, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(256, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(Flatten())
    model.add(Dense(dense1_neuron, activation=params['activation']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
  
    
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    out = model.fit(
        x, y, epochs=10, batch_size=32, 
        verbose=0,
        validation_data=[x_val, y_val]
    )
    return out, model




para = {
    'dense1_neuron': [256, 512],
    'activation': ['relu', 'elu'],
    'conv_dropout': [0.25, 0.4]
}
scan_results = ta.Scan(x_train, y_train, para, fashion_mnist_fn)




























# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

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


# Fit the model
epochs = 50 
batch_size = 10
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_validate,Y_validate),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])



# Save model
model.save(os.path.join(data_dir,"cnn.hdf5"))

loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_validate, Y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("model.h5")




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
Y_pred = model.predict(X_validate)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_validate,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7))


label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(7), label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction classified incorrectly')


















plot_confusion_matrix(confusion_mtx, classes = range(7)) 