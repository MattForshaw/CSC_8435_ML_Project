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

##### grid and random search using Talos (ensure load_transform.py and CNN.py have been loaded first for new analyses)

# Limit parameters to only a few with minimal range over minimal epochs. Otherwise it will take forever

# import talos
import talos as ta

# Define model inside function to be called by ta.Scan
def rand_search(X_train, Y_train, X_val, Y_val, params):
    conv_dropout = float(params['conv_dropout'])
    dense1_neuron = int(params['dense1_neuron'])

    # Construct model
    model = Sequential()
#    model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), padding='same', activation=params['activation']))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), padding='same', activation=params['activation']))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(conv_dropout))

    # Extra layers used during random search
#    model.add(BatchNormalization(input_shape=X_train.shape[1:]))
#    model.add(Conv2D(256, (5, 5), padding='same', activation=params['activation']))
#    model.add(MaxPool2D(pool_size = (2, 2)))
#    model.add(Dropout(conv_dropout))

    model.add(Flatten())
    model.add(Dense(dense1_neuron, activation=params['activation']))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set training parameters and fit model
    out = model.fit(
        x, y, epochs=50, 
        batch_size=10, 
        verbose=1,
        validation_data=[x_val, y_val],
        callbacks=[learning_rate_reduction]
    )
    return out, model

# Set ranges for hyper-parameter optimisation
para = {
    'dense1_neuron': [128, 256],
    'activation': ['relu', 'elu'],
    'conv_dropout': [0.2, 0.4, 0.5]
}

# Start scan using input data (validation sets created automatically)
scan_results = ta.Scan(X_train, Y_train, para, rand_search)







