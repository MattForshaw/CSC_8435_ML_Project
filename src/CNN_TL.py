# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:22:06 2019

@author: b8059092
"""
##### Transfer Learn form ImageNet (ensure load_transform.py and CNN.py have been loaded first for new analyses)

# Results were quite bad - around 63% with almost immediate plateau

from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input

base_model = InceptionV3(weights='imagenet', include_top=False)

CLASSES = 7
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 50 
batch_size = 15
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])