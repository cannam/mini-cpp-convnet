#!/usr/bin/env python3

# Train a simple convolutional neural network using various hard-coded
# parameters, and dump the weights to a C++ source file.
#
# The neural network in question is inspired by the one in "Run your Keras
# models in C++ Tensorflow"
# (http://www.bitbionic.com/2017/08/18/run-your-keras-models-in-c-tensorflow/)
# with a few changes for illustrative reasons and to reduce the size of the
# trained model in version control.
#
# I used the same flowers dataset as referred to in that article
# (http://download.tensorflow.org/example_images/flower_photos.tgz).  I divided
# it into training, validation, and test sets, scaled images to a maximum of
# 128x128 using ImageMagick convert, and put the training set in
# ../data/train/{daisy,dandelion,roses,sunflowers,tulips}. That location is
# hardcoded in this script.

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import ZeroPadding2D

from keras.preprocessing.image import ImageDataGenerator

image_size = 128
num_categories = 5
batch_size = 40
num_epochs = 70

model = Sequential()

model.add(ZeroPadding2D(padding = (1,1),
                        input_shape = (image_size, image_size, 3),
                        data_format = 'channels_last'))

model.add(Conv2D(filters = 32,
                 kernel_size = (3,3),
                 padding = 'valid',
                 name = 'firstConv'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(ZeroPadding2D(padding = (1,1)))

model.add(Conv2D(filters = 16,
                 kernel_size = (3,3),
                 padding = 'valid',
                 name = 'secondConv'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(ZeroPadding2D(padding = (1,1)))

model.add(Conv2D(filters = 16,
                 kernel_size = (3,3),
                 padding = 'valid',
                 name = 'thirdConv'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(ZeroPadding2D(padding = (1,1)))

model.add(Conv2D(filters = 8,
                 kernel_size = (3,3),
                 padding = 'valid',
                 name = 'fourthConv'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(units = 256, name = 'firstDense'))

model.add(Activation('relu'))

model.add(Dense(units = num_categories, name = 'labeller'))

model.add(Activation('softmax'))

model.compile(optimizer = 'nadam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

with open('my-architecture.json', 'w') as fout:
    fout.write(model.to_json())
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2, 
                                   rotation_range = 25,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(
    "../data/train",
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_set = validation_datagen.flow_from_directory(
    "../data/validate",
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'categorical')

model.fit_generator(training_set,
                    steps_per_epoch = len(training_set.filenames)//batch_size,
                    validation_data = validation_set,
                    validation_steps = len(validation_set.filenames)//batch_size,
                    epochs = num_epochs,
                    workers = 32, 
                    max_queue_size = 32)
    
def write_weights(fout, ww):
    fout.write('{\n')
    leaf = (len(ww.shape) == 1)
    for i in range(0, ww.shape[0]):
        if leaf:
            fout.write(str(ww[i]))
        else:
            write_weights(fout, ww[i])
        if i+1 < ww.shape[0]:
            fout.write(', ')
    fout.write('\n}')

with open('my-weights.cpp', 'w') as fout:
    fout.write('#include "weights.hpp"\n')
    fout.write('#include <vector>\n')
    fout.write('using std::vector;\n\n')
    
    for layer in model.layers:
        weights = layer.get_weights()
        if weights == []:
            continue
        for ix in [0,1]:
            data = weights[ix]
            for i in data.shape:
                fout.write('vector<')
            fout.write('float')
            for i in data.shape:
                fout.write('>')
            fout.write(' ' + (['weights','biases'][ix]) +
                       '_' + layer.name + '\n')
            write_weights (fout, data)
            fout.write(';\n\n')
            
model.save_weights('my-weights.h5')

                    
