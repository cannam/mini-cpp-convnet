#!/usr/bin/env python3

import numpy as np

import sys

from keras.preprocessing import image
from keras.models import model_from_json

with open('architecture.json', 'r') as f:
    arch_json = f.read()

model = model_from_json(arch_json)

model.load_weights('weights.h5')

image_filename = sys.argv[1]
image_size = 128

img = image.load_img(image_filename, target_size = (image_size, image_size))
img = np.expand_dims(img, axis = 0)
img = img / 255.0

result = model.predict(img)[0]

labels = [ 'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips' ]

descending = sorted(range(0, 5), key = lambda x: -result[x])

for i in descending:
    print('%s: %f%%' % (labels[i], result[i] * 100.0))

