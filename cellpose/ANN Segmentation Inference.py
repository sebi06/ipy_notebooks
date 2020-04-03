#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

IMAGE_PATH = 'segment_nuclei_cnn.png'
MODEL_PATH = 'model_folder'

image = plt.imread(IMAGE_PATH)
# plt.imshow(image)
# plt.show()
image.shape

# If the input image is gray scale it would look like this
image = image[..., :1]
image.shape

# Add batch dimension
# The model expects the first dimension to be the batch dimension.
# If the loaded array does not have a batch dimension we need to add one.

image = image[np.newaxis]
# image.shape
# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Determine input shape required by the model and crop input image respectively
height, width = model.signatures["serving_default"].inputs[0].shape[1:3]
image = image[:, :height, :width]
#plt.imshow(image[0, ..., 0], cmap='gray')
# plt.show()

# Run prediction
prediction = model.predict(image)[0]  # Removes batch dimension

# Generate labels from one-hot encoded vectors
prediction_labels = np.argmax(prediction, axis=-1)

# get pixel values for all classes from prediction
classes = np.unique(prediction_labels)

# get the desired class
background = 0
nuclei = 1
borders = 2

# extract desired class
nuc = np.where(prediction_labels == nuclei, True, False)

# prediction_labels[prediction_labels]
plt.imshow(nuc, cmap='gray')
#plt.imshow(prediction_labels, cmap='gray')
plt.show()
