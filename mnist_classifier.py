# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:54:41 2020

@author: surya

Handwritten digits classifier using the MNIST database.

Each image is of size 28 x 28 with maximum pixel value 255.
"""

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

# =============================================================================
# Define a callback function to use while training, so that if accuracy has
# reached an acceptable level, training may be stopped.
# =============================================================================
min_acc = 0.98


class custom_callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > min_acc):
            print('\nAccuracy is above {}, stopping training'.format(min_acc))
            self.model.stop_training = True


callback = custom_callback()  # instantiate a callback object to use later


# =============================================================================
# Define a plot function that takes data of one image (28, 28) and displays
# it. Use this throughout the code to visualise data where necessary.
# pred=None for training data, pred is set to the predictions for test data.
# =============================================================================
def plot_figure(img_data, label, fig_index, pred=None):
    if len(fig_index) % 5 == 0:
        rows = (len(fig_index))/5
    else:
        rows = round(len(fig_index)/5) + 1
    for i, val in enumerate(fig_index):
        plt.subplot(rows, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.imshow(img_data[val], cmap=plt.cm.binary)
        if pred is None:
            plt.xlabel('Digit ' + str(label[val]))
        else:
            p = np.argmax(pred[val])  # index with highest prob. is the label
            plt.xlabel('Digit ' + str(label[val]) + 
                       '\nPredicted ' + str(p))
    if pred is None:
        plt.suptitle('Training data')
    else:
        plt.suptitle('Predictions on test data')
    plt.show()


# =============================================================================
# Import the MNIST dataset from keras and split data into train and test sets.
# Normalise the images by dividing with the largest pixel value (255).
# Visualise a few images by changing index range.
# =============================================================================
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train size: {}'.format(X_train.shape))
print('X_test size: {}'.format(X_test.shape))

X_train, X_test = X_train/255, X_test/255
# indices of images to display, upto X_train.shape[0]
fig_index = list(range(0, 20))
plot_figure(X_train, y_train, fig_index)

# =============================================================================
# Build a sequential model using the keras interface.
# Experiment with various parameter values for number of neurons, dropout rate,
# activation function etc.
# =============================================================================
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),  # dropout layer prevents overfitting
    keras.layers.Dense(512),
    ])
print(model.summary())

# =============================================================================
# Use a loss function (here, sparse categorical cross entropy) to estimate the
# performance of the model.
# =============================================================================
loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# =============================================================================
# Compile and train the model, estimate accuracy using loss function and
# test the model on the test/validation data
# Use the callback function to stop training for more epochs if a minimum
# acceptable accuracy has been attained.
# =============================================================================
model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=['accuracy'],
    )
model.fit(X_train, y_train, epochs=10, callbacks=[callback])
print('\nModel performance on test data:')
model.evaluate(X_test, y_test, verbose=2)

# =============================================================================
# Add a softmax layer to the model to get a class label as output.
# The index with the highest predicted probability is the predicted 
# class label.
# Visualise model predictions and compare with actual labels.
# Change index to see different images.
# =============================================================================
probability_model = keras.Sequential([model, 
                                         keras.layers.Softmax()])
predictions = probability_model.predict(X_test)

fig_index = list(range(4, 18))  # within the size limits of X_test
plot_figure(X_test, y_test, fig_index, predictions)
