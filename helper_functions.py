#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import math
import cv2
import csv


# Helper functions:


def print_csv_report(csv_file, output_plots=True):
    '''Report values loaded from CSV file'''
    steering_angles = csv_file['Steering Angle'].values
    print('Number of images: {}'.format(len(csv_file.values)))
    print('Mean speed: {}'.format(np.mean(csv_file['Speed'].values)))
    print('Mean steering angle: {}'.format(np.mean(steering_angles)))
    print('Minimum steering angle: {}'.format(np.amin(steering_angles)))
    print('Maximum steering angle: {}'.format(np.amax(steering_angles)))
    # Maybe we're running this in the console, so skip plotting:
    if output_plots:
        print('Distribution of steering angles:')
        plt.hist(steering_angles, bins=1000)
        plt.title("Distribution of steering angles")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def make_imagepath_relative(path):
    '''Convert the loaded filepaths from the CSV file into relative paths'''
    return './' + path[path.index('IMG'):]

# Image (pre)processing:


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize(image):
    image = (image / 255) - 0.5
    return image


def crop_resize(img):
    shape = img.shape
    # Crop image -- remove 25 off the bottom and 70 pixels on top:
    img = img[70:shape[0] - 25, 0:shape[1]]
    # Resize to 200x66:
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    return img


def change_brightness(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * random_bright
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def flip_horizontal(img):
    return cv2.flip(img, 1)

# Batch and training/validation generators


def generate_randomized_batch(features, labels, batch_size):
    '''Load each image and corresponding steering angle randomly from dataset'''
    while True:
        batch_features = []
        batch_labels = []
        for i in range(batch_size):
            choice = int(np.random.choice(len(features), 1))
            batch_features.append(features[choice])
            batch_labels.append(labels[choice])
        # print('Batch size in batch_gen: {}'.format(len(batch_features)))
        yield batch_features, batch_labels


def training_generator(batch_size):
    while True:
        batch_features = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)
        features, labels = next(get_train_batch)
        for i in range(len(features)):
            # Load the image and label:
            batch_feature = features[i]
            batch_label = labels[i]
            # Crop to needed dimensions:
            batch_feature = crop_resize(batch_feature)

            # Change brightness?
            coin_brightness = random.randint(0, 1)
            if coin_brightness == 1:
                batch_feature = change_brightness(batch_feature)

            # Flip image?
            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                batch_feature = flip_horizontal(batch_feature)
                batch_label = -batch_label

            batch_features[i] = batch_feature
            batch_labels[i] = batch_label
        # print('Batch size in train_gen: {}'.format(len(batch_features)))
        yield batch_features, batch_labels


def validation_generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)
    while True:
        features, labels = shuffle(features, labels)
        for i in range(batch_size):
            choice = int(np.random.choice(len(features), 1))
            batch_label = labels[choice]
            # Load the image:
            batch_feature = features[choice]
            # Crop to needed dimensions:
            batch_feature = crop_resize(batch_feature)

            batch_features[i] = batch_feature
            batch_labels[i] = batch_label
        yield batch_features, batch_labels

# Models:


def nvidia_model():

    model = Sequential()

    # Input + Normalization:
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(66, 200, 3),
                     output_shape=(66, 200, 3), name='Normalization'))

    # NVIDIA paper says:
    # "We use strided convolutions in the first three convolutional layers
    #  with a 2×2 stride and a 5×5 kernel and a non-strided convolution
    #  with a 3×3 kernel size in the last two convolutional layers."

    # Convolutional 1
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_1'))
    model.add(Dropout(0.5, name='Convo1_Dropout'))
    # Convolutional 2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_2'))
    model.add(Dropout(0.5, name='Convo2_Dropout'))
    # Convolutional 3
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_3'))
    model.add(Dropout(0.5, name='Convo3_Dropout'))
    # Convolutional 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_4'))
    model.add(Dropout(0.5, name='Convo4_Dropout'))
    # Convolutional 5
    model.add(Convolution2D(64, 3, 3, border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_5'))
    model.add(Dropout(0.5, name='Convo5_Dropout'))
    # Flatten
    model.add(Flatten(name='Flatten'))
    # Fully Connected (Dense) 1
    model.add(Dense(100, name='Fully_Connected_1'))
    # Fully Connected (Dense) 2
    model.add(Dense(50, name='Fully_Connected_2'))
    # Fully Connected (Dense) 3
    model.add(Dense(10, name='Fully_Connected_3'))
    # Output
    model.add(Dense(1, name='Output'))
    return model
