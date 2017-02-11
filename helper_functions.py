#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import json
import math
import cv2
import csv


prob_threshold = 1
correction = 0.25

# Helper functions:


def drop_probability(value, probability=0.95):
    '''Callback function for Pandas to delete a row based on a probability.'''
    return value & bool(random.random() <= probability)


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
        plt.hist(steering_angles, bins=100)
        plt.title("Distribution of steering angles")
        plt.xlabel("Value")
        plt.xticks(np.arange(-1.1, 1.1, 0.1))
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
    # Crop image -- remove 62px pixels off the top and 25px off the bottom:
    img = img[62:shape[0] - 25, 0:shape[1]]
    # Resize to 200x66:
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    return img


def change_brightness(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * random_bright
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


# From Vivek Yadav's Medium post:
def trans_image(image, steer, trans_range):
    rows, cols, channels = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang


def flip_horizontal(img):
    return cv2.flip(img, 1)


def preprocess_image_training(line_data):
    choose_camera = np.random.randint(3)

    if choose_camera == 0:
        image_path = line_data['Left Image'][0].strip()
        shift_ang = correction
    if choose_camera == 1:
        image_path = line_data['Center Image'][0].strip()
        shift_ang = 0.
    if choose_camera == 2:
        image_path = line_data['Right Image'][0].strip()
        shift_ang = -correction

    steering_angle = line_data['Steering Angle Smooth'][0] + shift_ang
    image = load_image(image_path)
    image, steering_angle = trans_image(image, steering_angle, 150)
    image = change_brightness(image)
    image = crop_resize(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        image = flip_horizontal(image)
        steering_angle = -steering_angle

    return image, steering_angle


def preprocess_image_predict(line_data):
    image_path = line_data['Center Image'][0].strip()
    image = load_image(image_path)
    image = crop_resize(image)
    image = np.array(image)
    return image

# Batch and training/validation generators


def training_generator(pandas_data, batch_size=32):
    """
    Generator for use while training the model.

    Load each image and corresponding steering angle randomly from the Pandas dataset,
    and turn it into a batch that can be handled by the computer.
    :param pandas_data: DataFrame loaded from the simulator's CSV file.
    :param batch_size: Amount of the training images to be sent to the model.
    :return: Features and labels for training the model.
    """
    batch_features = np.zeros((batch_size, 66, 200, 3))
    batch_labels = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            line_choice = int(np.random.choice(len(pandas_data), 1))
            line_data = pandas_data.iloc[[line_choice]].reset_index()
            keep_prob = 0
            # Discard low steering angles if a probability is above a set threshold:
            # (This threshold is reduced while the model trains).
            while keep_prob == 0:
                batch_feature, batch_label = preprocess_image_training(line_data)
                if abs(batch_label) < .1:
                    prob_val = np.random.uniform()
                    if prob_val > prob_threshold:
                        keep_prob = 1
                else:
                    keep_prob = 1
            batch_features[i] = batch_feature
            batch_labels[i] = batch_label

        yield batch_features, batch_labels


def validation_generator(pandas_data):
    """
    Generator for use while validating the model.
    :param pandas_data: DataFrame loaded from the simulator's CSV file.
    :return: Features and labels for training the model.
    """
    # Validation generator
    while True:
        for i in range(len(pandas_data)):
            line_data = pandas_data.iloc[[i]].reset_index()
            feature = preprocess_image_predict(line_data)
            feature = feature.reshape(1, feature.shape[0], feature.shape[1], feature.shape[2])
            label = line_data['Steering Angle Smooth'][0]
            label = np.array([[label]])
            yield feature, label

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
    model.add(Dropout(0.50, name='Convo1_Dropout'))
    # Convolutional 2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_2'))
    model.add(Dropout(0.50, name='Convo2_Dropout'))
    # Convolutional 3
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_3'))
    model.add(Dropout(0.50, name='Convo3_Dropout'))
    # Convolutional 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_4'))
    model.add(Dropout(0.50, name='Convo4_Dropout'))
    # Convolutional 5
    model.add(Convolution2D(64, 3, 3, border_mode='valid',
                            activation='elu', dim_ordering='tf', name='Convolution_5'))
    model.add(Dropout(0.50, name='Convo5_Dropout'))
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
