#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import math
import cv2

### Reporting and static values:

new_size_col, new_size_row = 200, 66
pr_threshold = 1

''' Report values loaded from CSV file '''
def print_csv_report(csv_file, output_plots=True):
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

''' Convert the loaded filepaths from the CSV file into relative paths '''
def make_imagepath_relative(path):
    return './' + path[path.index('IMG'):]

### Image (pre)processing:

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def normalize(image):
    image = (image / 255) - 0.5
    return image

def change_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    return cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

def flip_horizontal(image):
    return cv2.flip(image, 1)

def crop_resize(image):
    shape = image.shape
    # Crop image -- remove 1/5 off the bottom and 25 pixels on top:
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    # Resize to 200x66:
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
    return image

def preprocess_image_file_training(line_data):
    choose_camera = np.random.randint(3)
    if (choose_camera == 0):
        path_file = line_data['Left Image'][0]
        shift_ang = .25
    if (choose_camera == 1):
        path_file = line_data['Center Image'][0]
        shift_ang = 0.
    if (choose_camera == 2):
        path_file = line_data['Right Image'][0]
        shift_ang = -.25
    y_steer = line_data['Steering Angle'][0] + shift_ang
    image = load_image(path_file)
    image = change_brightness(image)
    image = crop_resize(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = flip_horizontal(image)
        y_steer = -y_steer
    
    return image, y_steer

def create_batch(data, batch_size):
    while True:
        for i in range(batch_size):
            x, y = preprocess_image_file_training(i, data)
            image_batch[i] = x
            steering_batch[i] = y
        yield image_batch, steering_batch

''' This is a generator that takes in a Pandas DataSet and processes it in batches '''
def generate_training_batches(data, batch_size = 32):
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while True:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            
            keep_pr = 0
            #x,y = preprocess_image_file_training(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_training(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

### Model Definition / Operations:

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def nvidia_model():

    model = Sequential()
    # Input + Normalization:
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=(66, 200, 3), output_shape=(66, 200, 3), name='Input_Normalization')) #))
    
    # NVIDIA paper says:
    # "We use strided convolutions in the first three convolutional layers
    #  with a 2×2 stride and a 5×5 kernel and a non-strided convolution
    #  with a 3×3 kernel size in the last two convolutional layers."

    # Convolutional 1
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering='tf', name='Convolution_1'))
    model.add(Dropout(0.5, name='Convo1_Dropout'))
    # Convolutional 2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering='tf', name='Convolution_2'))
    model.add(Dropout(0.5, name='Convo2_Dropout'))
    # Convolutional 3
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering='tf', name='Convolution_3'))
    model.add(Dropout(0.5, name='Convo3_Dropout'))
    # Convolutional 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', dim_ordering='tf', name='Convolution_4'))
    model.add(Dropout(0.5, name='Convo4_Dropout'))
    # Convolutional 5
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', dim_ordering='tf', name='Convolution_5'))
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

### Workflow:

# Load CSV file
csv_file = pd.read_csv('driving_log.csv', sep=',', header=None, names=['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Brake', 'Speed'])

# Cleanup the image path for all files:
csv_file['Center Image'] = csv_file['Center Image'].apply(make_imagepath_relative)
csv_file['Left Image'] = csv_file['Left Image'].apply(make_imagepath_relative)
csv_file['Right Image'] = csv_file['Right Image'].apply(make_imagepath_relative)

print_csv_report(csv_file, False)

# Get a summary of the model and then train it:
model = nvidia_model()
model.summary()

model.compile(optimizer='adam', loss='mse')

# Initialize the training images generator:
training_generator = generate_training_batches(csv_file)

# Train with all images (left, center, right), 5 epochs:
history = model.fit_generator(training_generator, len(csv_file['Steering Angle'])*3, 5)
print('Model trained.')

# Save the weights and model to their respective files:
model.save_weights('model.h5')
with open('model.json', 'w') as model_file:
    json.dump(model.to_json(), model_file)

# Finished!
