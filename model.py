#!/usr/bin/env python
from keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import helper_functions as helper
import pandas as pd
import numpy as np
import math

# Static values and initialization:
resize_w, resize_h = 200, 66
batch_size = 128
number_epochs = 5
correction = 0.25
output_steering_plot = False

# Workflow:
print('Loading CSV file...')

csv_file = pd.read_csv('driving_log.csv', sep=',', header=None, names=[
                       'Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Brake', 'Speed'])

# Cleanup the image path for all files (in case it already isn't, unlike the Udacity data):
csv_file['Center Image'] = csv_file['Center Image'].apply(helper.make_imagepath_relative)
csv_file['Left Image'] = csv_file['Left Image'].apply(helper.make_imagepath_relative)
csv_file['Right Image'] = csv_file['Right Image'].apply(helper.make_imagepath_relative)

# The dataset is extremely biased towards straight driving, so let's discard some random images
# with steering angles that are zero or close to it:
csv_file['close_zero'] = np.isclose(csv_file['Steering Angle'], 0, 1.5e-1)
csv_file['drop_row'] = csv_file['close_zero'].apply(helper.drop_probability)
csv_file = csv_file[csv_file['drop_row'] == False]
csv_file.drop('drop_row', axis=1)
csv_file.drop('close_zero', axis=1)

helper.print_csv_report(csv_file, output_steering_plot)

# Now let's add the relevant data to the lists:
center_images = csv_file['Center Image'].values
left_images = csv_file['Left Image'].values
right_images = csv_file['Right Image'].values
center_measurements = csv_file['Steering Angle'].values
left_measurements = csv_file['Steering Angle'].values + correction
right_measurements = csv_file['Steering Angle'].values - correction

# Split the center image dataset into training and validation:
center_images, center_measurements = shuffle(
    center_images, center_measurements)
center_images, X_valid, center_measurements, y_valid = train_test_split(
    center_images, center_measurements, test_size=0.35)

print('Training and validation sets split -- Training: {}, Validation: {}'.format(
    len(center_images), len(X_valid)))

# Now combine the rest of the images into the center dataset:
left_images, left_measurements = shuffle(
    left_images, left_measurements)
right_images, right_measurements = shuffle(
    right_images, right_measurements)
X_train = np.concatenate((center_images, left_images, right_images), axis=0)
y_train = np.concatenate((center_measurements, left_measurements, right_measurements), axis=0)

print('Training set size with side-camera images: {}'.format(len(X_train)))

# Free up memory:
del(center_images, left_images, right_images, center_measurements, left_measurements, right_measurements)

# Init generators:

# NOTE: Not used since for some reason, shuffling the dataset works now.
# get_train_batch = helper.generate_randomized_batch(X_train, y_train, batch_size)
# train_generator = helper.training_generator(batch_size, get_train_batch)

train_generator = helper.training_generator(X_train, y_train, batch_size)
valid_generator = helper.validation_generator(X_valid, y_valid, batch_size)

# Test - distribution of generated images:
# (To see if the generator was working properly and outputting sane values.)

# gen_features = []
# gen_labels = []
# for i in range(0, 10):
#     features, labels = next(get_train_batch)
#     gen_features.extend(features)
#     gen_labels.extend(labels)
# print('Retrieved {} samples.'.format(len(gen_labels)))

# proc_gen_features = []
# proc_gen_labels = []
# test_train_generator = helper.training_generator(X_train, y_train, batch_size)
# for i in range(0,100):
#     processed_features, processed_labels = next(train_generator)
#     proc_gen_features.extend(processed_features)
#     proc_gen_labels.extend(processed_labels)
# show_index = 0
# plt.imshow(proc_gen_features[show_index])
# plt.title('Steering angle: {}'.format(proc_gen_labels[show_index]))
# plt.show()
#
# print('Generated {} samples from training set.'.format(len(proc_gen_labels)))
# print(np.unique(proc_gen_labels, return_counts=True))
# print('Distribution of steering angles:')
# plt.hist(proc_gen_labels, bins=200)
# plt.title("Distribution of steering angles")
# plt.xlabel("Value")
# plt.xticks(np.arange(-1.0, 1.0, 0.1))
# plt.ylabel("Frequency")
# plt.show()
# exit()

# Init model
print('Compiling the model...')
model = helper.nvidia_model()
adam_optimizer = Adam(lr=1e-3)
model.compile(optimizer=adam_optimizer, loss='mse', metrics=['accuracy'])
model.summary()

# Start training session and save after complete:
print('Starting training...')
# Make sure that it can train on all the possible batches (no incomplete batches allowed):
samples_epoch = (math.floor(len(X_train) / batch_size) * batch_size)
model.fit_generator(train_generator,
                    samples_per_epoch=samples_epoch,
                    nb_epoch=number_epochs,
                    validation_data=valid_generator,
                    nb_val_samples=len(X_valid),
                    verbose=1)

print('Training complete. Saving model...')
model.save('model.h5')

# Finished!
print('Model saved!')
