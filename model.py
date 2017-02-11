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
output_steering_plot = False

# Workflow:
print('Loading CSV file...')

csv_file = pd.read_csv('driving_log.csv', sep=',', header=None, names=[
                       'Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Brake', 'Speed'])

# Cleanup the CSV from unnecessary data:
# Before cleanup stats:
helper.print_csv_report(csv_file, output_steering_plot)
# Cleanup the image path for all files (in case it already isn't):
csv_file['Center Image'] = csv_file['Center Image'].apply(helper.make_imagepath_relative)
csv_file['Left Image'] = csv_file['Left Image'].apply(helper.make_imagepath_relative)
csv_file['Right Image'] = csv_file['Right Image'].apply(helper.make_imagepath_relative)
# Truncate (smooth) steering angles:
steering_angles = np.array(csv_file['Steering Angle'])
csv_file['Steering Angle Smooth'] = steering_angles
# Remove low throttle data:
csv_file = csv_file[csv_file['Throttle'] > .25]
# After cleanup stats:
helper.print_csv_report(csv_file, output_steering_plot)

# NOTE: not done because I'll let the dataset filter itself, by augmentation.
# The dataset is extremely biased towards straight driving, so let's discard some random images
# with steering angles that are zero or close to it:
# csv_file['close_zero'] = np.isclose(csv_file['Steering Angle'], 0, 1.5e-1)
# csv_file['drop_row'] = csv_file['close_zero'].apply(helper.drop_probability)
# csv_file = csv_file[csv_file['drop_row'] == False]
# csv_file.drop('drop_row', axis=1)
# csv_file.drop('close_zero', axis=1)

# Init generator:
# NOTE: The full recorded dataset can be used in this case as a validation set,
# since the training set is augmented from it (the probability of using an actual,
# unaltered image from it is very low.)
valid_generator = helper.validation_generator(csv_file)

# Init model
print('Compiling the model...')
model = helper.nvidia_model()
adam_optimizer = Adam(lr=1e-4)
model.compile(optimizer=adam_optimizer, loss='mse', metrics=['accuracy'])
model.summary()

# Training loop static values:
resize_w, resize_h = 200, 66
training_cycles = 10
validation_multiplier = 1  # NotImplemented (yet)
validation_size = int(validation_multiplier * len(csv_file))
batch_size = 256
sample_multiplier = 3
samples_epoch = int(sample_multiplier * (math.floor(len(csv_file) / batch_size) * batch_size))
prob_threshold = 1
best_cycle = 0
val_best = 1000

# Training loop -- searching for lowest validation accuracy model:
print('Starting training...')
print('Initial features to train with: {}'.format(len(csv_file)))
print('Batch size: {}'.format(batch_size))
print('Num. generated samples per epoch: {}'.format(samples_epoch))

for cycle_nb in range(training_cycles):
    print('Training cycle {} of {}...'.format(cycle_nb, training_cycles-1))
    train_generator = helper.training_generator(csv_file, batch_size)
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=samples_epoch,
                                  nb_epoch=1,
                                  validation_data=valid_generator,
                                  nb_val_samples=validation_size)
    filename = 'model_' + str(cycle_nb) + '.h5'
    print('Train cycle {}: complete. Saving model...'.format(cycle_nb))
    model.save(filename)

    val_loss = history.history['val_loss'][0]
    print('Validation loss: {}.'.format(val_loss))
    if val_loss < val_best:
        best_cycle = cycle_nb
        val_best = val_loss
        model.save('model_best.h5')
    prob_threshold = 1 / (cycle_nb + 1)

print('Best model found at iteration # ' + str(i_best))
print('Best Validation score : ' + str(np.round(val_best, 4)))
print('Training done.')
