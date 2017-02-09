#!/usr/bin/env python
import helper_functions

# Static values and initialization:
new_size_col, new_size_row = 200, 66
batch_size = 16
lines = []
center_images = []
left_images = []
right_images = []
center_measurements = []
left_measurements = []
right_measurements = []
correction = 0.25


# Workflow:
print('Loading CSV file...')
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


print('Getting data...')
for line in lines:
    for i in range(3):
        source_path = line[i]  # center image, then left, then right
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = './IMG/' + filename
        image = load_image(local_path)
        if i == 0:
            center_images.append(image)
        if i == 1:
            left_images.append(image)
        if i == 2:
            right_images.append(image)
    measurement = float(line[4])
    center_measurements.append(measurement)  # center, no correction
    # left, softer turn to the left
    left_measurements.append(measurement + correction)
    # right, softer turn to the right
    right_measurements.append(measurement - correction)

print('File loaded. Got {} images.'.format(3 * len(center_measurements)))

# Split the center dataset into training and validation:
center_images, center_measurements = shuffle(
    center_images, center_measurements)
center_images, X_valid, center_measurements, y_valid = train_test_split(
    center_images, center_measurements, test_size=0.10)

print('Training and validation sets split. Training: {}, Validation: {}'.format(
    len(center_images), len(X_valid)))

# Now combine the rest of the images into the center dataset:
images = center_images + left_images + right_images
measurements = center_measurements + left_measurements + right_measurements

# Turn them into np.array:
X_train = np.array(images)
y_train = np.array(measurements)

print('Training set size with side-camera images: {}'.format(len(X_train)))

# Free up memory
del(images, measurements, center_images, left_images, right_images, center_measurements,
    left_measurements, right_measurements)

# Init generator:
train_generator = training_generator(batch_size)
valid_generator = validation_generator(X_valid, y_valid, batch_size)
get_train_batch = generate_randomized_batch(X_train, y_train, batch_size)

# Init model
print('Compiling the model...')
model = nvidia_model()
model.compile(optimizer='adam', loss='mse')
model.summary()

# Start training session and save after complete:
print('Starting training...')
model.fit_generator(train_generator,
                    # make sure that it can train on all the data:
                    samples_per_epoch=(math.floor(
                        len(X_train) / batch_size) * batch_size),
                    nb_epoch=3,
                    validation_data=valid_generator,
                    nb_val_samples=len(X_valid))

print('Training complete. Saving model...')
model.save('model.h5')

# Workflow:

# Load CSV file
# csv_file = pd.read_csv('driving_log.csv', sep=',', header=None, names=[
#                       'Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Brake', 'Speed'])

# Cleanup the image path for all files:
# csv_file['Center Image'] = csv_file[
#    'Center Image'].apply(make_imagepath_relative)
#csv_file['Left Image'] = csv_file['Left Image'].apply(make_imagepath_relative)
# csv_file['Right Image'] = csv_file[
#    'Right Image'].apply(make_imagepath_relative)

#print_csv_report(csv_file, False)

# Get a summary of the model and then train it:
#model = nvidia_model()
# model.summary()
#model.compile(optimizer='adam', loss='mse')

# Initialize the training images generator:
#training_generator = generate_training_batches(csv_file)

# Train with all images (left, center, right), 5 epochs:
# history = model.fit_generator(
#    training_generator, len(csv_file['Steering Angle']) * 3, 5)
#print('Model trained.')

# Save the weights and model to their respective files:
# model.save_weights('model.h5')
# with open('model.json', 'w') as model_file:
#    json.dump(model.to_json(), model_file)

# Finished!
