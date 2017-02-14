#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/center_driving.png "Center Image"
[image2]: ./writeup_imgs/left_recovery.png "Left Recovery Image"
[image3]: ./writeup_imgs/right_recovery.png "Right Recovery Image"
[image4]: ./writeup_imgs/generated_images.png "Generator Output"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted and Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* helper_functions.py containing the various preprocessing functions for the model
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and drive.py file (modified to include the same preprocessing functions as the model), the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. Additionally, helper_functions.py hold the actual model architecture and the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of an implementation of NVIDIA's End-to-End model, which is simple enough that it can be replicated in Keras with very little effort, so it seemed like a good idea to do so in this project.

It should be noted that, rather than classification (which was done in [project 2](https://github.com/kenshin23/CarND-Traffic-Sign-Classifier-Project)), this is a regression problem, where the neural network is shown an image, and the output of the network should be a steering angle, to keep the car inside the drivable portion of the track, as noted on the project's review rubric.

The model includes Exponential Linear Unit (ELU) activations in each convolutional layer to introduce nonlinearity (code lines 236, 240, 244, 248, 252 in helper_functions.py), and the data is normalized in the model using a Keras lambda layer (code line 227). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (helper_functions.py lines 238, 242, 246, 250, 254, 257). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. This was achieved by using a generator which augments data on-the-fly, using the provided images obtained from driving around the track in the simulator, in training mode. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 72).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The specific process is detailed below, in the "Creation of the Training Set" section.

The resulting images and CSV file were moved to a GPU-enabled Ubuntu Linux local instance, where they were to be processed.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My approach in solving the problem, was to look for models that have achieves good results on this particular problem. Course classmates have used the NVIDIA model or the self-driving model from [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py), or other successful models, either pre-trained or trained from scratch, so I decided to try and implement the NVIDIA model.

Then, I drove around the track as detailed below, getting data which appeared balanced, in order to provide good training data for the model.

To make sure the dataset was as balanced as possible, I had initially gotten rid of approximately 4/5 of straight driving, since it dominates the training set and therefore makes the model biased towards driving straight as well, but since I started to use on/the/fly image augmentation, I decided it wasn't really necessary.

After, I trained the model as explained below.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, in particular the area right of the first curve to the left, then the fork on the road after the bridge, and then just after the S turns. To try and help the model get through those areas, I used [Thomas Antony's live trainer](https://github.com/thomasantony/sdc-live-trainer),  in order to gather more training data for each trouble area, which helped immensely.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (helper_functions.py lines 222-271) consisted of a convolution neural network with the following layers and layer sizes:

```____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
Input_Normalization (Lambda)     (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
Convolution_1 (Convolution2D)    (None, 31, 98, 24)    1824        Input_Normalization[0][0]        
____________________________________________________________________________________________________
Convolution_2 (Convolution2D)    (None, 14, 47, 36)    21636       Convolution_1[0][0]              
____________________________________________________________________________________________________
Convolution_3 (Convolution2D)    (None, 5, 22, 48)     43248       Convolution_2[0][0]              
____________________________________________________________________________________________________
Convolution_4 (Convolution2D)    (None, 3, 20, 64)     27712       Convolution_3[0][0]              
____________________________________________________________________________________________________
Convolution_5 (Convolution2D)    (None, 1, 18, 64)     36928       Convolution_4[0][0]              
____________________________________________________________________________________________________
Flatten (Flatten)                (None, 1152)          0           Convolution_5[0][0]              
____________________________________________________________________________________________________
Flatten_Dropout (Dropout)        (None, 1152)          0           Flatten[0][0]                    
____________________________________________________________________________________________________
Fully_Connected_0 (Dense)        (None, 1152)          1328256     Flatten_Dropout[0][0]            
____________________________________________________________________________________________________
Fully_Connected_1 (Dense)        (None, 100)           115300      Fully_Connected_0[0][0]          
____________________________________________________________________________________________________
Fully_Connected_2 (Dense)        (None, 50)            5050        Fully_Connected_1[0][0]          
____________________________________________________________________________________________________
Fully_Connected_3 (Dense)        (None, 10)            510         Fully_Connected_2[0][0]          
____________________________________________________________________________________________________
Output (Dense)                   (None, 1)             11          Fully_Connected_3[0][0]          
====================================================================================================
Total params: 1,580,475
Trainable params: 1,580,475
Non-trainable params: 0
```

#### 3. Creation of the Training Set and Training Process

To capture good driving behavior, the simulator was run on Windows 10 x64 using a PlayStation 4 controller to try and keep inputs smooth. Approximately 5-6 laps were run around the test (left) course, with 2 additional laps training for recovery, i.e.: one where I recorded driving going back to center from the right part of the track, and a similar one but from the left side. To avoid bias towards straight driving, 5-6 more laps were run around the track, but in the opposite direction. Here is an example image of center lane driving:

![Center Driving][image1]

Here are examples of left and right-side recovery, to teach the car what to do when it is too far left or right from the center of the road. These images show what a recovery looks like, from the left and right sides of the road, respectively:

![Left Recovery][image2]
![Right Recovery][image3]

To augment the data set, I originally used the entire training set as a validation set, and instead used cropping, resizing, brightness changes, image translations, and horizontal flips, to create an entirely different training set using a Keras generator, all suggested by Vivek Yadav's Medium post. For example, here is the output of the generator for a batch size of 16:

![Generator output (Data Augmentation)][image4]

After the collection process, I had 13698 data points, which tripled if you include left and right cameras.

I used this training data for training the model. I trained the model for a set number of epochs, while looking at the validation loss. I set to implement an early-stopping callback to the model so that it would stop training when the validation loss would stop decreasing, but did so manually instead. 

