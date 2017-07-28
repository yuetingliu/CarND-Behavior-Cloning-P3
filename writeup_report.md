# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Nvidia-cnn-architecture.png "Model Visualization"
[image2]: ./images/center-line-driving.jpg "center line driving image"
[image3]: ./images/center-line-driving-flip.png "center line driving image flipped"
[image4]: ./images/data-distribution.png "data distribution"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My solution includes the following required files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

My solution includes the following additional files:
* video.py containing the script to create a video from the images the agent sees
* run1.mp4 is a video created from the images that the agent sees in the autonomous mode 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model resembles architecture used in the [Nvidia end to end self-driving paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
The architecture consists of five convolution neural network with three 5x5 filter sizes and two 3X3 filter sizes. The depths are between 24 and 64 (model.py lines 194-198). 
The activation functions for all conv layers are ELU. The data is normalized in the model using a Keras lambda layer (model.py code line 193). The model contains three fully
connected layers followed by the output layer (model.py code 202-206).  

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 201, 205). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but the learning rate was reduced to 0.0001 instead of the default 0.001 (model.py line 208).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and augmented data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to improve the results through iterations. 

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because LeNet worked well in extracting features from images, and it is simple to implement and run.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (5% validation set). I found that my first model could not make to car to drive autonomously. 
Then, I adopted the model from the [Nvidia paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). To my surprise, the model did not work well either. Then I thought about the quality of the data I collected.
My driving behavior was pretty bad, but the Data provided by Udacity should be fairly good. So I ditched my own data and used only the data of Udacity. The results were reasonably well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I increased the correction value (The value 
compensation for the images seen by left and right cameras). Through trail and error, I chose the values 0.95 (for left camera) and -0.95 (for right camera).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 190-206) consisted of a convolution neural network illustrated below 

Here is a visualization of the architecture 

![Model architecture][image1]

#### 3. Creation of the Training Set & Training Process

I used only Udacity data. Here is an example image of center lane driving:

![center line driving image][image2]


To augment the data sat, I also flipped images and angles thinking that this would simply gain more data. For example, here is the image. The left is the original image, and the right is the flipped image:

![center line driving image flipped][image3]


After the collection process, I had 32144 number of data points. But the distribution of the data was pretty bad, which was responsible for the unsuccessful autonomous driving. 
I then preprocessed this data by trimming some data to make the distribution more reasonable. The following image shows the distribution before and after trimming (left: original distribution; right: after trimming)
![data distribution][image4]

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by both training and validation losses were stopped dropping noticeably.
I used an adam optimizer with a learning rate 0.0001.
