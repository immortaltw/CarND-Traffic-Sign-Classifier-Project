# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_resources/dist_training.png "Training Set"
[image2]: ./writeup_resources/rgb.png "RGB"
[image3]: ./writeup_resources/normalized.png "Normalized"
[image4]: ./writeup_resources/01.png "Traffic Sign 1"
[image5]: ./writeup_resources/02.png "Traffic Sign 2"
[image6]: ./writeup_resources/03.png "Traffic Sign 3"
[image7]: ./writeup_resources/04.png "Traffic Sign 4"
[image8]: ./writeup_resources/05.png "Traffic Sign 5"
[image9]: ./writeup_resources/dist_valid.png "Validation Set"
[image10]: ./writeup_resources/dist_test.png "Test Set"


## Rubric Points
The writeup actually covers all the rubric points. See the writeup below for detailed information.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/immortaltw/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I used seaborn's countplot to show the distribution of training, validation and test set. Turned out that their distribution is veri similar. Some of the labels (~25%) have more total count than others, so they have more data for training. This "bias" might make the model trained more accurate when predicting those labels.

![Training Set][image1]
![Validation Set][image9]
![Test Set][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because I think the shapes of the traffic sign is more important than color. Also grayscale image somehow preserved truncated color information. It reduces the dimension of image data from 3 to 1, which can somehow shorten the computation time for training.

Here is an example of an original image and an augmented image:

![alt text][image2]
![alt text][image3]


The difference between the original data set and the augmented data set is that the characteristics of the image is amplified. Also many unnecessary details are removed.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected       | Input 400, Output 120                         |   
| Fully connected       | Input 120, Output 84                          |   
| Fully connected       | Input 84, Output 43                           |   


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with batch size 128 and 10 ephochs. I used 0.007 as my training rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.934
* test set accuracy of 0.915

I used the LeNet implementation in LeNet lab and did not change anything. I only tried different combinations of learning rate and ephochs. The original LeNet paper can achieve an accuracy of 0.98, so I think this architecture  should be good enough for this project.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because it is tilted. The others seem to be very easy to identify their characteristics.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve          | Double curve   								| 
| Dangerous curve to the right | Dangerous curve to the right 			|
| End of all speed and passing limits| End of all speed and passing limits|
| Ahead only	      	| Ahead only					 				|
| Vehicles over 3.5 tons prohibited|Vehicles over 3.5 tons prohibited   |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.911. I think the main reason is that these 5 images are too easy to identify even with human eyes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under the cell "Output Top 5 Softmax Probabilities For Each Image Found on the Web" in the Ipython notebook.



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Double curve   								| 
| .0     				| Right-of-way at the next intersection         |
| .0					| Beware of ice/snow                            |
| .0	      			| Wild animals crossing                         |
| .0				    | Slippery Road      							|


For the second image:

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .99999                | Dangerous curve to the right                  | 
| .000008               | Children crossing                             |
| .0000007              | Turn left ahead                               |
| .0                    | Go straight or right                          |
| .0                    | Road work                                     |

The third image:

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .9925                 | End of all speed and passing limits           | 
| .005                  | Speed limit (30km/h)                          |
| .00125                | Yield                                         |
| .00089                | End of no passing                             |
| .00014                | General caution                               |

Fourth:

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .999998               | Ahead only                                    | 
| .0000018              | No passing                                    |
| .0                    | Speed limit (60km/h)                          |
| .0                    | Turn left ahead                               |
| .0                    | Go straight or right                          |

Fifth:

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.00                  | Vehicles over 3.5 tons prohibited             | 
| .0                    | Speed limit (100km/h)                         |
| .0                    | No passing                                    |
| .0                    | Roundabout mandatory                          |
| .0                    | End of no passing by vehicles over 3.5 metric tons |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the first CNN layer, neural network tried to look for bigger shapes. In the second layer it tried to look for small details.


