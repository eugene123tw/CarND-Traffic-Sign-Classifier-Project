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

[image1]: ./examples/y_train.png "Visualization"
[image2]: ./examples/Brightness.png "Brightness.png"
[image3]: ./examples/Rotate.png "Rotation"
[shift]: ./examples/Shift.png "Shift"
[image4]: ./examples/30km_1.jpg "30km"
[image5]: ./examples/50km_2.jpg "50km"
[image6]: ./examples/60km_3.jpg "60km"
[image7]: ./examples/No_entry_17.jpg "No entry"
[image8]: ./examples/pedestrians_27.jpg "pedestrians"
[image9]: ./examples/stop_14.jpg "Stop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32x3
* The number of unique classes/labels in the data set is: 43 

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing all the label counts in traning set. 
The number of images for each label is highly uneven.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to add brightness to images. I wrote a program randomly choose images to add brightness with random value. 
Here is an example of a traffic sign image before and after adding brightness.

![alt text][image2]

Then, I randomly add shift and rotation to image because I want my model to be more generalized to unseen images and less sensitive to noise.

Here are examples of an original image and a rotation image:

![alt text][image3]

Here are examples of an original image and a shift image:
![alt text][shift]

Finally, I normalized pixel values between 0-1 to staballize the variance between pixels. It helps during training.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, no padding, outputs 28x28x32 	    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, no padding, outputs 10x10x64		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Fully connected		| Input = 1600. Output = 400.					|
| Batch Norm.			| 						                        |
| RELU					|												|
| Fully connected		| Input = 400. Output = 120.					|
| Batch Norm.			| 						                        |
| RELU					|												|
| Fully connected		| Input = 120. Output = 43.		     			|
| Logits     			|												|
| Softmax cross-entropy	|                                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, 16 images as a batch. Set the learninig rate to 0.001 and trained 40 epochs. 
During traning, 34799 of training images are suffled randomly. Images are rotated, shifted or added brightness if the generated probabllity from U(0,1) is less than 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy : 0.9865  
* validation set accuracy : 0.9624 
* test set accuracy : 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
Ans: The ordinary LeNet5.
* What were some problems with the initial architecture? 
Ans: The kernal size of LeNet5 is too small. The model can be easily overfitted since there are no regularizers.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Ans: 
1. Variance are added to the images (Rotate, Brightness, Shift) as regularizer to noise
2. Added batch norm layer as regularizer
3. Increase Conv Layer output kernel size. It helps the model to learn more features.

* Which parameters were tuned? How were they adjusted and why?
Ans: Increase Conv Layer output kernel size. It helps the model to learn more features.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Ans: Convolution works well in image, since conv layer can learn spatial information from images. Dropout helps to regularize the model, it prevents the model from overfitting.


If a well known architecture was chosen:
* What architecture was chosen? 
Ans: I didn't use any other Net.
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The images are all very clear. It's should not be too difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		    | Speed limit (30km/h)    						| 
| 50 km/h     			| Speed limit (50km/h) 						    |
| 60 km/h				| Speed limit (60km/h)							|
| No Entry			    | No entry      	       						|
| Pedestrians			| Speed limit (30km/h)    						|
| Stop			        | Stop      							        |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%.
This compares favorably to the accuracy on the test set of 95%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.94), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .941         			| Speed limit (30km/h)   									| 
| .065     				| Speed limit (50km/h) 										|
| .003					| Traffic signals											|
| .000	      			| Speed limit (100km/h)					 				|
| .000				    | Speed limit (120km/h)      							|


For the second image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 0.87), and the image does contain a Speed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .873         			| Speed limit (50km/h)   									| 
| .065     				| Speed limit (30km/h) 										|
| .003					| Speed limit (60km/h)											|
| .000	      			| Stop					 				|
| .000				    | Speed limit (80km/h)      							|

For the third image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 0.95), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .945					| Speed limit (60km/h)											|
| .003         			| Speed limit (50km/h)   									| 
| .001     				| Speed limit (30km/h) 										|
| .000	      			| Stop					 				|
| .000				    | No passing      							|

For the fourth image, the model is relatively sure that this is a No entry sign (probability of 1.00), and the image does contain a No entry sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000					| No entry											|
| .000         			| Stop   									| 
| .000     				| Vehicles over 3.5 metric tons prohibited 			|
| .000	      			| Speed limit (100km/h)					 				|
| .000				    | No passing for vehicles over 3.5 metric tons  							|

For the fifth image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.66). However, the image is a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .664					| Speed limit (30km/h)			     			|
| .207         			| No passing   									| 
| .103     				| General caution 			                    |
| .002	      			| Traffic signals					 			|
| .000				    | Go straight or right  						|

For the sixth image, the model is relatively sure that this is a Stop sign (probability of 1.00), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000					| Stop											|
| .000         			| No entry   									| 
| .000     				| Speed limit (50km/h) 			|
| .000	      			| Speed limit (30km/h)					 				|
| .000				    | Right-of-way at the next intersection  							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?