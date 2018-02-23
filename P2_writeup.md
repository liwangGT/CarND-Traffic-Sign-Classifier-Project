# **Traffic Sign Recognition** 

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

[image1]: ./images/relative_data.png "bar plot1"
[image2]: ./images/accuracy_origin_bar.png "bar plot2"
[image3]: ./images/data_augment.png "accuracy plot"
[image4]: ./images/layer_visual.png "layer plot"
[image5]: ./images/relative_data_augment.png "relative augment"
[image6]: ./images/hiddenlayer1.png "hidden1"
[image7]: ./images/hiddenlayer2.png "hidden2"
[image8]: ./images/augment.png "augment"
[image9]: ./images/refdata.png "refdata"
[image10]: ./images/webdata.png "webdata"
[image11]: ./images/accuracy_after.png "accafter"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The original data set provided by for the project has the following properties.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distributed in training, validation, and testing data set. Note that since the magnitue of these three sets of data are different (i.e., 34799:4410:12630), they are all normalized by the maximum number of samples in each catergory. 

![data distribution][image1]

It can be concluded that the distribution of the training, validation, and testing data sets for each traffic sign is similar. However, some traffic signs have lots of training data, while some have very limited number of data. This will lead to problem of overfitting, and will be addressed in the data preprocessing section.

In order to better understand the traffic sign data, six random samples from the training data set are extracted and plotted below. It can be seen that some traffic signs are very blurry or dark.

![sample ref][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For the preprocessing of data, three different techniques are used.

* RGB to grayscale.

When color is considered in the training process, the overall amount of computation is multiplied by 3 with little improvement to the final result. Here, we argue that color plays smaller role as compared to other features in the image. The RGB images are converted into grayscale images to increase the computation efficiency. This is done by performing a weighted sum of RGB channels ([0.299, 0.587, 0.114]) using the [luminosity method](https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm). 

* Normalization of the image data

When the image data are scaled back into [-1, 1], the learning algorithm is expected to converge much faster. Thus, all training, validation, and testing data are normalized to [-1, 1].

* Augmenting training data

When the original training data is used to train the CNN model, it is observed that signs with insufficient training data are easily misclassified. This fact can be visualized with the following plot. For easier comparison, the numbers of training data for different traffic signs are normalized by the largest number of a single traffic sign. 

![accuracy vs. distribution][image2]

It can be observed that low validation accuracy is almost always related to low number of training data. The classification result is biased towards traffic signs with more data. To address this problem, additional training data are generated. 


Generating more training data is actually not a trivial task. The data augmentation code is provided in the "data_augment.ipynb". To save computation time, the additional data is generated only once then used for all models. There are three methods used to randomly generate new training data: shifting, rotation, and changing contrast. An illustrative example of these three techniques are visualized below. 
![example augment][image8]

To ensure enough training data is provided for each type of traffic sign, additional data are generated for those types with less than 800 training images. The comparison between the original and augmented training data set is provided in the figure below.

![ba augment][image5]

It can be observed that the accuracy across every traffic sign is more evenly distributed after data augmentation. As seen in the random samples, some images are very blurry or dark to classify. The classification accuracy for these traffic signs can be relatively low. 

![accuracy augment][image11]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					| introduce nonlinearity	    				|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU                  |  introduce nonlinearity                   |
| Max pooling           | 2x2 stride,  outputs 5x5x32                   |
| Flatten               | output 800                                    |
| Dropout               | keep probability = 0.8                        |
| Fully connected		| 200        									|
| RELU                  | introduce nonlinearity                        |
| Fully connected       | 43                                            |
| Softmax				| scale logits to [0,1] 						|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam gradient descent optimizer with batch size of 100, epoch of 30, learning rate of 0.001, and keep probability of 0.8. The reasons for choosing these options or parameters are described below.

* Optimizer: There are multiple optimizers to choose from, for example Adam, SGD+Nesterov, and momentum. Since Adam is often recommended as the default algorithm to use, we adopt the AdamOptimizer for our gradient descent algorithm.

* Batch size: Since we can't feed the entire data set into the optimizer at once. The training data are sliced into small batches to feed into the optimizer. As stochastic gradient descent method is used, large batch size will lead to more ignored data sets. On the other, the total optimization time increases significantly if the batch size is too small. The batch size of 100 is chosen as a tradeoff between optimization time and prediction accuracy.

* Epoch: As the available training data set is limited, it might not be sufficient to drive the training parameters to the desired value. To overcome the lack of training data, we can feed the training data into the optimizer multiple times. In each epoch, the training data is shuffled to avoid overfitting. When the epoch number is too large, we will notice that the validaction accuracy starts decline. This is a sign that the model is overfitted to the training data. The epoch number 30 is decided by observing when the model starts to overfit.

* Learning rate: The learning rate decides the actual stepsize taken for optimization. Small learning rate leads to very slow convergence rate, while large learning rate causes overshoting the desired value. The learning rate 0.001 is determined as a balance between convergence time and training accuracy.

* Keep-prob: During the training process, it can be observed that sometimes the training accuracy is close to 100% while the validation accuracy is very low. This is a sign that the CNN model is overfitted to the training data. To avoid overfitting, multiple techniques such as dropout and regularization can be leaveraged. Here we choose dropout with keep probability of 0.8 to reach a balance between overfitting and underfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.942
* test set accuracy of 0.933

The approach used for finding an appropriate CNN structure is an iterative approach.
* The first architecture tried is the LeNet CNN model. The reason is that similar problem of classifying digits in images is considered. Conversion to grayscale image and normalization of pixel values were performed to gain faster and more accurate result. After tuning different hyperparameters, the final validation accuracy was about 0.895. 
* Since the distribution of training data is not uniform across all traffic signs. We find that the prediction accuracy for certain traffic signs are relatively low. Data augmentation approach were taken to augment the traffic sign training data with insufficient data set. The validataion accuracy increased to 0.910.
* It is further observed that Lenet has 10 outputs while traffic sign CNN has 43 outputs. Thus, the current number of features might be insufficient to reach a good final prediction. The last several hidden layers' neurons were increase from 400x120x43 to 800x200x43. In addition to double the depth of first two convolution layers. With the increase number of neurons and some tuning, the validation accuracy increased to around 0.923.
* With the improved CNN model, the validation accuracy plateaus at around 0.92. A further investigation discovered that the training accuracy for CNN is already 0.998. Thus we reach the region of overfitting, with not hope for further improvement by simple tunings. To avoid overfitting, a dropout layer is incorperated to make the current model more fault tolerant. After tuning the keep probability for the dropout layer, the CNN model is able to produce validation accuracy of around 0.945.
* The most important design choices are: augmenting training data, increasing neurons in hidden layers to generate more useful features, and adding dropout layer to address overfitting.

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Six German traffic signs are seleted from the website. These traffic signs are then intentionally blurred, tilted, and dimmed to make it difficult to classify. The correct label for reference are found from the signnames.csv. These testing images are shown below.
![web data][image10]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 11 Right-of-way  		| 11 Right-of-way   							| 
| 9 No passing     		| 9 No passing							        |
| 18 General caution	| 18 General caution		    				|
| 23 Slippery road      | 23 Slippery road      		 				|
| 25 Road work          | 25 Road work     						    	|
| 1 Speed limit (30km/h)| 1 Speed limit (30km/h)                        |


The model was able to correctly predict all 6 out of 6 traffic signs, which gives an accuracy of 100%. Since the sample size is small, it is very close to our test set accuracy of 0.9333. After going back into the testing data set, we found that there are multiple very dark or blurred test images. This might be the reason that our samples from the website is predicted better.

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

To look at how certain these predictions are, a softmax function is applied to the CNN outputs. The top five probabilities are given below.

| Image           | Correct Label            |  Top 5   Predictions      | Top 5 Probability              | 
|:---------------:|:------------------------:|:------------------------:|:------------------------------:|
|   Right-of-way  | 11                       | [ 11, 30, 20]     |[  1.000,   4.023e-12,   2.66e-15] |
|   No passing    | 9                        | [ 9, 16, 17 ]     |[  1.000,   7.233e-07,   5.062e-09] |
|   General caution  | 18                    | [ 18, 26, 27]     |[  1.000,   6.270e-06,   2.497e-07] |
|   Slippery road  | 23                      | [ 23, 19, 21]     |[  9.698e-01,   3.016e-02,   1.148e-06] |
|   Road work  | 25                          | [ 25, 30, 21]     |[  9.125e-01,   8.735e-02,   4.609e-05] |
|   Speed limit (30km/h)  | 1                | [ 1, 40,  0]      |[  1.000,   4.023e-12,   2.66e-15] |

Most images are correctly identified with more than 99.9% confidence. The classifier get a little confused between 1) Slippery road (97%) and dangerous curve to left (3%) 2) Road work (92%) and be aware of ice and snow (8%). More training data can be added to these categories to further improve the prediction accuracy.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

To further understand what is happening under the hood. We took a "Right-of-way" traffic sign and feed it into the CNN model. The first layer output (after convolution) is visualized below.
![hidden1][image6]
The second layer output (after convolution) is also visualized below.
![hidden2][image7]

It can be observed that the first layer output pick up multiple small features in the image, for example inner boundary, outer boundary, inisde feature, shape of sign, and so on. The second layer took these small features and combine them into higher level features. Since there are 43 difference classes to classify, the difference between them is identified with these high dimensional features.
