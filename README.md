# COMP 4107 Assignment #3
## Due: 19th November 2017 at 11:55 pm

### Objective

The primary objective of this assignment is to have the student develop self-organizing map and Hopfield neural network implementations.

### Submission

Submission should be to cuLearn. You are to upload a zip file containing python files and a single PDF file containing experimental results which document your answers to the questions attempted. Submission is in pairs. One partner should upload a zip file, the other a readme.txt file containing the names and student numbers of both partners.

### Description

You may use any and all functionality found in scikit-learn and tensorflow.

#### Question 1

(a) Using the scikit-learn utilities to load the MNIST data, implement a Hopfield network that can classify the image data for a subset of the handwritten digits. Subsample the data to only include images of '1' and '5'. Here, correct classification means that if we present an image of a '1' an image of a '1' will be recovered; however, it may not be the original image owing to the degenerate property of this type of network. You are expected to document classification accuracy as a function of the number of images used to train the network. Remember, a Hopfield network can only store approximately 0.15N patterns with the "one shot" learning described in the lecture (slides 58-74).

(b) BONUS 10% Improvements to basic Hopfield training have been proposed (Storkey 1997). Implement this improvement and contrast the accuracy with part (a).

#### Question 2

We can use self organizing maps as a substitute for K-means.

In Assignment 2, Question 2, K-means was used to compute the number of hidden layer neurons to be used in an RBF network. Using a 2D self-organizing map compare the clusters when compared to K-means for the MNIST data. Sample the data to include only images of '1' and '5'. Use the scikit-learn utilities to load the data. You are expected to (a) document the dimensions of the SOM computed and the learning parameters used to generate it (b) provide 2D plots of the regions for '1' and '5' for both the SOM and K-means solutions. You may project your K-means data using SVD to 2 dimensions for display purposes.

#### Question 3 BONUS 20%

Using Principal Component Analysis (PCA) and scikit-learn face data compare the classification accuracy of faces when using this orthonormal basis as input to a feed forward neural network when compared to using the raw data as input features. You are expected to document (a) the size of your feed forward neural network in both cases and (b) the prediction accuracy of your neural networks using a K-fold analysis.
