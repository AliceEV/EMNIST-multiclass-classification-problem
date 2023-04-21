# EMNIST-multiclass-classification-problem

Two neural networks (MLP and CNN) for the balanced split of the EMNIST dataset

Link to the .ipynb file: https://colab.research.google.com/drive/1FBmaKYc1K71tmIIlUgp4idK4Oe02aLdY?usp=sharing


Project Goal:

This project aims to build a neural network model to solve the EMNIST classification problem. As deep neural networks are widely used in various domains, such as computer vision, natural language processing and medical image analysis, how to create a proper neural network to achieve a specific task has become very important. 


Dataset Introduction:

In the project, we use the EMNIST  (Extended MNIST) dataset (https://www.kaggle.com/datasets/crawford/emnistLinks to an external site.), which is a set of handwritten character digits derived from the NIST Special Database 19 and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset.

Due to the EMNIST dataset including 6 different splits, we selected the “Balanced” dataset, which addressed the balance issues in the “ByClass” and “ByMerge” datasets. It is derived from the “ByMerge” dataset to reduce misclassification errors due to capital and lower-case letters and also has an equal number of samples per class. The “Balanced” dataset information is as follows:

Train: 112,800
Test: 18,800
Total: 131,600
Classes: 47 (balanced)
If we visualize the EMNIST images, they look as follows:
 image.png


Project Introduction:

Write the python code to build various neural networks and compare the performance of different network models. 

In this project, you need to implement two types of neural networks; one is the multilayer perceptron (MLP) networks with at least three hidden layers (i.e., neural networks with only fully-connected layers); the other is the Convolutional Neural Networks (CNNs) with at least two convolutional layers.  For each network model, you need to consider to multiple parameters to obtain the best performance. You can use various techniques (you have learnt) to overcome overfitting, underfitting, unstable gradient, etc. You need to utilize and explore the following techniques to train your neural network models:

Adaptive Learning Rate, e.g., learning rate schedulers (explore at least two learning rate scheduling methods)

Activation function, e.g., ReLU, Leaky ReLU, ELU, etc. (explore at least three activation functions)

Optimizers, e.g., SGD, ADAM, RMSprop, ASGD, AdaGrad, etc. (explore at least three optimizers)

Batch Normalization (explore two options: with Batch normalization, without Batch normalization)

L1 & L2 regularization (explore three options: without L1&L2 regularization, with L1 regularization, with L2 regularization)

Dropout (explore two options: with or without Dropout)
