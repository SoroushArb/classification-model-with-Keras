# Image Classification with Keras

## Project Description

This project involves building and training a neural network for image classification using the popular MNIST dataset. The MNIST database consists of handwritten digits and is commonly used for training various image processing systems. The goal is to compare the performance of conventional neural networks with convolutional neural networks, which will be explored in the next module.

## Functionality

1. **Download and Explore MNIST Dataset:**
   - Utilize the MNIST dataset containing 60,000 training images and 10,000 testing images of handwritten digits.

2. **Build a Neural Network:**
   - Flatten the images into one-dimensional vectors.
   - Normalize inputs and one-hot encode outputs.
   - Build a classification model with dense layers.

3. **Train and Test the Network:**
   - Train the model on the training dataset.
   - Evaluate the model's accuracy on the testing dataset.

4. **Save and Load the Model:**
   - Save the trained model as 'classification_model.h5'.
   - Load the pretrained model for future use.

## Usage

### Prerequisites

Make sure you have the required libraries installed:

```bash
pip install numpy pandas keras matplotlib
