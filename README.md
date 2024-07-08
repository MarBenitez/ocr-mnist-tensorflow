# Handwritten Digit Recognition with MNIST

This repository contains a complete project for recognizing handwritten digits using a Convolutional Neural Network (CNN) and the MNIST dataset. The project includes data preprocessing, model training, and deployment using an interactive Gradio interface.

## Project Overview

The goal of this project is to build a machine learning model capable of recognizing handwritten digits (0-9) with high accuracy. The MNIST dataset is used for training and testing the model. The project demonstrates the following steps:

1. **Data Preprocessing**: Loading and preparing the MNIST dataset.
2. **Model Building**: Constructing a CNN model using TensorFlow/Keras.
3. **Model Training**: Training the model and evaluating its performance.
4. **Model Deployment**: Creating an interactive web interface using Gradio to allow users to draw digits and get predictions.

## Files in the Repository

- `ocr_mnist.ipynb`: The Jupyter Notebook containing the complete workflow, from data preprocessing to model deployment.
- `model2.h5`: The pre-trained Keras model file.
- `README.md`: Project documentation and overview.

## Project Workflow

### 1. Data Preprocessing

The MNIST dataset is loaded and preprocessed. This includes normalizing the pixel values and reshaping the data to fit the input requirements of the CNN model.

### 2. Model Building

A Convolutional Neural Network (CNN) is built using TensorFlow/Keras. The architecture includes:
- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Dense layers for classification.

### 3. Model Training

The model is trained using the training dataset, and its performance is evaluated using the validation dataset. Key metrics such as accuracy and loss are tracked during training.

### 4. Model Deployment

An interactive interface is created using Gradio, allowing users to draw digits on a sketchpad and get real-time predictions. The interface leverages the pre-trained model to classify the drawn digits.


![model-scheme](https://github.com/MarBenitez/ocr-mnist-tensorflow/blob/main/mnist_model.h5.png)
