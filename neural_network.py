import numpy as np
import tensorflow as tf
from utils import retrieve_data, show_weights, DATA_SIZE_X, DATA_SIZE_Y, NUM_EPOCHS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def main():
    (train_x, train_y), (test_x, test_y) = retrieve_data()

    start_time = time.time()

    # Build and evaluate all three models for the data
    slnn = single_layer_neural_network(train_x, train_y, test_x, test_y)
    mlff = multi_layer_feed_forward(train_x, train_y, test_x, test_y)
    cnn = convolutional_neural_network(train_x, train_y, test_x, test_y)

    # Print the metrics of each model on the data
    print(f"Achieved {slnn}% Accuracy with Single-Layer Neural Network.")
    print(f"Achieved {mlff[1] * 100.0}% Accuracy with Multi-Layer Feed Forward Neural Network.")
    print(f"Achieved {cnn[1] * 100.0}% Accuracy with Convolutional Neural Network.")
    print(f"Finished building and training all Neural Networks in {time.time() - start_time}s.")


def multi_layer_feed_forward(train_x, train_y, test_x, test_y):
    """
        Method to build, train and evaluate a multi-layer feed forward neural network.
    """

    # Build the model, one layer to flatten the inputs and then two Dense feed-forward layers with ReLU activation. 
    model = Sequential([
        Flatten(input_shape=(DATA_SIZE_X, DATA_SIZE_Y)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model with the ada optimizer, using categorical cross entropy loss, and accuracy as the metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model to the training data using a preset number of epochs
    model.fit(train_x, train_y, epochs=NUM_EPOCHS)

    # Return the results of evaluating the model on the test data
    return model.evaluate(test_x, test_y)

def convolutional_neural_network(train_x, train_y, test_x, test_y):
    """
        Method to build, train and evaluate a Convolutional Neural Network on the dataset.
    """

    # Reshape inputs
    train_x = train_x.reshape(train_x.shape[0], DATA_SIZE_X, DATA_SIZE_Y, 1)
    test_x = test_x.reshape(test_x.shape[0], DATA_SIZE_X, DATA_SIZE_Y, 1)

    # Normalize the inputs
    train_x = tf.keras.utils.normalize(train_x)
    test_x = tf.keras.utils.normalize(test_x)

    # Create the model for the CNN, this uses two alternating layers of Convolution and Pooling followed by a single Dense layer
    model = Sequential([
        Conv2D(32, (3, 3), strides=1, input_shape=(DATA_SIZE_X, DATA_SIZE_Y, 1)),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(64, (3, 3), strides=1),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model using the adam optimizer, categorical cross entropy loss and using accuracy as the metric.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model with a preset number of epochs
    model.fit(train_x, train_y, epochs=NUM_EPOCHS)

    # Return the results of evaluating the model on the test data
    return model.evaluate(test_x, test_y)

def single_layer_neural_network(train_x, train_y, test_x, test_y):
    """
        Method to build, train and evaluate a single-layer neural network.
        Makes use of gradient descent to update weights incrementaly as well as the bias.
        Furthermore, shows that to within a certain degree the universal approximation theory holds. 
        The function to determine digits can be represented in only a single layer neural network.
    """

    # Reshape the input
    images = train_x[0:train_x.shape[0]].reshape(train_x.shape[0],28*28)
    images = images.T

    # Reshape the test inputs, each image will be a single vector the length of the dimensions multiplied together
    images_test = test_x.reshape(test_x.shape[0], 28*28)
    images_test = images_test.T

    # Step-size
    alpha = 0.01

    # Weight matrix
    W = np.zeros((10, 28*28))

    # Bias matrix
    B = np.zeros((10,1))

    # Performs 1000 iterations of gradient descent
    for i in tqdm(range(1000)):
        W -= alpha * ((1/images.shape[1])*(((W.dot(images) + B) - train_y.T).dot(images.T))) # Computes update to W based on Gradient of loss wrt the weights
        B -= alpha * np.sum(((W.dot(images) + B) - train_y.T), axis=1, keepdims=True) * (np.divide(1,images.shape[1])) # Computes update to B using Gradient of the loss wrt bias which is just the sum of the error
        
        
    # Compute the percentage of images from the test set classified correctly from the results on the test set.
    test_result = (W.dot(images_test) + B)
    count = 0

    for i in range(0, test_result.shape[1]):
        if(np.argmax(test_result[:,i]) == np.argmax(test_y[i])):
            count += 1
    show_weights(W)

    # Return the percentage of the images classified correctly
    return count/test_result.shape[1]*100


if __name__ =="__main__":
    main()