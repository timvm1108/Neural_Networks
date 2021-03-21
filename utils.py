import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constants used to determine the size of the input data and the number of epochs to use
DATA_SIZE_X = 28
DATA_SIZE_Y = 28
NUM_EPOCHS = 10

def retrieve_data():
    """
        Method to retrieve and normalize the MNIST data.
    """

    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()

    # Each pixel will be a value from 0 to 255, this will server to normalize the pixel values to a range 0 to 1
    mnist_train_x = mnist_train_x / 255.0
    mnist_test_x = mnist_test_x / 255.0

    # Turn the outputs into a one-hot vector of each picture
    mnist_train_y = tf.keras.utils.to_categorical(mnist_train_y)
    mnist_test_y = tf.keras.utils.to_categorical(mnist_test_y)

    return (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y)

def show_weights(W, f='weights.png'):
    """
        Method to output the final weights for each digit of the single-layer neural network.
    """

    # Create a figure to output the weights of the single layer network
    import matplotlib
    matplotlib.use('Agg')
    plt.figure(figsize=(14,8))

    # For each set of final weights, output the weights as an image
    # Lighter pixels indicate that the weight is very high for that location, implying that the neural network believes that location is often associated with that number
    for i in range(W.shape[0]):
        plt.subplot(2, 5, i+1, title=str(i))
        individual_weight = W[i,:].reshape(DATA_SIZE_X, DATA_SIZE_Y)
        plt.imshow(individual_weight, cmap='gray')

    # Output the weights into a file
    plt.savefig(f)