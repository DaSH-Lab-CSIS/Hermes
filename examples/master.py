# These three lines are required.
import sys
sys.path.append('../hermes/')
import HermesMaster


# # External libraries for your use
# import tensorflow as tf
# import tensorflow.keras as keras
# import numpy as np

# # Initialize the model and dataset
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(784, 1)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Categories the labels
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# # Convert data type to float32 for compatibility with the model
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# # Normalize the pixel values to the range [0, 1]
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# y_train = y_train.reshape(-1)

# y_train = y_train[:, np.newaxis]
# x_train = x_train.reshape(-1, 28 * 28) # Flatten the training input images

# y_test = y_test.reshape(-1)

# y_test = y_test[:, np.newaxis]
# x_test = x_test.reshape(-1, 28 * 28) # Flatten the testing input images


# These two lines are necessary to initialize and begin the framework.
hermes = HermesMaster.HermesMaster(model, x_train, y_train, x_test, y_test, 25, 0.1, 10)
hermes.start()

