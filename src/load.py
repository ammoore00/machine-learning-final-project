import numpy as np

from os import path
from tensorflow.python.keras.utils.np_utils import to_categorical
from extra_keras_datasets import emnist

#EMNIST-Letters
# Extended MNIST (EMNIST) contains digits as well as uppercase and lowercase handwritten letters.
# EMNIST-Letters contains 145.600 characters across 26 balanced classes (letters only).
# 
# Imported to keras by Christian Versloot at https://github.com/christianversloot/extra_keras_datasets
# Provided under MIT liscense
(input_train, target_train), (input_test, target_test) = emnist.load_data(type='byclass')

# Saves data to avoid needing to repeatedly load the dataset which can take minutes every time
# Reshapes 3D inputs to 2d to write to a file, original dimensions saved for later
np.savetxt("dataset/sizes.txt", (input_train.shape[2], input_test.shape[2]))

if not path.exists("dataset/input_train.txt"):
    np.savetxt("dataset/input_train.txt", input_train.reshape(input_train.shape[0], -1))
if not path.exists("dataset/target_train.txt"):
    np.savetxt("dataset/target_train.txt", target_train)
if not path.exists("dataset/input_test.txt"):
    np.savetxt("dataset/input_test.txt", input_test.reshape(input_test.shape[0], -1))
if not path.exists("dataset/target_test.txt"):
    np.savetxt("dataset/target_test.txt", target_test)
    
def load_data_from_file():
    original_dims = np.loadtxt("dataset/sizes.txt")
    
    input_train_reshape = np.loadtxt("dataset/input_train.txt")
    target_train = np.loadtxt("dataset/target_train.txt")
    input_test_reshape = np.loadtxt("dataset/input_test.txt")
    target_test = np.loadtxt("dataset/target_test.txt")
    
    #Reshapes arrays stored in 2d back to 3d, and adds a 4th dimension of size 1 for CNN
    input_train = input_train_reshape.reshape((input_train_reshape.shape[0],
                                                input_train_reshape.shape[1] // int(original_dims[0]),
                                                int(original_dims[0]), 1))
    input_test = input_test_reshape.reshape((input_test_reshape.shape[0],
                                                input_test_reshape.shape[1] // int(original_dims[0]),
                                                int(original_dims[0]), 1))
    
    # Converts targets from a single number to one hot for 26 categories
    target_train = [y - 1 for y in target_train]
    target_test = [y - 1 for y in target_test]
    
    target_train_cat = to_categorical(target_train, num_classes=62)
    target_test_cat = to_categorical(target_test, num_classes=62)
    
    return input_train, target_train_cat, input_test, target_test_cat