import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import mean, std
from sklearn.model_selection import KFold
from src.load import load_data_from_file
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.layers.core import Flatten, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

def normalize(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    
    train_norm /= 255
    test_norm /= 255
    
    return train_norm, test_norm

def basic_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(62, activation='softmax'))
    
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def complex_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3,3), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(62, activation='softmax'))
    
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# evaluate a model using k-fold cross-validation
def cross_validate(x, y, model, n_folds=3):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for x_train_i, x_test_i in kfold.split(x):
        # select rows for train and test
        x_train, y_train, x_test, y_test = x[x_train_i], y[x_train_i], x[x_test_i], y[x_test_i]
        # fit model
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
        # evaluate model
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 2, 1)
        plt.title('Cross Entropy Loss Train')
        plt.ylim([0, 1.0])
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.subplot(2, 2, 2)
        plt.title('Cross Entropy Loss Test')
        plt.ylim([0, 1.0])
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 2, 3)
        plt.title('Classification Accuracy Train')
        plt.ylim([0.7, 1.0])
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.subplot(2, 2, 4)
        plt.title('Classification Accuracy Test')
        plt.ylim([0.7, 1.0])
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()

def evaluate_kfold(x_train, y_train):
    scores, histories = cross_validate(x_train, y_train, complex_model())
    
    summarize_diagnostics(histories)
    summarize_performance(scores)
    
def build(x_train, y_train):
    model = complex_model()
    history = model.fit(x_train, y_train)
    print(history.history['accuracy'])

if __name__ == "__main__":
    t = time.time
    x_train_raw, y_train, x_test_raw, y_test = load_data_from_file()
    x_train, x_test = normalize(x_train_raw, x_test_raw)
    
    #build(x_train, y_train)
    evaluate_kfold(x_train, y_train)
    print("Time:", time.time - t)