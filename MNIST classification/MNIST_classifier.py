import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

def import_data():
    # fix random seed for reproducibility
    seed = 19
    numpy.random.seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be samples  x width x height x channels
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # normalize input
    X_train = X_train / 255
    X_test = X_test / 255

    # one-hot-encoding of label
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return X_train, X_test, y_train, y_test, num_classes


def baseline_model():
    #initialize model
    model = Sequential()
    #add convolution layer
    model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    #add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #add dropout layer
    model.add(Dropout(0.2))
    #flatten layer to converts the 2D matrix data to a vector
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(64, activation='relu'))
    #output layer
    model.add(Dense(num_classes, activation='softmax'))
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def larger_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def diff_optimizer_model():
    #initialize model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    #optimizer changed
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def batch_normalized_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #add batch_normalization layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    #prepare data
    X_train, X_test, y_train, y_test, num_classes = import_data()
    #build the model
    model = baseline_model()
    #fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    #evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Error Rate: {}}".format((100-scores[1]*100)))