import keras
import keras_metrics
from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, Reshape, MaxPooling1D, Dropout, concatenate, Activation
import numpy as np
from sklearn.preprocessing import StandardScaler

from MaLeNeuralNetwork import MaLeNeuralNetwork


class MaLeSimple(MaLeNeuralNetwork):
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, val):
        self.__model = val

    def __init__(self, xshape, neuron, optimizer):
        super(MaLeNeuralNetwork, self).__init__()

        inputs = Input(shape=(xshape[1],))
        inminmax = Input(shape=(4,))
        inre = Reshape((xshape[1], 1), input_shape=(xshape[1],))(inputs)
        flat = Flatten(name='flatten')(inre)

        concat = concatenate([flat, inminmax], axis=1)

        den1 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(concat)
        den2 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(den1)
        den3 = Dense(2)(den2)
        pred = Activation('linear', name='linear')(den3)

        self.model = Model(inputs=[inputs, inminmax], outputs=pred)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer,
                           metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                                    'cosine_proximity'])

    def input_scale(self, x_train, x_test):
        min = np.min(x_train, axis=1)
        max = np.max(x_train, axis=1)
        argmax = np.argmax(x_train, axis=1)
        argmin = np.argmin(x_train, axis=1)
        trainmm = np.array((min, max, argmax, argmin)).T
        min = np.min(x_test, axis=1)
        max = np.max(x_test, axis=1)
        argmax = np.argmax(x_test, axis=1)
        argmin = np.argmin(x_test, axis=1)
        testmm = np.array((min, max, argmax, argmin)).T
        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return [x_train, trainmm], [x_test, testmm]
