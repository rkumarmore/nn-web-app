import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow

class Regression:

    def __init__(self):
        self.model = tensorflow.keras.models.Sequential()
        self.model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(11,)))
        self.model.add(tensorflow.keras.layers.Dense(1))
        self.model.compile(optimizer='sgd', loss='mse')

    def train(self, filename, target_variable):
        data = pd.read_csv(filename)
        parameters = data.drop(target_variable, axis=1).values
        target_variable_data = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(parameters, target_variable_data, test_size=.30, random_state=1)
        return self.model.fit(X_train, y_train, epochs=100, validation_split=0.35)

    def pickle(self):
        self.model.save('pickled_regression_model')
        return True