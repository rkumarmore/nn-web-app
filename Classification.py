import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow

class Classification:

    def __init__(self):
        self.model = tensorflow.keras.models.Sequential()
        self.model.add(tensorflow.keras.layers.Flatten())
        self.model.add(tensorflow.keras.layers.Dense(128, activation="relu"))
        self.model.add(tensorflow.keras.layers.Dense(11, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="sgd")

    def train(self, filename, target_variable):
        data = pd.read_csv(filename)
        parameters = data.drop(target_variable, axis=1).values
        target_variable_data = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(parameters, target_variable_data, test_size=.30, random_state=1)
        y_train_cat = tensorflow.keras.utils.to_categorical(y_train, num_classes=11)
        y_test_cat = tensorflow.keras.utils.to_categorical(y_test, num_classes=11)
        return self.model.fit(x=X_train, y=y_train_cat, batch_size=32, epochs=50, validation_data=(X_test, y_test_cat))

    def pickle(self):
        self.model.save('pickled_classification_model')
        return True