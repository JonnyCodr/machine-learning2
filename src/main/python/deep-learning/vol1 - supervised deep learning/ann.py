

# Installing Theano - computations library (based on numpy) runs on gpu

# Installing Tensorflow - computations library and runs on either CPU or GPU

# Installing keras - wrapper for Theano and Tensorflow


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the data set
dataset = pd.read_csv('./Churn_Modeling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# adds a dummy variable so that the encoding for country isn't misconstrued as 1 being greater than 2
#  or greater than 0 (or any other variation)
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# this removes one of the dummy variable columns as to not fall into the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# # Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# A good starting place for choosing the number of nodes (output_dim=) in the first hidden laqter is the average between
# the number of input nodes and the number of output nodes--otherwise its sort of a guessing game.
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding a second hidden layer
# A good starting place for choosing the number of nodes (output_dim=) in the first hidden layer is the average between
# the number of input nodes and the number of output nodes--otherwise its sort of a guessing game.
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print((cm[0, 0] + cm[1, 1]) / 2000)


