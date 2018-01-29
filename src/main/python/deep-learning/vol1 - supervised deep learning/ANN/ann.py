

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
from keras.layers import Dropout
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

# Part 2 - Making the ANN

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# A good starting place for choosing the number of nodes (output_dim=) in the first hidden laqter is the average between
# the number of input nodes and the number of output nodes--otherwise its sort of a guessing game.
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p=0.1))

# Adding a second hidden layer
# A good starting place for choosing the number of nodes (output_dim=) in the first hidden layer is the average between
# the number of input nodes and the number of output nodes--otherwise its sort of a guessing game.
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print((cm[0, 0] + cm[1, 1]) / 2000)

# Part 3.5 - Predicting a single new observation
"""
Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

# using double brackets so that the data is a horizontal row of the arraqy rather than virtical column

# *Note: that the Geography is a categorical variable. this means that is was encoded above. what we need to do is figure
# out what the encoding is for France and enter it here. we do that by finding a row in the data ser wher the
# Geography is France, and choosing its encoding -- 0, 0 in this case.

# *Note: the gender is also categorical--therefore encoded, so we use the same process as above to find the encoded
# version of the Gender, in this case its 1.

# note that the ANN was trained on scaled data. therfore we must scale the prediction input to match the same scale.
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

new_prediction = (new_prediction > 0.5)

# Part 4 - Evaluating, improving and Tuning the ANN

# Evaluating the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
#
#
# def build_classifier():
#
#     c = Sequential()
#     c.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
#     c.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
#     c.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
#     c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return c
#
#
# classifier = KerasClassifier(build_fn=build_classifier(), batch_size=10, nb_epoch=100)
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

# Test the accuracy and variance of all iterations within the K-Fold
# mean = accuracies.mean()
# variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce over-fitting if needed
# this is done by adding classifier.add(Dropout(p=0.1))to each hidden layer

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    c = Sequential()
    c.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    c.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    c.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    c.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return c


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


