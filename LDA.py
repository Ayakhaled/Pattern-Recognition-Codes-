import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv
from sklearn.metrics import accuracy_score
import random
from sklearn import preprocessing

#Display data
filename = "letter-recognition.data"
raw_data = open(filename, 'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x)

#split data and labels
X_data = data[:,1:]
Y_data = data[:,0]

#splitting data into train and test with percentage
perc = 0;

for i in range(0, 3):
    if i == 0:
        perc = 0.75
    elif i == 1:
        perc = 0.6
    elif i == 2:
        perc = 0.5

    no_of_points = X_data.shape[0]
    total_perc = perc*no_of_points

    #train data
    Y_train = Y_data[:total_perc]
    X_train = X_data[:total_perc]
    X_train = X_train.astype('float')


    #test data
    Y_test = Y_data[(total_perc+1):]
    X_test = X_data[(total_perc+1):]
    X_test = X_test.astype('float')

    #train classifier
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)

    #prediction
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print("accuracy: ")
    print(accuracy)
