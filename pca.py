import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.metrics import accuracy_score
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


filename = "letter-recognition.data"
raw_data = open(filename, 'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x)

#split data and labels
X_data = data[:,1:]
Y_data = data[:,0]

#train data
Y_train = Y_data[:15000]
X_train = X_data[:15000]
X_train = X_train.astype('float')


#test data
Y_test = Y_data[(15001):]
X_test = X_data[(15001):]
X_test = X_test.astype('float')

pca = PCA(n_components=16)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#train classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

#prediction
y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

print(accuracy*100)

	# X_data = X_data.astype('float')
	# transformed_data = pca.transform(X_data)
	# first_pc = pca.components_[0]
	# sec_pc = pca.components_[1]
	# for ii, jj in zip(transformed_data, X_data):
	# 	plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
	# 	plt.scatter(sec_pc[0]*ii[1], sec_pc[1]*ii[1], color="c")
	# 	plt.scatter(jj[0], jj[1], color="b")
	# plt.xlabel("bonus")
	# plt.ylabel("long-term incentive")
	# plt.show()
