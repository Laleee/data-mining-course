#!/usr/bin/python3

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

def preprocess(file_1, file_2):

	# Load files
	df_1 = pd.read_csv(file_1, index_col = False)
	df_2 = pd.read_csv(file_2, index_col = False)

	# Transpose and remove column label
	df_1 = df_1.transpose()
	df_1.drop(df_1.index[0], inplace = True)
	df_2 = df_2.transpose()
	df_2.drop(df_2.index[0], inplace = True)

	# Set class for each file
	df_1['class'] = 1
	df_2['class'] = 2

	# Concat into one matrix for further preprocessing and classification
	df = pd.concat([df_1, df_2])

	print("Table dimensions: " + str(df.shape))
	print("\tTable 1 dimensions: " + str(df_1.shape))
	print("\tTable 2 dimensions: " + str(df_2.shape))

	# Remove rows with only 0 values, don't add class value into sum
	df = df.loc[:, (df != 0).any(axis=0)]

	print("Table dimensions after removing 0s: " + str(df.shape))

	df.to_csv("./data/preprocessed.csv", index = False)

	return df

def predict(clf, x_train, x_test, y_train, y_test):
	class_train_predicted = clf.predict(x_train)
	class_test_predicted  = clf.predict(x_test)

	train_acc = clf.score(x_train, y_train)
	test_acc  = clf.score(x_test, y_test)

	print("Train acc: " + str(train_acc))
	print("Test  acc: " + str(test_acc))

	train_report = classification_report(y_train, class_train_predicted)
	test_report  = classification_report(y_test, class_test_predicted)

	print("Train report:\n" + str(train_report))
	print("Test  report:\n" + str(test_report))

	train_conf = confusion_matrix(y_train, class_train_predicted)
	test_conf  = confusion_matrix(y_test, class_test_predicted)

	print("Confussion matrix [train]:\n" + str(train_conf))
	print("Confussion matrix [test]:\n"  + str(test_conf))

def knn(x_train, x_test, y_train, y_test, neighbors):
	clf = KNeighborsClassifier(n_neighbors = neighbors, weights = 'uniform')
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)
	
def dtc(x_train, x_test, y_train, y_test, max_depth):
	clf = DecisionTreeClassifier(max_depth = max_depth)
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)

def naive_bayes(x_train, x_test, y_train, y_test):
	clf = MultinomialNB()
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)

def svm_classifier(x_train, x_test, y_train, y_test, kernel):
	clf = svm.SVC(kernel = kernel)
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)

def mplc(x_train, x_test, y_train, y_test, solver):
	clf = MLPClassifier(solver = solver)
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)

def sgdc(x_train, x_test, y_train, y_test):
	clf = SGDClassifier(max_iter = 1000, loss = 'hinge', penalty = 'l2')
	clf.fit(x_train, y_train.values.ravel())

	predict(clf, x_train, x_test, y_train, y_test)

def main():
	file_class_1 = "./data/024_Single-cell_RNA-seq_of_six_thousand _purified_CD3+_T_cells_from _human_primary_TNBCs_csv.csv";
	file_class_2 = "./data/025_Single-cell_RNA-seq_of_six_thousand _purified_CD3+_T_cells_from _human_primary_TNBCs_csv.csv";

	df = preprocess(file_class_1, file_class_2)
	# df.to_csv("./data/preprocessed.csv", index = False)

	#df = pd.read_csv("./data/preprocessed.csv", index_col = False, nrows = 500)

	# Split data into test and train sets
	#data    = df.loc[:, df.columns != 'class']
	#classes = df[['class']]

	#x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size = 0.3, stratify = classes)
	
	# knn(x_train, x_test, y_train, y_test, 3)
	# dtc(x_train, x_test, y_train, y_test, 5)
	# naive_bayes(x_train, x_test, y_train, y_test)
	# svm_classifier(x_train, x_test, y_train, y_test, 'poly')
	# mplc(x_train, x_test, y_train, y_test, 'adam')
	#sgdc(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
	main()