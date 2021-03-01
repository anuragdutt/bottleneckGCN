import numpy as np
import os, sys
import h5py
from optparse import OptionParser 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import svm
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import f1_score, accuracy_score

def meanNorm(X):
	dfx = []
	col = X.columns.tolist()
	
	for ix,r in X.iterrows():
		r = pd.Series(r)
		val = r.values.tolist()
		s = sum(val)

		rw = [i/s for i in val]

		dfx.append(rw)

	X = pd.DataFrame(dfx, columns = col)
	return X

def createModel(modelname="svm"):
	from sklearn.linear_model import SGDClassifier
	global model

	# SVM classifier trained online with stochastic gradient descent
	model = SGDClassifier(loss="hinge", penalty="l2")
	if modelname == "log":
		# Logistic Regresion classifier trained online with stochastic gradient descent
		model = SGDClassifier(loss="log", penalty="l2")  
		print("Using Logistic Regression...")
	if modelname == "non-linear-svm":
		model = svm.SVC(kernel='rbf')
		print("Using RBF kernel SVM...")
	else:
		print("Using Hinge Loss SVM...")

		
def train( X_train, y_train, nClasses=2, batchSize=256):
	best_score=0
	X_count = X_train.shape[0]
	batchCount= int(X_count // batchSize)

	j=0
	shuffledRange = range(X_count)
	shuffledX = X_train[shuffledRange,]
	shuffledY = [y_train[i] for i in shuffledRange]

	global model
	for i in range(0, batchCount):  # Iterate over "mini-batches" of 1000 samples each
		j+=1
		y_train_batch = shuffledY[i*batchSize :(i +1)* batchSize]
		X_train_batch = shuffledX[i*batchSize :(i +1)* batchSize,]
		# vectorizer.fit_transform(train_data[i:i + batchSize])
		# Update the classifier with documents in the current mini-batch
		model.partial_fit(X_train_batch, y_train_batch, classes=range(nClasses))

def test(X_test, y_test):
	global model
	score = model.score(X_test, y_test)
	return score



if __name__ == "__main__":
	global model
	model = ""
	nclasses = 6
	batchsize = 256
	epochs = 100
	model_path = "../models/svm.checkpoint"
	n0 = pd.read_csv("../data/tracing-data/hotel-reservation/0_frontend.csv")
	n0['label'] = 0
	n1 = pd.read_csv("../data/tracing-data/hotel-reservation/1_search.csv")
	n1['label'] = 1
	n2 = pd.read_csv("../data/tracing-data/hotel-reservation/2_geo.csv")
	n2['label'] = 2
	n3 = pd.read_csv("../data/tracing-data/hotel-reservation/3_rate.csv")
	n3['label'] = 3
	n4 = pd.read_csv("../data/tracing-data/hotel-reservation/4_profile.csv")
	n4['label'] = 4
	n5 = pd.read_csv("../data/tracing-data/hotel-reservation/5_locale.csv")
	n5['label'] = 5


	dat = pd.concat([n0, n1, n2, n3, n4, n5])
	print(dat.shape)
	# dat = dat.iloc[0:10000, :]

	y = dat['label']
	X = dat[["0_frontend", "1_search", "2_geo", "3_rate", "4_profile", "5_locale"]]

	X = meanNorm(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	createModel()
	
	for i in range(epochs):
		train(X_train.to_numpy(), y_train.to_numpy(), nclasses, batchsize)
		dump(model, model_path)
		
	y_pred = model.predict(X_test)

	print(f1_score(y_pred, y_test))
	print(accuracy_score(y_pred, y_test))