import numpy as numpy
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

print("Reading Data..")
data = pd.read_csv("train.csv").as_matrix()
print("Reading Data Complete..")

# Classifiers
clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()

# Training Data

xtrain = data[0:30000,1:]
ytrain = data[0:30000,0]

# Testing Data
xtest = data[30000:,1:]
ytest = data[30000:,0]

# Training
print("Training Decision Tree Classifier..")
clf1.fit(xtrain,ytrain)
print("Training Random Forest Classifier..")
clf2.fit(xtrain,ytrain)
print("Training Naive Bayes Classifier..")
clf3.fit(xtrain,ytrain)
print("Training Complete..")

# Calculate Accuracy
print("Predicting Decision Tree Classifier..")
ypred1 = clf1.predict(xtest)
print("Predicting Random Forest Classifier..")
ypred2 = clf2.predict(xtest)
print("Predicting Naive Bayes Classifier..")
ypred3 = clf3.predict(xtest)
print("Prediction Complete..")

cnt = 0
for i in range(len(ytest)):
	if(ytest[i]==ypred1[i]):
		cnt += 1
print("Accuracy of Decision Tree ",cnt*100/len(ytest),"%")

cnt = 0
for i in range(len(ytest)):
	if(ytest[i]==ypred2[i]):
		cnt += 1
print("Accuracy of Random Forest ",cnt*100/len(ytest),"%")

cnt = 0
for i in range(len(ytest)):
	if(ytest[i]==ypred3[i]):
		cnt += 1
print("Accuracy of Naive Bayes ",cnt*100/len(ytest),"%")

while(True):
	print("Enter index between 1 to 12000: ",end="")
	x = int(input())
	d = xtest[x-1]
	d.shape = (28,28)
	pt.imshow(255-d, cmap='gray')
	print("Decision Tree Classifier Predicts ",clf1.predict([xtest[x-1]]))
	print("Random Forest Classifier Predicts ",clf2.predict([xtest[x-1]]))
	print("Naive Bayes Classifier Predicts ",clf3.predict([xtest[x-1]]))
	pt.show()