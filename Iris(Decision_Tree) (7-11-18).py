import pandas as pd
import numpy as np
from sklearn import tree
import imp
from matplotlib.pyplot import *
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
#print (ds)
#print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())
#df.apply(pd.Series.value_counts)
#print(pd.isnull(df))
x = df.iloc[:,0:4].values
y = df.iloc[:,4].values

#split dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5)
decision_tree = tree.DecisionTreeClassifier()

decision_tree.fit(x_train,y_train)
predictions = decision_tree.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))