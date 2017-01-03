import numpy as np
import pandas as pd
import re
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from os import system

#read train and test data
train = pd.read_csv("train.csv", header = 0, sep = ",")
test = pd.read_csv("test.csv", header = 0, sep = ",")
#replace empty string by missing values
train.replace('', np.nan, inplace=True)
train.replace('', np.nan, inplace=True)

#split the training and testing dataset
y = train['Survived']
train = train.drop('Survived',1)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.25, random_state=2)

#generate a new column named Title in the train and testing dataset
Title_train = []
X_train=X_train.reset_index(drop=True)
for elem in X_train['Name']:
    value = re.sub('(.*, )|(\\..*)', '', elem)
    Title_train.append(value)
Title_train=pd.DataFrame(Title_train)
X_train = pd.concat([X_train, Title_train],axis=1)
X_train.columns.values[11]='Title'

Title_test = []
X_test=X_test.reset_index(drop=True)
for elem in X_test['Name']:
    value = re.sub('(.*, )|(\\..*)', '', elem)
    Title_test.append(value)
Title_test=pd.DataFrame(Title_test)
X_test = pd.concat([X_test, Title_test],axis=1)
X_test.columns.values[11]='Title'

#fill in missing values in Age in the training and testing data set using mean age grouped by title from training data set
for elem in X_train.Title.unique():
    title = elem
    mean_title = np.mean(X_train[X_train.Title==title]['Age'])
    for i,row in X_train.iterrows():
      if np.isnan(row[4]) and row[11]==title:
         X_train.set_value(i,'Age',mean_title)
    for i,row in X_test.iterrows():
      if np.isnan(row[4]) and row[11]==title:
         X_test.set_value(i,'Age',mean_title)

#print X_train[['Age']].isnull().any().any()
#print X_test[['Age']].isnull().any().any()

#extract relevant features in the train and test datasets
myfeatures = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
X_train = X_train[myfeatures]
X_test = X_test[myfeatures]
#check proportions of missing values
print X_train.apply(lambda x: sum(x.isnull().values), axis = 0)/X_train.shape[0]
print X_test.apply(lambda x: sum(x.isnull().values), axis = 0)/X_test.shape[0]

#a lot of missing values for carbin information, drop carbin from train and test datasets
X_train = X_train.drop('Cabin', 1)
X_test = X_test.drop('Cabin', 1)
#remove row with missing value for embarked in the train datasets
#print X_train.shape
null_row=[]
for i,row in X_train.iterrows():
     if pd.isnull(row[6]):
       null_row.append(i)
y_train=y_train.reset_index(drop=True)
X_train.drop(X_train.index[null_row])
y_train.drop(y_train.index[null_row])
#print X_train.shape

#Logistic Regression
model_log = LogisticRegression(fit_intercept = False, C = 1e9)
#print X_train.dtypes
#print y_train.dtypes
print X_train.head(5)
X_train = pd.get_dummies(X_train,columns=['Sex','Embarked','Pclass'])
#print X_train.shape
#print X_train.dtypes
#print X_train.head(5)
y_train=y_train.as_matrix()
X_train=X_train.as_matrix()
#print X_train.names
#print X_train[0,:]
#print y_train[0]
mdl = model_log.fit(X_train, y_train)
#print model_log.coef_
X_test = pd.get_dummies(X_test,columns=['Sex','Embarked','Pclass'])
y_test=y_test.as_matrix()
X_test=X_test.as_matrix()
print model_log.score(X_train,y_train)
print model_log.score(X_test, y_test)
#print model_log.intercept_, model_log.coef_

#random forest model
model_rfm = RandomForestClassifier(n_estimators=5, oob_score=True)
model_rfm.fit(X_train, y_train)
print model_rfm.score(X_train,y_train)
print model_rfm.score(X_test, y_test)

#decision tree model
model_dtm = DecisionTreeClassifier(max_depth=9, criterion="entropy")
model_dtm.fit(X_train,y_train)
print model_dtm.score(X_train,y_train)
print model_dtm.score(X_test, y_test)
#print X_train[0:5]
columns = ['Age','SibSp','Parch','Fare','Sex_Female','Sex_Male','Embarked_C','Embarked_Q','Embarked_S','Pclass_1','Pclass_2','Pclass3']
dotfile = open("tree.dot", 'w')
export_graphviz(model_dtm, out_file='tree.dot', feature_names=columns)
dotfile.close()
system("dot -Tpng tree.dot -o tree.png")
