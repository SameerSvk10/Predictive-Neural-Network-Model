# -*- coding: utf-8 -*
"""
Created on Fri Aug 03 19:15:22 2017

@author: Sameer
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.optimizers import RMSprop

#Reading a tab-delimited file
data = pd.read_csv('ClassificationProblem1.txt', sep='\t')
data.head()

#Checking for any missing values
data.isnull().sum()
data.info()

#converting the date feature to an ordinal i.e. an integer
#representing the number of days since year 1 day 1
import datetime as dt
data['F15'] = pd.to_datetime(data['F15'])
data['F15']=data['F15'].map(dt.datetime.toordinal)

data['F16'] = pd.to_datetime(data['F16'])
data['F16']=data['F16'].map(dt.datetime.toordinal)

#UPsampling to make the observations from both classes qual
from sklearn.utils import resample
df_majority = data[data.C==0]
df_minority = data[data.C==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=76353,
                                 random_state=123) 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.C.value_counts()
                                 
df_upsampled.drop(['Index'],axis=1,inplace=True)
df_upsampled.head()

#separating input features(X) and target variable(y)
X = df_upsampled.ix[:,:22].values
y = df_upsampled.ix[:,22].values

#standardize features by removing the
#mean and scaling to unit variance from entire dataset
scaler = preprocessing.StandardScaler().fit(X)
 
X=scaler.transform(X)

#Splitting the 20% portion of the dataset into validation set
#to evaluate the performance of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=20)

#Defining NN Model
model = Sequential()
model.add(Dense(1024, input_dim=22, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

#Training of model
model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=512,validation_data=(X_test, y_test),
          shuffle=True,verbose=2)

#Saving a model
model.save('my_model1.h5')  

#Classification Report and Confusion matrix
y_pred = model.predict_classes(X_test) 
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
model.score(X_test,y_test)

#k-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
import numpy
seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(1024, input_dim=22, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])
    model.fit(X[train], y[train], epochs=20, batch_size=512, verbose=0)
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    
    
