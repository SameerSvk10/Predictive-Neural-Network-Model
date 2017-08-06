# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 22:27:26 2017

@author: Sameer
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np

test_data = pd.read_csv('Classification1Test.txt', sep='\t')
test_data.head()

test_data.isnull().sum()
test_data.info()

import datetime as dt
test_data['F15'] = pd.to_datetime(test_data['F15'])
test_data['F15']=test_data['F15'].map(dt.datetime.toordinal)

test_data['F16'] = pd.to_datetime(test_data['F16'])
test_data['F16']=test_data['F16'].map(dt.datetime.toordinal)

                                 
test_data.drop(['Index'],axis=1,inplace=True)
test_data.head()


test_data_ = test_data.ix[:,:].values

scaler = preprocessing.StandardScaler().fit(test_data_)
 
test_data=scaler.transform(test_data_)

from keras.models import load_model
model = load_model('my_model1.h5')

test_target_pred = model.predict_classes(test_data_) 

np.savetxt('test_target_pred.txt',test_target_pred ,fmt='%i', delimiter=' ')
