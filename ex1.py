from  Embedd import dataproc, modeler, experiment
import numpy as np
import pandas as pd
from PIL import Image

from keras.models import load_model
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

embedding_model = '/home/piotr/data/test/models/fun_300_Embeding_baseline.h5'
categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']

X_train, X_test = experiment.dataload_minmaxall(categorical,embedding_model,weights)
X_train,y_train = dataproc.split_data(X_train,'Salary')
X_test, y_test = dataproc.split_data(X_test,'Salary')

X_train = dataproc.to_numpy_data(X_train,X_train.columns)
X_test = dataproc.to_numpy_data(X_test,X_test.columns)

X_train =X_train.reshape(-1,7,8)

experiment.img_save(X_train,10,500)





















