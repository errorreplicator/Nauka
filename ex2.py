from Embedd import dataproc
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)
from pathlib import Path
categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
weights_path ='/home/piotr/data/test/models/fun_300_Embeding_baseline.h5'

path = Path('/home/piotr/data/test/')

X_train,X_test = dataproc.dataload_minmaxall(categorical,weights_path,weights_list)

X_train.to_csv(path/'X_train_minmaxON.csv')
X_test.to_csv(path/'X_test_minmaxON.csv')
print(X_train.head())
