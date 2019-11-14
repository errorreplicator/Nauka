import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Embedd import modeler, dataproc
from sklearn.preprocessing import MinMaxScaler

desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical)

model = modeler.get_model_Seq((13,))
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=300)

model.save('/home/piotr/data/test/seq_300_EmbeddOFF.h5')

eval = model.evaluate(X_test,y_test)
print(eval)

pred = model.predict(X_test)
print(pred)
pred_class = model.predict_classes(X_test)
print(pred_class)
# proba = model.predict_proba(X_test)
# print(proba)
# confiusion matrinx
# probabilities and classes to dataframe