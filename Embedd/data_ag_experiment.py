import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from Embedd import modeler, dataproc
import numpy as np
from keras.models import load_model
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)


dataproc.fix_seeds(1)

embedding_model = '/home/piotr/data/test/models/fun_300_EmbeddSource.h5'
categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']



############ TRAIN SEQ 50 Embedd2DF ##########################################

epochs = 200
model_name = f'seq_{epochs}_Embed2DF_baseline-outof-300emb_3'
#
train, test = dataproc.dataload_minmaxall(categorical, embedding_model, weights)
X_train,y_train = dataproc.split_data(train,'Salary')
X_test, y_test = dataproc.split_data(test,'Salary')

model = modeler.get_model_Seq((56,))
model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
modeler.evaluateSeqModel(X_test, y_test, model, model_name)





























