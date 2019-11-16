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

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']


########################### Sequential Embeddings OFF########################################################
# model_name = 'seq_200_EmbeddOFF_tmp'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Seq((13,))
# model.fit(X_train,y_train,epochs=200,batch_size=1024)
# model.save(f'/home/piotr/data/test/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)

########################### Functional Embeddings ON########################################################
# epochs = 200
# model_name = f'fun_{epochs}_EmbeddONBIG_3run_128startlayer'
#
# X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Emb1DropoutBIG()
#
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

###########################Embedings to Features#####################################################
epochs = 200
model_name = f'seqDrop_{epochs}_EmbeddinDF_1run'
X_train, y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical,numpyON=False)
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
model_path ='/home/piotr/data/test/models/fun_200_EmbeddON_1run.h5'

X_train = dataproc.weights2df(X_train,model_path,weights)
X_test = dataproc.weights2df(X_test,model_path,weights)

model = modeler.get_model_SeqDrop((56,))
model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
model.save(f'/home/piotr/data/test/models/{model_name}.h5')

modeler.evaluateSeqModel(X_test,y_test,model,model_name)

# property = X_train.describe()
# print(property.transpose())

































