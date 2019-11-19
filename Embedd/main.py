from Embedd import modeler, dataproc
import numpy as np
import pandas as pd
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
np.set_printoptions(threshold=np.inf)
from keras.models import Sequential, load_model, Model
from sklearn.metrics import confusion_matrix
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']

###############Train Embeddings####################################################################

# epochs = 300
# model_name = f'fun_{epochs}_EmbeddSource_flatten_names'
#
# X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Emb1DropoutBIG()
#
# model.fit(X_train,y_train,epochs=epochs,batch_size=128)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

############ TRAIN SEQ stage 1 OFF embeddings OFF one hot##########################################

# epochs = 300
# model_name = f'seq_{epochs}_EmbedOFF_oneHotOFF_baseline'
#
# train, test = dataproc.dataload_stage1(categorical,numerical)
# X_train,y_train = dataproc.split_data(train,'Salary')
# X_test, y_test = dataproc.split_data(test,'Salary')
#
# model = modeler.get_model_Seq((13,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test, y_test, model, model_name)

############ TRAIN SEQ stage 1 OFF embeddings ON one hot##########################################

# epochs = 300
# model_name = f'seq_{epochs}_EmbedOFF_oneHotON_baseline'
#
# train, test = dataproc.dataload_stage1(categorical,numerical,onehot=True)
# X_train,y_train = dataproc.split_data(train,'Salary')
# X_test, y_test = dataproc.split_data(test,'Salary')
#
# model = modeler.get_model_Seq((107,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test, y_test, model, model_name)

############ TRAIN Embedding baseline ##########################################
# epochs = 300
# model_name = f'fun_{epochs}_Embeding_baseline'
#
# X_train,y_train,X_test,y_test = dataproc.fun_swithOFF(categorical,numerical,to_dict=True)
#
# model = modeler.get_model_Emb1Dropout()
# model.fit(X_train,y_train,batch_size=1024,epochs=epochs)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateFunModel(X_test,y_test,model,model_name)

###################SEQ Embedding to DF ###########################################
epochs = 300
model_name = f'seq_{epochs}_Embeding_toDF'
embedding_model = '/home/piotr/data/test/models/fun_300_Embeding_baseline.h5'
train, test = dataproc.dataload_stage1(categorical,numerical)

# print(train.head())
X_train = dataproc.weights2df(train,embedding_model,weights,del_categ=True,normalize=False)
X_test = dataproc.weights2df(test,embedding_model,weights,del_categ=True,normalize=False)

X_train,y_train = dataproc.split_data(X_train,'Salary')
X_test, y_test = dataproc.split_data(X_test,'Salary')


# X_train.to_csv('/home/piotr/data/test/models/train.csv')
# X_test.to_csv('/home/piotr/data/test/models/test.csv')
model = modeler.get_model_Seq((56,))
model.fit(X_train,y_train,batch_size=1024,epochs=epochs)
modeler.evaluateSeqModel(X_test,y_test,model,model_name)



###################SEQ Embedding to DF + not delete label encoded#############################


###################Fun Embedding to DF #############################
#try simple embedding model with embedd to DF
#CNN ? try again CNN model with embedd to DF conv1D
#normalize embedding with other features so all is on the same scale (try sequential and functional)
#try bucketizing numerical variables
#switch data
#try CNN after data switch























##### TO DO
# Do not delete categorical only one hot encode or other idea and only add embedding representation 1.
# bucketiz numerical values and embedd it as categorical 1. check if model can lern if YES 2. get embeddings to DF and try to learn
# check google reuse embeddins