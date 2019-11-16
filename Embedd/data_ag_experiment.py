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

###############Train Embeddings####################################################################

# epochs = 300
# model_name = f'fun_{epochs}_EmbeddSource_flatten_names'
# #
# X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical, numerical,to_dict=True)
#
# model = modeler.get_model_Emb1DropoutBIG()
#
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel(X_test, y_test, model, model_name)
########################### Sequential Embeddings OFF########################################################
# epochs = 200
# model_name = f'seq_{epochs}_EmbeddOFF'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Seq((13,))
# model.fit(X_train,y_train,epochs=200,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)
############################# Sequential Embedding ON ######################################################
epochs = 500
model_name = f'seq_{epochs}_EmbeddON'
X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical,numpyON=False)
weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource_flatten_names.h5'

X_train = dataproc.weights2df(X_train,weights_path,weights_list)
X_test = dataproc.weights2df(X_test,weights_path,weights_list)

model = modeler.get_model_Seq((56,))
model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
model.save(f'/home/piotr/data/test/models/{model_name}.h5')

modeler.evaluateSeqModel(X_test,y_test,model,model_name)
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
# epochs = 300
# model_name = f'fun_{epochs}_tester'
# X_train, y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical,numpyON=False)
# weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
# weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource_flatten_names.h5'
#
# X_train = dataproc.weights2df(X_train,weights_path,weights_list)
# X_test = dataproc.weights2df(X_test,weights_path,weights_list)
#
# # print(X_train.head())
# X_train_cat, X_train_num = dataproc.split_data(X_train, numerical)
# X_test_cat, X_test_num = dataproc.split_data(X_test, numerical)
#
# model = modeler.tester()
# model.fit([X_train_cat, X_train_num], y_train, epochs=epochs, batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel([X_test_cat, X_test_num],y_test, model,model_name)




# model = modeler.get_model_SeqDrop((56,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')

# modeler.evaluateSeqModel(X_test,y_test,model,model_name)

# property = X_train.describe()
# print(property.transpose())

































