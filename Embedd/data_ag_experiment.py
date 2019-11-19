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
# model_name = f'fun_{epochs}_EmbeddSource'
#
# X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical, numerical,to_dict=True)
#
# model = modeler.get_model_Emb1DropoutBIG()
#
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel(X_test, y_test, model, model_name)
########################### Sequential Embeddings OFF########################################################
# epochs = 300
# model_name = f'seq_{epochs}_EmbeddOFF'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Seq((13,))
# model.fit(X_train,y_train,epochs=200,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test, y_test, model, model_name)
############################# Sequential Embedding ON ######################################################
# epochs =300
# model_name = f'seq_{epochs}_EmbeddON_delCategON'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical, numerical, numpyON=False, expandY=False)
# weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
# weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource.h5'
#
# X_train = dataproc.weights2df(X_train,weights_path,weights_list)
# X_test = dataproc.weights2df(X_test,weights_path,weights_list)
#
# model = modeler.get_model_Seq((56,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024,validation_split=0.2)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)
############################# Sequential Embedding ON delCateg OFF######################################################
epochs =300
model_name = f'seq_{epochs}_EmbeddON_delCategOFF'
X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical, numerical, numpyON=False, expandY=False)
weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource.h5'

X_train = dataproc.weights2df(X_train,weights_path,weights_list,del_categ=False,normalize=True)
X_test = dataproc.weights2df(X_test,weights_path,weights_list,del_categ=False,normalize=True)
print(X_train.iloc[:,1:5].head())
print(X_train.head())
X_train = dataproc.minmax_row(X_train, X_train.columns)
X_test = dataproc.minmax_row(X_test, X_test.columns)
print(X_train.head())
# model = modeler.get_model_Seq((64,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024,validation_split=0.2)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)
########################### Functional Embeddings ON del Categ ON########################################################

# epochs = 300
# model_name = f'seq_{epochs}_EmbeddON_delCategOFF'
# X_train, y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical,numpyON=False)
# weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
# weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource.h5'
#
# X_train = dataproc.weights2df(X_train,weights_path,weights_list,del_categ=False)
# X_test = dataproc.weights2df(X_test,weights_path,weights_list,del_categ=False)
#
# # print(X_train.head())
# X_train_cat, X_train_num = dataproc.split_data(X_train, numerical)
# X_test_cat, X_test_num = dataproc.split_data(X_test, numerical)
#
# model = modeler.tester()
# model.fit([X_train_cat, X_train_num], y_train, epochs=epochs, batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel([X_test_cat, X_test_num],y_test, model,model_name)

################################ CNN ###############################################################

# epochs =100
# model_name = f'seq_{epochs}_EmbeddON_CNN_tester'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical,numpyON=False)
# weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
# weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource.h5'
#
# X_train = dataproc.weights2df(X_train,weights_path,weights_list)
#
# X_test = dataproc.weights2df(X_test,weights_path,weights_list)
# X_train = dataproc.to_numpy_data(X_train,X_train.columns)
#
# X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
#
#
#
#
#
#
# model = modeler.model_Fun_CNN1((X_train.shape[1],1))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024,validation_split=0.2)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)




# model = modeler.get_model_SeqDrop((56,))
# model.fit(X_train,y_train,epochs=epochs,batch_size=1024)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')

# modeler.evaluateSeqModel(X_test,y_test,model,model_name)

# property = X_train.describe()
# print(property.transpose())

































