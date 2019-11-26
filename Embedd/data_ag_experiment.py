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
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Country_emb']

###############Train Embeddings 50 baseline ######################################################
#[0.31105740891888395, 0.8560899207335887]
[0.44659990448325987, 0.8505005834692929]
# epochs = 50
# model_name = f'fun_{epochs}_EmbeddSource_15x15'
#
# X_train, X_test = dataproc.dataload_stage1(categorical,numerical,onehot=False)
#
# X_train, y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train = dataproc.data_tomodel(X_train,categorical,numerical)
# X_test = dataproc.data_tomodel(X_test,categorical,numerical)
#
# model = modeler.exp_model_Emb1DropoutBIG_corr()
# #
# model.fit(X_train,y_train,epochs=epochs,batch_size=128)
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

################### CNN2D + Dense on Embeddings swith ON MinMax ALL#############################################
import numpy as np
embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_15x15.h5'
epochs = 100
model_name = f'picture_baseline_{epochs}_Embeding_CNN2DandDense_2'
batch_size = 1024
X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)


numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
pixels = [col for col in X_train.columns if col not in numerical_col]
# X_train = dataproc.swith_merge(X_train, numerical_col)
X_train_pixels = X_train[pixels]
X_train_numerical = X_train[numerical_col]
X_test_pixels = X_test[pixels]
X_test_numerical = X_test[numerical_col]

X_test_numerical.drop('Salary',axis=1,inplace=True)
X_train_numerical.drop('Salary',axis=1,inplace=True)

_,y_train = dataproc.split_data(X_train,'Salary')
_, y_test = dataproc.split_data(X_test,'Salary')

X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)
# print(X_train_pixels.shape)
train_zeros = np.zeros((len(X_train),11))
test_zeros = np.zeros((len(X_test),11))
X_train_pixels = np.concatenate((X_train_pixels,train_zeros),1)
X_test_pixels = np.concatenate((X_test_pixels,test_zeros),1)

# [0.4194015270048362, 0.8439899268757176]
# [0.41646632663532657, 0.8435599778553872]
# print(X_train_pixels[0])
# print(X_train_pixels.shape)
# print(X_train_pixels[0].shape)

X_train_pixels =X_train_pixels.reshape(-1,14,14,1)
X_test_pixels = X_test_pixels.reshape(-1,14,14,1)
#
model = modeler.model_CNN_Dense(CNN_shape=(14,14,1),Dense_shape=(6,))
model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)





























