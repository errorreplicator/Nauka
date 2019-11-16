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


# print(X_test.shape)


###############Train Embeddings####################################################################

# X_train, y_train, X_test, y_test = dataproc.data_func_swithOFF()
# model = modeler.get_model_Emb1DropoutBIG()
# model.fit(X_train,y_train,epochs=300)
# model.save('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')

############ TRAIN SEQ with Embeddings in DATAFTAME##########################################

X_train, y_train, X_test, y_test = dataproc.dataframe_seq_swithOFF()
model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')
weight_dict = {}
for layer in weights:
    weight_dict[layer] = model.get_layer(layer).get_weights()
df_train = dataproc.dict2df(weight_dict,X_train)
model_seq = modeler.get_model_Seq((56,))
model_seq.fit(df_train,y_train,epochs=100)
model_seq.save('/home/piotr/data/test/model_300_swithOFF_seq_embedd2DF.h5')


print(df_train.head())
print([x for x in df_train.columns if x.startswith('W')])

sumary = df_train.describe()
sumary = sumary.transpose()
print(sumary)


###############TEST $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# X_train, y_train, X_test, y_test = dataproc.dataframe_seq_swithOFF()
# model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')

# check if encoding columns are deleted
# check mean and std
# check data X_train X_test
# weight_dict = {}
# for layer in weights:
#     weight_dict[layer] = model.get_layer(layer).get_weights()
# df_train = dataproc.dict2df(weight_dict,X_train)
# df_test = dataproc.dict2df(weight_dict,X_test)
#
# # model_test = load_model('/home/piotr/data/test/model_300_swithOFF_seq_embedd2DF.h5')
# model_test = modeler.get_model_Seq((56,))
# model_test.fit(df_train,y_train,epochs=300)
# model_test.save('/home/piotr/data/test/model_Seq_300_trainONembedd.h5')
# result = model_test.evaluate(df_test,y_test)
# print(result)

###################################### train vs test data comparision #############################

X_train, y_train, X_test, y_test = dataproc.dataframe_seq_swithOFF(categorical,numerical=numerical)
model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')
# print(X_test.head())
# print(pd.value_counts(y_test))
# print(3846/12435)
weight_dict = {}
for layer in weights:
    weight_dict[layer] = model.get_layer(layer).get_weights()
df_train = dataproc.dict2df(weight_dict,X_train)
df_test = dataproc.dict2df(weight_dict,X_test)
model = load_model('/home/piotr/data/test/model_Seq_300_trainONembedd.h5')
# y_test = np.array(y_test)
# y_test = y_test.reshape((1,-1))
# print(model.summary())
print(type(df_test.values))
print(type(y_test.values))
classes = model.predict_classes(df_test.values)
print(classes)