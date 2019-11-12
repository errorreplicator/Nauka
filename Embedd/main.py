from Embedd import modeler, dataproc
import numpy as np
import pandas as pd

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

X_train, y_train, X_test, y_test = dataproc.data_func_swithOFF()
model = modeler.get_model_Emb1DropoutBIG()
model.fit(X_train,y_train,epochs=300)
model.save('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')

############ TRAIN SEQ with Embeddings in DATAFTAME##########################################

# X_train, y_train, X_test, y_test = dataproc.dataframe_seq_swithOFF()
# model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')
# weight_dict = {}
# for layer in weights:
#     weight_dict[layer] = model.get_layer(layer).get_weights()
# df_train = dataproc.dict2df(weight_dict,X_train)
# model_seq = modeler.get_model_Seq((56,))
# model_seq.fit(df_train,y_train,epochs=300)
# model_seq.save('/home/piotr/data/test/model_300_swithOFF_seq_embedd2DF.h5')

###############TEST $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# df_test = dataproc.dict2df(weight_dict,X_test)
# model_test = load_model('/home/piotr/data/test/model_300_swithOFF_seq_embedd2DF.h5')
# result = model_test.evaluate(df_test,y_test)
# print(result)

###################################### train vs test data comparision #############################
# train, test = dataproc.read_data()
# train = train.drop(axis=0,index=19609)
#
#
# train = dataproc.remove_data(train, 'Id')
# test = dataproc.remove_data(test, 'Id')
#
# train = dataproc.labelencoder_bycopy(train, categorical)
# test = dataproc.labelencoder_bycopy(test, categorical)
# train = dataproc.labelencoder(train, ['Salary'])
# test = dataproc.labelencoder(test, ['Salary'])
#
#
# for x in categorical:
#     print(train.groupby([x, f'{x}_encode']).size())
#     print(test.groupby([x, f'{x}_encode']).size())
#     # print(list(test[x].unique()), test[f'{x}_encode'].unique())
#     print('#'*50)


# filter_col = [x for x in df_train.columns if x.startswith('W')] # remove cloumns
# print(df_train[filter_col])
# print(df_test[filter_col])

########################################################################################################


# # print(df.loc[df['Sex'] == 0,filter_col] )
#
# df = dataproc.to_numpy_data(df,df.columns)
#
# model_seq = modeler.get_model_Seq((56,))
# model_seq.fit(df,y_train,epochs=300)
# model.save('/home/piotr/data/test/model_300_swithOFF_seq_embedd2DF.h5')


# model = Model()
# model.load_weights('/home/piotr/data/test/model_300_swithOFF_simple.h5')

# model = modeler.get_model_Emb1DropoutBIG()
# X_train, y_train, X_test, y_test = dataproc.data_func_swithOFF()
# model.fit(X_train,y_train,epochs=300)
# model.save('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')



# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.predict(X_test)
# y_pred = model.predict_classes(X_test)