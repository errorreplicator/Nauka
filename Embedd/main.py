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

epochs = 300
model_name = f'fun_{epochs}_EmbeddSource_flatten_names'
#
# X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Emb1DropoutBIG()
#
# model.fit(X_train,y_train,epochs=epochs,batch_size=128)
# model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

############ TRAIN SEQ with Embeddings in DATAFTAME##########################################

X_train,y_train,X_test,y_test = data_func_swithOFF(categorical,numerical,to_dict=False)

print(X_train)

model = load_model(f'/home/piotr/data/test/models/{model_name}.h5')

print(X_train)

X_train_dict = {col: dataproc.to_numpy_data(X_train, col) for col in categorical}
X_train_dict['Numerical'] = dataproc.to_numpy_data(X_train, numerical)


def to_numpy_data(df, column_list):
    return np.array(df[column_list])


embedding = model.get_layer('Sex_emb').get_weights()
# flatten = model.get_layer('Sex_emb').output
#
# concattenate = model.get_layer('concat_all').get_weights()
#
# print(flatten)
# # print(concattenate)
# print(model.summary())

layer_name = 'concat_all'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_train_dict)
print(embedding)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(intermediate_output)


##### TODO
# Do not delete categorical only one hot encode or other idea and only add embedding representation 1.
# bucketiz numerical values and embedd it as categorical 1. check if model can lern if YES 2. get embeddings to DF and try to learn
# check google reuse embeddins