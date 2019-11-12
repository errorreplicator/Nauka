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

X_train, y_train, X_test, y_test = dataproc.dataframe_seq_swithOFF()

model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')

weight_dict = {}

for layer in weights:
    weight_dict[layer] = model.get_layer(layer).get_weights()


df = dataproc.dict2df(weight_dict,X_train)

filter_col = [x for x in df.columns if x.startswith('S')] # remove cloumns
# print(df.loc[df['Sex'] == 0,filter_col] )
print(df[filter_col])



# model = Model()
# model.load_weights('/home/piotr/data/test/model_300_swithOFF_simple.h5')

# model = modeler.get_model_Emb1DropoutBIG()
# X_train, y_train, X_test, y_test = dataproc.data_func_swithOFF()
# model.fit(X_train,y_train,epochs=300)
# model.save('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')

# model = load_model('/home/piotr/data/test/model_300_swithOFF_EmbeddBIGDrop_batch32.h5')
# result = model.evaluate(X_test,y_test)
# print(result)

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.predict(X_test)
# y_pred = model.predict_classes(X_test)