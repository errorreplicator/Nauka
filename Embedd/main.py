from Embedd import modeler, dataproc
import numpy as np
np.set_printoptions(threshold=np.inf)
from keras.models import Sequential, load_model, Model
from sklearn.metrics import confusion_matrix
categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

model = Model()
# model.load_weights('/home/piotr/data/test/model_300_swithOFF_simple.h5')

X_train, y_train, X_test, y_test = dataproc.data_func_swithONscaleROW()

model = load_model('/home/piotr/data/test/model_300_swithON_EmbeddBIGDrop.h5')
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.predict(X_test)


# y_pred = model.predict_classes(X_test)
result = model.evaluate(X_test,y_test)
print(result)
