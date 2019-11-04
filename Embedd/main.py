from Embedd import modeler, dataproc
import numpy as np
np.set_printoptions(threshold=np.inf)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
#
# X_train, y_train, X_test, y_test = dataproc.clean_data_categ(categorical, numerical)
#
# model = modeler.get_model_Emb1()
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# train, test = dataproc.read_data()


X_train, y_train, X_test, y_test = dataproc.data_categ_numpy(categorical, numerical)

print(y_test)

