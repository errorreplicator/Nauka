import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_predict
pd.set_option('display.width',320)

ds = pd.read_csv('../1/PastHires.csv')

maps = {'Y':1,'N':0}
ds['Employed?'] = ds['Employed?'].map(maps)
ds['Top-tier school'] = ds['Top-tier school'].map(maps)
ds['Interned'] = ds['Interned'].map(maps)
ds['Hired'] = ds['Hired'].map(maps)
ds['LevelofE'] = pd.Categorical(ds['Level of Education']).codes
ds.drop('Level of Education',axis=1,inplace=True)


train = pd.DataFrame(ds[:11])
test = pd.DataFrame(ds[11:])
X_train = train.drop('Hired',axis=1)
y_train = train['Hired']
X_test = test.drop('Hired',axis=1)
y_test = test['Hired']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)
# score = clf.score(X_test,y_test)
soreVal = cross_val_predict(clf,X_test,y_test,cv=2)

print(soreVal)


