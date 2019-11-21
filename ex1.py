from  Embedd import dataproc
import Embedd.modeler as mod
import numpy as np
import pandas as pd
from keras.models import load_model
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

d1 = {'a': [1,2,3,4],'b': [5,6,7,8],'c':[7,33,22,11]}
d2 = {'a': [22,23],'b':[24,25]}

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

# print(df1.shape)
# df = dataproc.swith_merge(df1,[])

df = dataproc.minmax_row(df1,[])
print(df)

























