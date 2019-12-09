import numpy as np
import pandas as pd
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)
from Embedd import dataproc,modeler


train,test  = dataproc.read_data()

sub_t = train.iloc[:10,:3]
new_t = pd.DataFrame()
print(sub_t)
for x in range(4):
    for col in sub_t.columns:
        new_t[f'{col}_{x}'] = sub_t[col]

print(new_t)
