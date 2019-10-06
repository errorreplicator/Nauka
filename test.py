import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
np.set_printoptions(linewidth=np.inf)
np.random.seed(10)
zm1 = np.random.randint(0,24*60*60,200)
# print(zm1)
# print(24*60*60)

def rand_time(n):
    sec = np.random.randint(0,24*60*60,n)
    return sec
df = pd.DataFrame()

df['sec'] = rand_time(1000)

df = df.sort_values('sec').reset_index(drop=True)
print(df.head())
print(np.pi).