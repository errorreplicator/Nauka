import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Input


path = Path('c:/data/amazon')
files = [x.name for x in path.iterdir() if x.suffix == '.csv']

df_train_all = pd.read_csv(path/files[2])
# print(df_train_all.columns)
# print(df_train_all.info())
# print(pd.isnull(df_train_all).sum())
print(df_train_all.apply(lambda x: len(x.unique())))
train_sub = df_train_all[['ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_TITLE','ROLE_FAMILY']]

encode = LabelEncoder()
scaler = MinMaxScaler()

train_sub['rr1_encode'] = encode.fit_transform(train_sub['ROLE_ROLLUP_1'])
train_sub['rr2_encode'] = encode.fit_transform(train_sub['ROLE_ROLLUP_2'])
train_sub['rtit_encode'] = encode.fit_transform(train_sub['ROLE_TITLE'])
train_sub['rfam_encode'] = encode.fit_transform(train_sub['ROLE_FAMILY'])

train_sub['rr2_scale'] = scaler.fit_transform(train_sub['ROLE_ROLLUP_2'])
train_sub['rtit_scale'] = scaler.fit_transform(train_sub['ROLE_TITLE'])
train_sub['rfam_scale'] = scaler.fit_transform(train_sub['ROLE_FAMILY'])

vocab_len = train_sub['rr1_encode'].nunique()
emb_size = min(vocab_len,50)

model = Sequential()
model.add(Embedding(input_dim=vocab_len,output_dim=emb_size,input_length=3, name='rr1_embedding'))