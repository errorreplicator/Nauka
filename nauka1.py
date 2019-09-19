import Quandl as qd
import pandas as pd

ds = qd.get("FMAC/HPI_AL", authtoken="_VC8mdqyTzENHcUoUxnu")
# print(ds.head())

web_data  = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

web_table = web_data[0][0][1:]

# print(web_table[0][0][1:])
# print(ds.shape)

for abbv in web_table:
    print('FMAC/HPI_'+str(abbv))