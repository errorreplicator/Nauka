import Quandl as qd
import pandas as pd
import matplotlib.pyplot as mpp
import matplotlib.style as ms

ms.use('ggplot')

def get_States():
    web_url = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    web_tab = web_url[0][0][1:]
    return web_tab


def set_DataFrame():
    df = pd.DataFrame()
    list = get_States()

    for abbv in list:
            ds = qd.get('FMAC/HPI_' + str(abbv),authtoken="_VC8mdqyTzENHcUoUxnu")
            ds.rename(columns={'Value': str(abbv)}, inplace=True)
            ds[abbv] = (ds[abbv] - ds[abbv][0]) / ds[abbv][0] * 100.0
            if df.empty:
                df = ds
            else:
                df = df.join(ds)


    df.to_pickle('StateData2.dat')

# set_DataFrame()

main_data = pd.read_pickle('StateData2.dat')
# print(type(main_data))
# print(main_data.head())

# main_data.plot()
# mpp.legend().remove()
# mpp.show()

# corelation = main_data.corr()
info = main_data.describe()
print(info)