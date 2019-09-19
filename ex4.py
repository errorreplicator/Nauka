import Quandl as qd
import pandas as pd
import pickle

def get_States():
    web_url = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    web_tab = web_url[0][0][1:]
    return web_tab

# print(web_tab)
def set_DataFrame():
    df = pd.DataFrame()
    list = get_States()

    for abbv in list:
            ds = qd.get('FMAC/HPI_' + str(abbv),authtoken="_VC8mdqyTzENHcUoUxnu")
            # ds.columns = ds.columns + str(abbv)
            ds.rename(columns={'Value': str(abbv)}, inplace=True)
            if df.empty:
                df = ds
            else:
                df = df.join(ds)

    # pickle_out = open('DataSetStates.pickle','wb')
    # pickle.dump(df,pickle_out)
    # pickle_out.close()

    df.to_pickle('StateData.dat')

#set_DataFrame()

main_data = pd.read_pickle('StateData.dat')
print(type(main_data))
print(main_data.head())

# pickel_in = open('DataSetStates.pickle','rb')
# main_df = pickle.load(pickel_in)
# print(main_df)


# print(df.head())
# print(df.shape)


