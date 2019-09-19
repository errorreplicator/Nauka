import pandas as pd
import matplotlib.pyplot as mpp
import matplotlib.style as ms

ms.use('ggplot')

main_data = pd.read_pickle('StateData2.dat')
main_data = pd.DataFrame(main_data)

main_data['TX1Y'] = main_data['TX'].resample('A').mean()
print(main_data[['TX','TX1Y']].head(5))
# main_data.dropna(inplace=True)
# main_data.fillna(method='bfill',inplace=True)
# main_data.fillna(value=-9999,inplace=True)
main_data.fillna(value=-9999,limit=100, inplace=True)


print(main_data[['TX','TX1Y']].head())
print(main_data['TX1Y'].isnull().sum())