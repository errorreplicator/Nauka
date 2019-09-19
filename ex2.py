import pandas as pd
import numpy as np
import re
pd.set_option('display.width',320)

names = ['user_id','movie_id','rating']
movies = pd.read_csv('../1/u.data', sep='\t',names=names,usecols=range(3))
cols = ['movie_id','title']
movies2 = pd.read_csv('../1/u.item',sep='|',usecols=range(2),encoding = 'ISO-8859-1',names=cols)

imdb = pd.merge(movies,movies2,on='movie_id')
imdb = pd.DataFrame(imdb)
# print(imdb[imdb['title'] == '1-900 (1994)'])
ratings = imdb.pivot_table(index='user_id',columns='title',values='rating')
ratingsSW = ratings['Star Wars (1977)']
ratings = pd.DataFrame(ratings)
# print(ratingsSW.head())

similarMovies = ratings.corrwith(ratingsSW)
similarMovies = similarMovies.dropna()
# print(similarMovies.sort_values(ascending=False))

popularMovies = imdb.groupby('title').agg({'rating':[np.size,np.mean]})
popularMovies = popularMovies[popularMovies['rating']['size'] >=200]
# print(popularMovies.head(20))
df = popularMovies.join(pd.DataFrame(similarMovies,columns=['similarity']))
print(df.sort_values('similarity',ascending=False)[:20])
# tmp = []
# tmp = ratings.columns
#
# r = re.compile("Star")
# newlist = filter(r.match, tmp)
# print (list(newlist))

# ratingsSW = ratings['Star Wars']
# print(ratings.head())