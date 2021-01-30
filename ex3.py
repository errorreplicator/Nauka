from numpy.random import choice
from collections import Counter

probabils = []
dania = ['hot-dog', 'pizza', 'hamburger']
mc_chain = {'hot-dog':[0.5, 0.0, 0.5],'pizza':[0.7, 0.0, 0.3],'hamburger':[0.2, 0.6, 0.2]}
start = mc_chain['hot-dog']
n=100000
for x in range(n):
    if x%10000==0:print(x)
    pick = choice(dania, 1, p=start)
    start = mc_chain[pick[0]]
    probabils.append(pick[0])
    # print(pick[0])
print(Counter(probabils))
zlicz = Counter(probabils)

for key, item in zlicz.items():
    print(key,item, item/n)