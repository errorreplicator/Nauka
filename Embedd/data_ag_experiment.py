slowo = 'ppapadaaaamaaapaappaddmppaapadaaaamppaa'
nalesnik = ['p', 'a', 'p', 'a','d','a','m']
index = 0
ile_n = 0

for litera in slowo:
    if litera == nalesnik[index]:
        print(nalesnik[index],index)
        index+=1
    if litera == 'm' and index == 7:
        ile_n+=1
        index=0

print(ile_n)

