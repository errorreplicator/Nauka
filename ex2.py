from collections import Counter
tablica = ['julka','lubi','psyaayyu']
slowo = ''.join(tablica)
print(slowo)
slownik = Counter(slowo)
print(slownik)
najwieksza = 0
literki = []
for key,item in slownik.items():
    if item > najwieksza:
        najwieksza = item

for key,item in slownik.items():
    if item == najwieksza:
        literki.append(key)
print(literki)
literki.sort()
print(literki)


