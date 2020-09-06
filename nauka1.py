liczba = 20
L1 = []
L2 = []
L3 = []
L4 = []

licznik = 1
flaga = 1
glowna, lista = [],[]
for z in range(1,liczba+1):
    lista.append(z)
print(lista)

for x in range(1,liczba+1):
    if x%4 != 0:
        glowna.append(licznik)
        # print(licznik)
    if x%4==0:
        licznik+=1
        glowna.append(licznik)
        # print(licznik)
        licznik+=1

print(glowna)
print(45*'#')
index = 0
flaga_tablicy = 0

for y in glowna:
    if index >= len(lista):
        break
    if flaga_tablicy >= 4: flaga_tablicy = 1
    else:flaga_tablicy+=1

    for z in range(y):
        if flaga_tablicy == 1:
            L1.append(lista[index])
            index+=1
            if index >= len(lista):
                break

        elif flaga_tablicy == 2:
            L2.append(lista[index])
            index+=1
            if index >= len(lista):
                break

        elif flaga_tablicy == 3:
            L3.append(lista[index])
            index+=1
            if index >= len(lista):
                break
        else:
            L4.append(lista[index])
            index += 1
            if index >= len(lista):
                break


print(L1)
print(L2)
print(L3)
print(L4)
