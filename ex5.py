
def dukaty(ile_dni):
    suma = 0
    for x in range(1,ile_dni+1):
        if (x%5==0):
            suma+=3
        else:
            suma+=1
        if suma==50:
            suma-=37
    print(suma)
dukaty(36)


