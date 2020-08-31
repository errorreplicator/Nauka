def ile(n):
    for x in range(1,n+1):
        if x == 1:
            start = 1
        if x%2 == 0:
            start +=3
        if x%2 != 0 and x>2:
            start-=1
        if x%12==0:
            start-=9
    print(start)
ile(100)
