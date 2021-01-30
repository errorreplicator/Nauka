import turtle as t
from random import randint
#turtle.dot(60, color="yellow")
#t.pos()
# A, B, C = 0, 0, 0
t.speed(0)
t.delay(0)
ABC = []
def trojkat():
    t.penup()
    t.bk(220)
    t.lt(90)
    t.bk(220)
    # A = t.pos()
    ABC.append(t.pos())
    t.dot()
    t.rt(90)
    t.fd(450)
    t.dot()
    ABC.append(t.pos())
    # B = t.pos()
    t.lt(120)
    t.fd(450)
    t.dot()
    ABC.append(t.pos())
    # C = t.pos()
    t.goto(0,0)
def czworo():
    t.penup()
    t.bk(220)
    t.lt(90)
    t.bk(220)
    t.rt(90)
    for x in range(4):
        ABC.append(t.pos())
        t.dot()
        t.fd(450)
        t.lt(90)
    t.goto(0,0)
def loop(n,do_ilu):
    for x in range(n):
        index = randint(0,do_ilu)
        t.setheading(t.towards(ABC[index]))
        distance = t.distance(ABC[index])/2
        t.fd(distance)
        t.dot()
        if x%100==0:print(x)

def loop2(n,do_ilu):
    index = 0
    for x in range(n):
        index2 = randint(0,do_ilu)
        while index == index2:
            index2 = randint(0, do_ilu)
        index = index2
        t.setheading(t.towards(ABC[index]))
        distance = t.distance(ABC[index])/2
        t.fd(distance)
        t.dot()
        if x%100==0:print(x)
czworo()
# trojkat()
loop2(10000,3)
t.done()