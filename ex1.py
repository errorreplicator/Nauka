import turtle as t
import math as m
def trojkat(bok):

    t.fd(bok)
    t.lt(135)
    t.fd(m.sqrt(bok**2+bok**2))
    t.lt(135)
    t.fd(bok)

def okont5():
    t.rt(135)
    t.fillcolor('grey')
    t.begin_fill()
    for x in range(5):
        trojkat(100)
        t.rt(360/5-90)
    t.end_fill()
okont5()
t.done()