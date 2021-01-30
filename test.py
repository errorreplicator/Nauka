import turtle as t

t.lt(90)
t.speed(0)
def recur(n):
    t.fd(n)
    t.rt(90)
    if n>10: recur(n-30)
    t.lt(180)
    if n>10: recur(n-30)
    t.rt(90)
    t.bk(n)
recur(120)
t.done()