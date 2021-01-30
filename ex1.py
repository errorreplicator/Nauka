import turtle as t
def tree(dlug,wysokosc):
    t.fd(dlug)
    t.rt(20)
    if wysokosc>0: tree(dlug-10,wysokosc-1)
    t.lt(40)
    if wysokosc>0: tree(dlug-10,wysokosc-1)
    t.rt(20)
    t.bk(dlug)
t.lt(90)
tree(80,5)
t.done()



