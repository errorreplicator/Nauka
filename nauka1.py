import turtle as t

def kwadrat(bok):
    t.fd(bok)
    if bok>50: kwadrat(bok/2)
    t.lt(90)
    t.fd(bok)
    t.lt(90)
    t.fd(bok)
    t.lt(90)
    t.fd(bok)
    t.lt(90)
    t.bk(bok/2)
    if bok>50: kwadrat(bok/2)
    t.fd(bok/2)
kwadrat(130)
t.done()
