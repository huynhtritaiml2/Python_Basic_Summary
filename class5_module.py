# note
'''
Khoong Khác mẹ gì cả :))) , Y chang :)))
'''
from class5_plublic_private_method import Parent, Children

parent = Parent("Tim", "16", "Blue")
parent = Parent("Tim", "16", "Blue")
print()
print(parent.var1)
print(parent._var2)


parent.displayPublic()
parent._displayProtected()


parent.accessPublic()
parent.accessProtected()
parent.accessPrivate()

children = Children("Jim", "25", "Green","Meow")
print(children.var1)
children.var1.append("Helllooooooo")
print(children.var1)
print(parent.var1)
print(children._var2)


children.displayPublic()
children._displayProtected()

children.accessPublic()
children.accessProtected()
children.accessPrivate()