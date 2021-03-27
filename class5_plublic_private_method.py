# NOTe:
'''
variable of parent class and children class is the same. 
Thay đổi bên class con hay class cha thì variable of class 2 bên cũng thay đổi theo
Ngoài Private variable and private method ra, class children có thể  kế thừa từ class cha hết :))
'''
class Parent():
    var1 = ["pulic","data","member"]
    _var2 = "Protected data member".split()
    __var3 = "Private data member".split()

    def __init__(self, name, age, color):
        self.name = name
        self.age = age
        self.color = color


    def displayPublic(self):
        print("Public Method")
    
    def _displayProtected(self):
        print("Protected Method")

    def __displayPrivate(self):
        print("Private Method")

    
    def accessPublic(self):
        print("Public variable: ", self.var1)

    def accessProtected(self):
        print("Protected variable: ", self._var2)

    def accessPrivate(self):
        print("Private variable: ", self.__var3)

class Children(Parent):
    def __init__(self, name, age, color, speak):
        super().__init__(name, age, color)
        self.speak = speak


if __name__ == "__main__":

    parent = Parent("Tim", "16", "Blue")
    print(parent.var1)
    print(parent._var2)
    #print(parent.__var3) # Error

    print(Parent.var1)
    print(Parent._var2)
    #print(Parent.__var3) # Error

    parent.displayPublic()
    parent._displayProtected()
    # parent.__displayPrivate() # Error

    # Parent.displayPublic() # Error :)) Because parameter is self 
    # Parent._displayProtected()
    # Parent.__displayPrivate() # Error

    parent.accessPublic()
    parent.accessProtected()
    parent.accessPrivate()

    # Children
    children = Children("Jim", "25", "Green","Meow")
    print(children.var1)
    children.var1.append("Helllooooooo")
    print(children.var1)
    print(parent.var1)
    print(children._var2)
    print(children.name)
    print(children.age)
    print(children.color)
    print(children.speak)
    # print(parent.__var3) # Error

    children.displayPublic()
    children._displayProtected()
    # parent.__displayPrivate() # Error

    children.accessPublic()
    children.accessProtected()
    children.accessPrivate()