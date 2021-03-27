class Dog():
    dogs = [] # Variable of class and instant,
    xc = 4

    def __init__(self, name):
        self.name = name 
        self.dogs.append(self)

    @classmethod
    def num_dogs(cls):
        return len(cls.dogs)

    @staticmethod
    def bark(n):
        for _ in range(n):
            print("Bark!")

tim = Dog("tim")
jim = Dog("jim")

print(tim.dogs) # [<__main__.Dog object at 0x7f6d3489bdf0>, <__main__.Dog object at 0x7f6d348032b0>]
# 2 DOG OBJECT 
print(Dog.num_dogs()) # 2 
print(tim.num_dogs()) # 2 