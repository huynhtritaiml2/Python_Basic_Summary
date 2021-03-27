# Inherent
class Dog():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print(f"Hi, I am {self.name} and I am {self.age} years old")

    def talk(self):
        print("Bark!!!")

class Cat(Dog):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def talk(self): # This talk method of Cat.method override that of  Dog.method 
        print("Meow!!!")

jim = Dog("jim", 10)
jim.speak() # Hi, I am jim and I am 10 years old
jim.talk() # Bark!!!


tim = Cat("tim", 5, "blue")
tim.speak() # Hi, I am tim and I am 5 years old
tim.talk() # Meow!!!