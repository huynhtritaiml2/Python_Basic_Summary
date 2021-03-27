# 1. Optinal Paremeter là default value cho paremeter khi khai báo function :)) Too Eassy man

# 2. Static and Class Methods
class person():
    population = 50
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod # Class.method can access this method without creating a instance 
    # @classmethod == method which the class can use without need to create a new intance
    # Note: class method donot need self, self is used by class intances
    def getPopulation(cls):
        cls.population = cls.population  +1 
        return cls.population
    
    @staticmethod 
    # Note: @staticmethod do not have self, or cls parameters, -> so intances cannot use this 
    # For organize the method ONLY 
    # Note: this method cannot access ANYTHING in class, because it do not have cls and self
    # -> so we only use the paremeter which pass into the function 
    def isAdult(age):
        return age >= 18
    
    def display(self):
        print(f"{self.name} is {self.age} years old")

newPerson = person("tim", 18)
print(person.population)
print(newPerson.population)

print(person.getPopulation())
# print(newPerson.getPopulation()) # Error: getPopulation() takes 1 positional argument but 2 were given

#print(person.getPopulation(person))
# print(newPerson.getPopulation(person)) # Error: getPopulation() takes 1 positional argument but 2 were given

print(person.isAdult(20))
print(newPerson.isAdult(20)) # Error : isAdult() takes 1 positional argument but 2 were given

# print(person.display()) # Error: display() missing 1 required positional argument: 'self'
print(newPerson.display()) # tim is 18 years old
