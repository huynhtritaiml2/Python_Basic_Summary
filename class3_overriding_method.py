# 
class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.coords = (self.x, self.y)

    def move(self, x, y):
        self.x += x
        self.y += y

    def __add__(self, p): # p is Point object
        # return self.x + p.x, self.y + p.y # Note: this return a Tupple
        return Point(self.x + p.x, self.y + p.y)
    
    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, p):
        return self.x * p.x + self.y * p.y

    def __str__(self): 
        # Every time we change Point object to string, 
        # object will call this method
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"__repr__ : ({self.x}, {self.y})"

    def length(self):
        import math
        return math.sqrt(self.x**2 + self.y**2)
    
    def __gt__(self, p):
        return self.length() > p.length()
    
    def __ge__(self, p):
        return self.length() >= p.length()

    def __lt__(self, p):
        return self.length() < p.length()

    def __le__(self, p):
        return self.length() <= p.length()

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    def __call__(self):
        print(f"You have called this function :")

    def __len__(self):
        return self.x + self.y

p1 = Point(3, 4)
p2 = Point(3, 2)
p3 = Point(1, 3)
p4 = Point(0, 1)

p5 = p1 + p2
print(p5) # <__main__.Point object at 0x7fe7ef031340>
# Note: p1 == self, p2 == p 
# (6, 6) After declare __str__ method

p6 = p4 - p2
print(p6) # (-3, -1) After declare __str__ method

p7 = p2 * p3
print(p7)

print(p1 == p2) # False
print(p1 > p2) # True 
print(p4 <= p3) # True

'''
str() is used for creating output for end user while repr() is mainly used for debugging and development. 
repr’s goal is to be unambiguous and str’s is to be readable. 
For example, if we suspect a float has a small rounding error, repr will show us while str may not.
https://www.geeksforgeeks.org/str-vs-repr-in-python/
'''

# Dunder method / Magic Method 

p1() # You have called this function :
print(len(p1))