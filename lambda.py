def func(x):
    return x + 5

func2 = lambda x: x+5
print(func(2)) # 7
print(func2(2)) # 7

####
def func3(x):
    func4 = lambda x: x+5
    return func4(5) + 85

print(func3(5)) # 95

func4 = lambda x, y=4: x + y #Note: y=4 is optinal parameter 
print(func4(5)) # 9


#####
li = [1, 2, 3 ,4, 5, 6 ,7, 8, 9, 10]

b = list(map(lambda x: x+4, li))
print(b) # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
c = list(filter(lambda x: x%2==0, li))
print(c) # [2, 4, 6, 8, 10]