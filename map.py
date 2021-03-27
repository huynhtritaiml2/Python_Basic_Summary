
# C1: map and list

li = [1, 2, 3 ,4 ,5, 6 ,7 ,8 ,9, 10]

def func(x):
    return x**x

print(list(map(func, li))) # [1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000]
'''
# if we want to apply 2 function we can use 
def func(x):
    return func2(x**x)

or 
'''
# C2: List comprehension

print([func(x) for x in li]) # [1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000]
print([func(x) for x in li if x%2==0]) # [4, 256, 46656, 16777216, 10000000000]

# C3: Bad way


newList = []
for x in li:
    newList.append(func(x))

print(newList) # [1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000]