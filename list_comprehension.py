# Removing from Lists using List Comprehension
healthy = ["kale chips", "broccoli"]
backpack = ["pizza", "frozen custard", "apple crisp", "kale chips"]

print(id(backpack)) # 140157422054848
#backpack = [item for item in backpack if item in healthy] # PROBLEM: NOTE: 2 id is not the same, we lost the backpack, because we referencing to other location
# SOLUTION:# this way we can modify the original list
backpack[:] = [item for item in backpack if item in healthy]
print(id(backpack)) # 140157422047552 
'''
140111138427072
140111138427072
['kale chips']
'''
print(backpack)


# 
squares = [i**2 for i in range(10)] # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares) 
# only square even number
squares = [i**2 for i in range(10) if i % 2 == 0] # [0, 4, 16, 36, 64]
print(squares) 

########## REMOVING ALL OCCURANCES IN LIST ##########

backpack = ["pizza slice", "button", "pizza slice", "fishing pole", 
"pizza slice", "nunchucks", "pizza slice", "sandwich from mcdonalds"]

backpack.remove("pizza slice")
print(backpack) #SO MUCH PIZZA!

while("pizza slice" in backpack):
    backpack.remove("pizza slice")

print(backpack)

#This may not be the most optimized solution as each removal requires
#an iteration from backpack.count. 
#You should also avoid modifying a list while iterating, so a for-in loop is bad

#for item in backpack:
#    if(item == "pizza slice"):
#        backpack.remove(item)

#The original solution is fine for removing data from reasonably sized lists

#Here is a better solution:
backpack = ["pizza slice", "button", "pizza slice", "fishing pole", 
"pizza slice", "nunchucks", "pizza slice", "pizza slice", "sandwich from mcdonalds"]

for item in backpack[:]: #uses copy to keep index
    if item == "pizza slice":
        backpack.remove(item)

print(backpack)

#Here is a list comprehension version:
backpack = ["pizza slice", "button", "pizza slice", "fishing pole", 
"pizza slice", "nunchucks", "pizza slice", "pizza slice", "sandwich from mcdonalds"]

backpack[:] = [item for item in backpack if item != "pizza slice"]

print(backpack)
