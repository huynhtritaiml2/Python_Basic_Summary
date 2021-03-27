########## REMOVING ALL OCCURANCES IN LIST ##########

backpack = ["pizza slice", "button", "pizza slice", "fishing pole",  "pizza slice", "nunchucks", "pizza slice", "sandwich from mcdonalds"]
backpack.remove("pizza slice") # ['button', 'pizza slice', 'fishing pole', 'pizza slice', 'nunchucks', 'pizza slice', 'sandwich from mcdonalds']
# we just only delete 1 "pizza slice"

while("pizza slice" in backpack): # This for loop modify backpack # LOGICAL ERROR
    backpack.remove("pizza slice") # ['button', 'fishing pole', 'nunchucks', 'sandwich from mcdonalds']

backpack = ["pizza slice", "button", "pizza slice", "fishing pole", "pizza slice", "nunchucks", "pizza slice", "pizza slice", "sandwich from mcdonalds"]
while("pizza slice" in backpack): # This for loop modify backpack # LOGICAL ERROR ******** THIS STILL WORK :)) FUCK NO ERROR
    backpack.remove("pizza slice") 
    # ['button', 'fishing pole', 'nunchucks', 'sandwich from mcdonalds']
    # In Youtube, it is ['button', 'fishing pole', 'nunchucks',"pizza slice", 'sandwich from mcdonalds']

print(backpack)

'''
You should also avoid modifying a list while iterating, so a for-in loop is bad
for item in backpack:
    if(item == "pizza slice"):
        backpack.remove(item)

Here is a better solution:
'''
backpack = ["pizza slice", "button", "pizza slice", "fishing pole", "pizza slice", "nunchucks", "pizza slice", "pizza slice", "sandwich from mcdonalds"]

for item in backpack[:]: #uses copy to keep index
    if item == "pizza slice":
        backpack.remove(item)

print(backpack)

#Here is a list comprehension version:
backpack = ["pizza slice", "button", "pizza slice", "fishing pole", "pizza slice", "nunchucks", "pizza slice", "pizza slice", "sandwich from mcdonalds"]
backpack[:] = [item for item in backpack if item != "pizza slice"]

print(backpack)

########## REVERSE LIST ##########


backpack = ["pizza slice", "button", "pizza slice", "fishing pole", "pizza slice", "nunchicks", "pizza slice", "sandwich from mcdonalds"]

print(backpack) # ['pizza slice', 'button', 'pizza slice', 'fishing pole', 'pizza slice', 'nunchicks', 'pizza slice', 'sandwich from mcdonalds']
backpack.reverse() # Reverse order in list, NOT in alphabet order 
print(backpack) # ['sandwich from mcdonalds', 'pizza slice', 'nunchicks', 'pizza slice', 'fishing pole', 'pizza slice', 'button', 'pizza slice']


########## SWAP AND REVERSE ALGORITHMS ##########

data = ["a", "b", "c", "d", "e", "f", "g", "h"]

for index in range(len(data) // 2): # NOTE: index = 0-> len//2
    data[index], data[-index-1] = data[-index-1], data[index]

print(data) # ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']

# METHOD 2:
########## REVERSED ITERATOR ##########
data = ["a", "b", "c", "d", "e", "f", "g", "h"]
data_reverse = []
for item in reversed(data):
    data_reverse.append(item)

print(data) # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
print(data_reverse) # ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']

#METHOD 3:
########## REVERSE USING SLICING ##########
'''
https://railsware.com/blog/python-for-machine-learning-indexing-and-slicing-for-lists-tuples-strings-and-other-sequential-types/
Có nhiều cách làm slicing khá nhiều :)) Coi đi 
'''
data = ["a", "b", "c", "d", "e", "f", "g", "h"]
data[:] = data[::-1]
print(data) # ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']

data = ["a", "b", "c", "d", "e", "f", "g", "h"]
print(data[::]) # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
print(data[::1]) # the same
print(data[::2]) # Step =2 # ['a', 'c', 'e', 'g']
print(data[::-1]) # Reverse list, step = 1 # ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
print(data[::-2]) # Reverse list, step = 2 # ['h', 'f', 'd', 'b']



