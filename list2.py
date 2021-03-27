greetings = ["hi", "hello", "wassap"]
print(greetings[2]) # wassap
print(len(greetings)) # 3
# n --> list size
# highest index = n - 1

# 
for item in greetings:
    print(item)

for i in range(len(greetings)):
    print(greetings[i]);


# 
backpack = ["sword", "rubber duck", "slice of pizza", "parachute", "sword", "sword"]

print(backpack.count("sword"))
if backpack.count("sword") < 5:
    backpack.append("sword")

# Insert
'''
list is dynamic array, so we insert new item, it will shift the RIGHT of inserted index all the elements to the RIGHT 1 index
'''
workdays = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
workdays.insert(2, "Wednesday") # ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
print(workdays)

# Remove
'''
list is dynamic array, so we insert new item, it will shift the RIGHT of inserted index all the elements to the LEFT 1 index
NOTE: we do need to know the position of item :))
'''
workdays.remove("Saturday") # ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(workdays)

# Del 
'''
NOTE: we know the position to remove
'''
workdays = ["Monday", "Tuesday", "Wednesday","Thursday", "Friday", "Saturday"]
del workdays[5] # == del workdays[-1] 
print(workdays)

# Del 2:
workdays = ["Monday", "Tuesday", "Wednesday","Thursday", "Friday", "Saturday"]
del workdays[workdays.index("Wednesday") : workdays.index("Wednesday") + 3] # 3 days off # ['Monday', 'Tuesday', 'Saturday']

workdays = ["Monday", "Tuesday", "Wednesday","Thursday", "Friday", "Saturday"]
del workdays[workdays.index("Wednesday"):] # ['Monday', 'Tuesday']
print(workdays)

########## REMOVE ELEMENT FROM LIST USING DEL AND SLICE ##########

print("########## REMOVE ELEMENT FROM LIST USING DEL AND SLICE ##########")
work_days = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]

del work_days[0:2] #remove first 2
print(work_days) # ['Thursday', 'Friday', 'Saturday']

work_days = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]

del work_days[-2:] #remove last 2 (start 2 from right and go to end)
print(work_days) # ['Monday', 'Tuesday', 'Thursday']

# Pop

workdays = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
popped = workdays.pop(4) # ['Monday', 'Tuesday', 'Thursday', 'Friday']
print("We just removed", popped) # We just removed Saturday
popped = workdays.pop(1) # ['Monday', 'Thursday', 'Friday']
print("We just removed", popped) # We just removed Tuesday
print(workdays)


########## REMOVING ALL OCCURANCES IN LIST ##########

backpack = ["pizza slice", "button", "pizza slice", "fishing pole", 
"pizza slice", "nunchucks", "pizza slice", "sandwich from mcdonalds"]

backpack.remove("pizza slice")
print(backpack) #SO MUCH PIZZA!

while("pizza slice" in backpack): # O(n) # ['button', 'fishing pole', 'nunchucks', 'sandwich from mcdonalds']
    backpack.remove("pizza slice")

print(backpack)