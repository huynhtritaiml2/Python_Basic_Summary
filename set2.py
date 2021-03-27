'''
1. SET is used for check item whether item exist or not
2. the order in SET do not matter for us :)) 
'''
backpack = ["sword", "rubber duck", "slice of pizza", "parachute", "sword", "sword"]
backpack2 = {"sword", "rubber duck", "slide of pizza", "parachute"}

print(backpack2)
print("sword" in backpack2) # True

# CASE: Count number element in list
backpack = ["sword", "rubber duck", "slice of pizza", "parachute", "sword", "sword"]
counts = [backpack.count(item) for item in backpack] # [3, 1, 1, 1, 3, 3] : PROBLEM: comprehension count each item multiple times
# SOLUTION
counts = [backpack.count(item) for item in set(backpack)] # [3, 1, 1, 1] : PROBLEM: set NOT in orders
# SOLUTION 2: list, set, dic
counts = [[backpack.count(item), item] for item in set(backpack)] # [[1, 'slice of pizza'], [1, 'parachute'], [3, 'sword'], [1, 'rubber duck']]
counts = [(backpack.count(item), item) for item in set(backpack)] # [(1, 'slice of pizza'), (1, 'rubber duck'), (1, 'parachute'), (3, 'sword')]
counts = [{backpack.count(item): item} for item in set(backpack)] # [{1: 'parachute'}, {1: 'rubber duck'}, {3: 'sword'}, {1: 'slice of pizza'}]
print(counts)

# CASE 2: Count number element. THE BEST WAY
from collections import Counter
counts = Counter(backpack) # Counter({'sword': 3, 'rubber duck': 1, 'slice of pizza': 1, 'parachute': 1})
print(counts) 


