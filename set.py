# Mutable 
# Un-Ordered == cannot use index to accesss the element == VERY FAST because hash table 
# Unique element == key == Two element cannt be the same name 
s = {1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5}
print(s) # {1, 2, 3, 4, 5}

s.add(6)
s.add("str")
s.add(-8)
print(s) # {1, 2, 3, 4, 5, 6, -8, 'str'} or sometimes {1, 2, 3, 4, 5, 6, 'str', -8}
s.remove(5)
print(s) # {1, 2, 3, 4, 6, 'str', -8}

# Find the element VERY FAST compare to List 
# for list
l = [x for x in range(100)]

searchFor = 98 
for i, ele in enumerate(l): # O(n) : too long 
    if ele == searchFor:
        print("True")
        break# the number is in list ??? NOT find the index 

# for set 
# -> We use set to find the item is existed, we donot care How many times it exist, Donot care what order it come in 
s = {x for x in range(100)}
searchFor in s # O(1)
s.add(5) # O(1)
s.remove(1) # O(1)

# Note: set(l) # O(n)
dup = len(set(l)) - len(l)

s = {1, 2, 3, 7, 8, "test"}
t = {3, 4, 5, 9, 10, "test" } 
print(s.intersection(t)) # {3, 'test'}
ss = {1, 2, 3}
tt = {1, 2}
print(ss.issubset(tt)) # ss is a subset/Children of tt # False
print(tt.issubset(ss)) # tt is a subset/Children of ss # True
print(ss.issuperset(tt)) # ss is a issuperset/Parent of tt # True
print(tt.issuperset(ss)) # tt is a issuperset/Parent of ss # False
print(s.difference(t)) # {8, 1, 2, 7}
print(t.difference(s)) # {9, 10, 4, 5}
print(t.symmetric_difference(s)) # {1, 2, 4, 5, 7, 8, 9, 10}
h = s.copy()
s.remove("test")
print(s) # {1, 2, 3, 7, 8}
print(h) # {1, 2, 3, 7, 8, 'test'}