
########## SORT METHOD ########## SORT in Alphabet 
# METHOD 1: CHANGE THE ORIGINAL LIST when SORT

#This is using python built in method and not custom built
work_days = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]

work_days.sort() # ['Friday', 'Monday', 'Saturday', 'Thursday', 'Tuesday']

print(work_days) 
# NOTE: Remember...no return. Modifies list

numbers = [1, 11, 115, 13, 1305, 43]
print(numbers)
# PROBELM: This is not possible because no return. 
# print(numbers.sort())
# SOLUTION: 
numbers.sort() # Sort in smaller and larger number # [1, 11, 13, 43, 115, 1305]
print(numbers)

# METHOD 2: keep original list
########## SORTED - COPY A LIST FOR SORT ##########
numbers = [1, 11, 115, 13, 1305, 43]
numbers_sorted = sorted(numbers)
numbers = sorted([1, 11, 115, 13, 1305, 43]) # the same


print(numbers) #original list # [1, 11, 115, 13, 1305, 43]
print(numbers_sorted) # [1, 11, 13, 43, 115, 1305]

# Sort and reverse, Larger to smaller
numbers = [1, 11, 115, 13, 1305, 43]
numbers_sorted = sorted(numbers)
sorted(numbers, reverse=True) # the same 
numbers_sorted.reverse()
print(numbers_sorted) # [1305, 115, 43, 13, 11, 1]

########## CASE INSENSITIVE SORT ##########
'''
When working with strings, 'a' and 'A' are different.
A < AA
A < a
'''
letters = ['a', 'A', 'abc', 'ABC', 'aBc', 'aBC', 'Abc']

letters_sort = sorted(letters)
print(letters_sort) # ['A', 'ABC', 'Abc', 'a', 'aBC', 'aBc', 'abc']

print(sorted(letters, key=str.lower)) # ['a', 'A', 'abc', 'ABC', 'aBc', 'aBC', 'Abc']
# note: they are in order from original list if they are not differen in lower case.

########## SORT BY STRING LENGTH ##########
'''
sort by function
'''
random = ["a", "A", "aa", "AAA", "HELLO", "b", "c", "a"]
print(sorted(random, key=len))
#no () on function! len refers to function. len() invokes function



########## SORT NUMBERS WITH LEXICOGRAPHICAL SORTING ##########
'''
We can sort numbers similar to strings 1, 11, 111, 2, 22, 222
Basically, when we are working with strings, 
"111" < "12"  because we compare by character left to right, So we can cast each to a str using the str constructor 
'''
numbers = [1, 54, 76, 12, 111, 4343, 6, 8888, 3, 222, 1, 0, 222, -1, -122, 5, -30]
print(sorted(numbers, key=str))


########## COMPARE NUMERICALLY ##########
'''
Just like we compared using strings in the previous section, we can do it with numbers
Talse is 0
True is 1
expression evaluates to true and maintains that value
No data is converted to a float in list. 
Strings are still strings. 
bools are bools.
'''

age = 5
stuff = [True, False, 0, -1, "0", "1", "10", age < 30, "20", "2", "5", "9001", "5.5", "6.0", 6]
print(sorted(stuff, key=float)) # [-1, False, 0, '0', True, '1', True, '2', '5', '5.5', '6.0', 6, '10', '20', '9001']




