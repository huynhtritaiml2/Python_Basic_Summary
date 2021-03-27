########## WORKING WITH 2D LISTS ##########

grades = [[6, 3, 5], [60, 43, 4, 23], [205]]

print(grades[0]) # [6, 3, 5]
print(type(grades[0])) #list # <class 'list'>

#Second element of first list in grades:
print(grades[0][1]) # 3
print(type(grades[0][1])) #int # <class 'int'>

#Lists are dynamic
grades.append([4]) # [[6, 3, 5], [60, 43, 4, 23], [205], [4]] #add an element, increasing len # 
print(grades)
grades.pop(1) # [[6, 3, 5], [205], [4]] #indexes rearranged # 
print(grades)
#grades.clear() #remove all.
print(len(grades))
print(len(grades[0]))
#Python keeps it simple in that there's not a static array type. Other languages often have an array and then list. 

# METHOD 1: some error 
#iterate through 2D list
# grades = [[6, 3, 5], [60, 43, 4, 23], 5, 3, [205]] # ERROR: because some element, have no sublist 
for inner_list in grades:
    for grade in inner_list:
        print(grade, end=" ")
    print()
'''
6 3 5 
205 
4 

ERROR: if
grades = [[6, 3, 5], [60, 43, 4, 23], 5, 3, [205]]

6 3 5 
60 43 4 23 
Error 
'''

# METHOD 2: ********************************************************************** Improved Method 1: 
grades = [[6, 3, 5], [60, 43, 4, 23], 5, 3, [205]]

for inner in grades:
    if  isinstance(inner, list): # if there is no sublist in list, then, print this 
        for grade in inner:
            print(grade, end=" ")
        print()
    else:
        print(inner)
'''
6 3 5 
60 43 4 23 
5
3
205 
'''

# METHOD 3:
#You can also iterate using range, no error if element is not a sublist 
grades = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]

for i in range(len(grades)):
    for j in range(len(grades[i])):
        print(grades[i][j], end=" ")
    print()

'''
1 2 3 
4 5 6 7 8 
9 
'''

######### FUNCTION TO PRINT LIST ##########
# METHOD 2: FUnction version 
def print_2d(passed_in):
    for inner in passed_in:
        if isinstance(inner, list):
            for data in inner:
                print(data, end=" ")
            print()
        else:
            print(inner)
    
print_2d(["chickity", "china", "chinese", "chicken", [1, 3, 3], [4]])

########## JOIN ##########


data = ["01001100", "01001111", "01001100"]

print(".".join(data)) # 01001100.01001111.01001100
print("-".join(data)) # 01001100-01001111-01001100


'''
Convert all data types from list into string and print purpose only :)) 
'''
data = [1, 0, 1, "2", "505"] # 1 0 1 2 505
print(" ".join(str(x) for x in data))
#100% original idea stolen from this guy
#https://stackoverflow.com/questions/3590165/join-a-list-of-items-with-different-types-as-string-in-python


########## SORTING 2D LIST ##########
data = [[10, 2, 3],[10, 20], [4, 5000, 6], [7, 8, 9], [10]]
print(sorted(data))

#behind the scenes a comparison is done. Here's some more practice
print([10] < [11]) #True
print([1, 10] < [1, 2]) #False
#What about this? #the shorter is considered less
print([5] < [5, -1]) #True ********************************************** SHORTER is LESS THAN 


########## Sorting by sum of list #########
#We can also sort using a different function to determine which comes first:

print(sorted(data, key=sum)) # [[10], [10, 2, 3], [7, 8, 9], [10, 20], [4, 5000, 6]]
print(sorted(data, key=sum, reverse=True)) # [[4, 5000, 6], [10, 20], [7, 8, 9], [10, 2, 3], [10]] #each list stays same, just order of list is flipped

########## CUSTOM KEY FUNCTION ##########
#Any fuction you could use on a list to manipulate the data should do the trick

def avg(data):
    avg = sum(data) / len(data)
    print(data, "average is", avg) #for our sake to see
    return avg

data = [[5, 5, 5], [3, 4, 5], [3, -3, 0], [1,1,1,79], [1, 10, 1, 20]]
print(sorted(data, key=avg)) # [[3, -3, 0], [3, 4, 5], [5, 5, 5], [1, 10, 1, 20], [1, 1, 1, 79]]










