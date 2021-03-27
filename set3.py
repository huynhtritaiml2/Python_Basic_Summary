######### SETS EXPLAINED #########
#Sets are similar to dictionaries in that the data is hashed and it is unordered.
#Sets are similar to lists in that they just contain the data and not a key-value pair
#Sets are different than lists in that you cannot have duplicates

stuff = {"sword", "rubber duck", "sice a pizza"}
print("sword" in stuff) # True 
print(stuff) # {'sword', 'rubber duck', 'sice a pizza'}
stuff.add("sword")
print(stuff) # {'sword', 'rubber duck', 'sice a pizza'}
#Notice only one occurance of sword even though already added
'''
#How is a set different than a dictionary?
#For a set, each element is only one piece of data
#for a dictionary, it is a key-value pair. 

#Behind the scenes, they both use hashing. The hashing is used to determine where to store the data.
#For dictionaries, the KEY is hashed
#for sets, we do not have a key, so the data itself is hashed.
#This means we cannot store something in sets that is not hashable. Ẽx: list can hashable 

#stuff.add(["trying to add a list"])

#It's important to understand the purpose of a set...
#Easily check if element in set
#such as to easily check to see if something has been tagged
#To do various set operations (coming soon)
'''
#An example would be to see if a word is ever used in a phrase. Not counted (that wold be a dictionary)

conjunctions = {"but", "or", "so", "and", "yet", "for", "nor"} #fanboys
seen = set() #THERE'S NOT AN EMPTY SET LITERAL!! #learn something new every day *********************************
completely_original_poem = """I still hear your voice when you sleep next to me
I still feel your touch in my dreams
Forgive me my weakness, but I don't know why
Without you it's hard to survive
'Cause every time we touch, I get this feeling
And every time we kiss I swear I could fly
Can't you feel my heart beat fast, I want this to last
Need you by my side"""

words = completely_original_poem.split()

for word in words:
    if str.lower(word) in conjunctions:
        seen.add(str.lower(word))

print(seen)


######### REMOVE DUPLICATES FROM LIST / CREATE SET FROM LIST ##########
#You can remove duplicate elements from a list by converting it to a set and back. 

colors = ["red", "red", "green", "green", "blue", "blue", "blue"]
print(id(colors), colors)

colors[:] = list(set(colors)) # Original data 
print(id(colors), colors)


#Earlier on in our life I showed some code to count each type of element in a list. 

colors = ["red", "red", "green", "green", "blue", "blue", "blue"]
counts = [[colors.count(item), item] for item in set(colors)]

print(counts)

#This works because is iterates through the set {"red", "green", "blue"} counting each in colors


######### UNION AND INTERSECTION #########

my_fav = {"red", "green", "black", "blue", "purple"}
her_fav= {"blue", "orange", "purple", "green"}

#union
all_favs = my_fav | her_fav
print(all_favs) #no repetition
#You may see + to combine lists, in which there are repeats.
#But we are not working with lists...so i'll try to focus here.

#intersection (elements shared between both)
wedding_colors = my_fav & her_fav
print(wedding_colors)
#this is like the inside section of a venn diagram

#There are also method versions:
all_favs = my_fav.union(her_fav)
print(all_favs)

wedding_colors = my_fav.intersection(her_fav)
print(wedding_colors)


######### DIFFERENCE AND SYMMETRIC DIFFERENCE ######### 


my_fav = {"red", "green", "black", "blue", "purple"}
her_fav= {"blue", "orange", "purple", "green"}

#Difference
only_my_colors = my_fav - her_fav
print(only_my_colors) #elements in left getting rid of all in right. 
#Could go other way too:
only_her_colors = her_fav - my_fav
print(only_her_colors)

#symmetric difference is like if you took colors only I liked union with colors only she liked and put em together:

symmetric = my_fav ^ her_fav
print(symmetric)

#This is like:
symmetric = only_my_colors | only_her_colors
print(symmetric)

#like union and intersection, there are method versions that return. --> .difference and .symmetric_difference