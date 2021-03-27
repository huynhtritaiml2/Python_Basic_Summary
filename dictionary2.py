######### INTRO TO DICTIONARIES #########
#The equiv of an associative array/hash table
emails = {
    "Caleb":"caleb@email.com", 
    "Gal":"g@example.com" 
}

print(emails)
#in this case, the key is the name, email is the value. 
#Data type doesn't matter at all for the value.
#Key must be a hashable --> What does this mean?
#Classes have a function __hash__ invoked when used as the key
print(hash("hello"))  # -2799942763559748738

#I'm not sure of exact internals on how the hash is used, but imagine it like so:
#You have an area of memory with 8 spots, and you need to store the value at some spot...
print(hash("hello") % 8) # 6


#Why use a hashtable? Extremely fast to add or look up data
#O(1) --> constant time. More elements does not mean slower unlike a sort or something
#https://en.wikipedia.org/wiki/Hash_table
#Hashtables often used for memoization

# NOTE: list cannot hashable, tupple, set and dict can hash 

######### RETRIEVE DATA FROM DICTIONARY ##########


print(list(emails))
print(sorted(emails))
#print(emails[0]) # NOPE!
print(emails["Caleb"])


if("Caleb" in emails):
    print("Emailing", emails["Caleb"])

#but this is different than a list where we iterate to find the element 
#The key is goes through hash to calculate index. Either there or not
#This is O(1)

#You may still not like the casing and in that situation there is a method

print(emails.get("Ryan")) #NONE #Returns none if not found
print(emails.get("Ryan", "Not found")) #Not found #optional return arg if not found


######### ADD DATA TO DICTIONARY ######### **********************************

#How to add data (3 ways here):
#METHOD 1: indexing
emails["josh"] = "josh@j.com"
print(emails)

#METHOD 2:
#update function
emails.update( {"josh": "evennewer@email.com"})
print(emails)

#METHOD 3:
#Weird variation
emails.update(josh = "new@email.com")
print(emails) # {'Caleb': 'caleb@email.com', 'Gal': 'g@example.com', 'josh': 'new@email.com'}

#Key must be hashable
emails[5] = "test"
emails[(1, 2)] = "yep"
#emails[[5, 3]] = "nope" #list is not hashable (mainly cuz mutable)


######### LOOPING THROUGH KEYS #########
#dictionary is an iterable (implements __iter__)

emails = {
    "Caleb":"caleb@email.com", 
    "Gal":"g@example.com",
    "Ted": "talk@gmail.com"
}

#k is a variable but k by convention for key
for k in emails:
    print(k)

#You can use the key to get the element  Not ideal.
#One reason being the key has to be hashed to get the value associated with it.
#(but will show better way in next section)
for k in emails:
    print("index", k, "is", emails[k])


######### LOOPING THROUGH KEY-VALUE PAIRS #########

#In the prev section we used the index with []. Although it works, you can do this:

for k, elem in emails.items():
    print(k, elem)

#Each iteration k will be the key and elem will be the item found at this key.

#As an example of what a hashtable can be used for, you can keep track of occurances:

conjunctions = {"but": 0, "or": 0, "so": 0, "and": 0, "yet": 0, "for": 0, "nor": 0} #fanboys

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
        conjunctions[str.lower(word)] += 1

print(conjunctions) {'but': 1, 'or': 0, 'so': 0, 'and': 1, 'yet': 0, 'for': 0, 'nor': 0}

#This could easily be wrapped in a function to take a msg and words to look for, returning a dict
#concept can be used to analyze documents to quantify how vulgar they are, search for phrases, etc
#dictionaries can be used to keep track of values that are hard to calculate (memoization)