########## SPLIT ##########

msg = "Pay attention to each word that I say..." 
words = msg.split() #returns list # ['Pay', 'attention', 'to', 'each', 'word', 'that', 'I', 'say...']
print(words)

msg = "this,is,important,data" #How to parse CSV
print(msg.split(",")) # ['this', 'is', 'important', 'data']

########## SPLIT STRING BY LINE ##########


#This may have came from a file, for example.
msg = """\
Hey there.
My name is Caleb!
What's your name?
You're name is NERD? Weird...
Bye for now!"""

print(msg) #to see how it is stored...
'''
Hey there.
My name is Caleb!
What's your name?
You're name is NERD? Weird...
Bye for now!
'''

print(msg.split('\n'))
# ['Hey there.', 'My name is Caleb!', "What's your name?", "You're name is NERD? Weird...", 'Bye for now!']

########## INPUT USING SPLIT ##########

print("List your favorite foods separated by ', '")
print("Example input: ")
print("Kale, bok choy, brussel sprouts")

foods = input().split(', ')

for food in foods:
    print("You said " + food)
'''
Kale, bok choy, brussel sprouts
You said Kale
You said bok choy
You said brussel sprouts
'''

########## LOOPING TO GET USER INPUT ##########

fav_foods = []
while True:
    print("Enter a food. q to quit: ", end="")
    fav = input()
    if str.lower(fav) == 'q':
        break
    fav_foods.append(fav)

print("all foods:", fav_foods)
'''
Enter a food. q to quit: food
Enter a food. q to quit: feet
Enter a food. q to quit: fen
Enter a food. q to quit: fiin
Enter a food. q to quit: q
all foods: ['food', 'feet', 'fen', 'fiin']
'''

########## LIST AS STACK ##########
stack = []
stack.append("added")
#and removing from the end of the list
stack.pop() 

fav_foods = []
while True:
    print("Enter a food. q to quit, r to remove: ", end="")
    fav = input()
    if str.lower(fav) == 'q':
        break
    if str.lower(fav) == 'r':
        popped = fav_foods.pop()
        print("removed " + popped)
        print("all foods:", fav_foods)
        continue
    fav_foods.append(fav)
    print("all foods:", fav_foods)

print("final foods:", fav_foods)
'''
Enter a food. q to quit, r to remove: banana
all foods: ['banana']

Enter a food. q to quit, r to remove: apple
all foods: ['banana', 'apple']

Enter a food. q to quit, r to remove: tiger
all foods: ['banana', 'apple', 'tiger']

Enter a food. q to quit, r to remove: single
all foods: ['banana', 'apple', 'tiger', 'single']

Enter a food. q to quit, r to remove: r
removed single
all foods: ['banana', 'apple', 'tiger']

Enter a food. q to quit, r to remove: r
removed tiger
all foods: ['banana', 'apple']

Enter a food. q to quit, r to remove: r
removed apple
all foods: ['banana']

Enter a food. q to quit, r to remove: q
final foods: ['banana']
'''

##########  LIST AS QUEUE ##########
#The data structure depends on adding to the end of a list:
stack = []
stack.append("added")
#and removing from the FRONT of the list
stack.pop(0) #remove index 0

fav_foods = []
while True:
    print("(QUEUE) Enter a food. q to quit, eat to remove: ", end="")
    fav = input()
    if str.lower(fav) == 'q':
        break
    if str.lower(fav) == 'eat':
        popped = fav_foods.pop(0)
        print("removed " + popped)
        print("all foods:", fav_foods)
        continue
    fav_foods.append(fav)
    print("all foods:", fav_foods)

print("final foods:", fav_foods)
'''
(QUEUE) Enter a food. q to quit, eat to remove: banana
all foods: ['banana']
(QUEUE) Enter a food. q to quit, eat to remove: apple
all foods: ['banana', 'apple']
(QUEUE) Enter a food. q to quit, eat to remove: tiger
all foods: ['banana', 'apple', 'tiger']
(QUEUE) Enter a food. q to quit, eat to remove: single
all foods: ['banana', 'apple', 'tiger', 'single']
(QUEUE) Enter a food. q to quit, eat to remove: eat
removed banana
all foods: ['apple', 'tiger', 'single']
(QUEUE) Enter a food. q to quit, eat to remove: eat
removed apple
all foods: ['tiger', 'single']
(QUEUE) Enter a food. q to quit, eat to remove: q
final foods: ['tiger', 'single']
'''