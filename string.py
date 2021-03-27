name = input("What is your name? ") # What is your name? Tai
name = name.strip() # Loại bỏ những space ở đầu và cuối string 
print("Hi, " + name + ", have a good day!!!") # Hi, Tai, have a good day!!!
print(f"Hi,  [{name}] have a good day!!!") # Hi,  [Tai] have a good day!!!

birth_year = int(input("What is your birth year? ")) # What is your birth year? 1996
print(f"Oh, {name}'s age is {birth_year}, so your age is {2021 - birth_year} ") # Oh, Tai's age is 1996, so your age is 25 
poem = ''' 
Hello every one,
Have a good day
Such a beatiful day  
Nice to see you
'''
print(poem)
course = "Python for Beginers"
print(course[:]) # Python for Beginers
print(course[-1]) # s
print(course[0:-2]) # Python for Begine
print(course[1:]) # ython for Beginers
print(course[:5]) # Pytho
print(type(course)) # <class 'str'>
print(len(course)) # 19
print(course.upper()) # PYTHON FOR BEGINERS
print(course.lower()) # python for beginers
print(course.title()) # Python For Beginers
print(course.find('P')) # 0
print(course.find('0')) # -1
print(course.find('o')) # 4 , vị trí chữ 'o' đầu tiên 
print(course.find("Beginers")) # 11 , là vị trí của chữ B
print(course.replace("Beginers", "Absolute Beginers")) # nó in ra Python for Absolute Beginers



str_2 = "split : input something into this"
print(str_2.split(' Tai')) # ['split', ':', 'input', 'something', 'into', 'this']