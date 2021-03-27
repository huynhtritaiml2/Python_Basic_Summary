import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter 

plt.style.use('fivethirtyeight')


#### METHOD 1: Standard library in python
with open("data.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file)

    language_counter = Counter()

    for row in csv_reader:
        language_counter.update(row['LanguagesWorkedWith'].split(";"))

    '''
    row = next(csv_reader) # Read the first row
    print(row) # {'Responder_id': '1', 'LanguagesWorkedWith': 'HTML/CSS;Java;JavaScript;Python'}
    print(row['LanguagesWorkedWith'].split(";")) # ['HTML/CSS', 'Java', 'JavaScript', 'Python']
    # Counter Class in standard python is good for count than List, diction, and count :)) *************
    '''
print(language_counter)
'''
Counter({'JavaScript': 59219, 'HTML/CSS': 55466, 'SQL': 47544, 'Python': 36443, 'Java': 35917, 'Bash/Shell/PowerShell': 31991, 
'C#': 27097, 'PHP': 23030, 'C++': 20524, 'TypeScript': 18523, 'C': 18017, 'Other(s):': 7920, 'Ruby': 7331, 'Go': 7201, 
'Assembly': 5833, 'Swift': 5744, 'Kotlin': 5620, 'R': 5048, 'VBA': 4781, 'Objective-C': 4191, 'Scala': 3309, 'Rust': 2794, 
'Dart': 1683, 'Elixir': 1260, 'Clojure': 1254, 'WebAssembly': 1015, 'F#': 973, 'Erlang': 777})
'''
print(language_counter.most_common(15))
'''
[('JavaScript', 59219), ('HTML/CSS', 55466), ('SQL', 47544), ('Python', 36443), ('Java', 35917), ('Bash/Shell/PowerShell', 31991), 
('C#', 27097), ('PHP', 23030), ('C++', 20524), ('TypeScript', 18523), ('C', 18017), ('Other(s):', 7920), ('Ruby', 7331), ('Go', 7201), 
('Assembly', 5833)]

'''

languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

print(languages)
'''
['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', 'Bash/Shell/PowerShell', 'C#', 'PHP', 'C++', 
'TypeScript', 'C', 'Other(s):', 'Ruby', 'Go', 'Assembly']
'''
print(popularity)
'''
[59219, 55466, 47544, 36443, 35917, 31991, 27097, 23030, 20524, 18523, 18017, 7920, 7331, 7201, 5833]
'''

plt.bar(languages, popularity) # PROBELM: Name to long, so we change axis

plt.title("Most popular Language")
plt.xlabel("Programming Languages")
plt.ylabel("Number of People Who Use")

plt.tight_layout()

plt.show()


##################### BETTER: SOLUTION
plt.barh(languages, popularity) # PROBELM: Name to long, so we change axis

plt.title("Most popular Language")
plt.xlabel("Number of People Who Use")
plt.ylabel("Programming Languages")

plt.tight_layout()

plt.show()

##################### BETTER: Show the most popular on TOP
languages.reverse() # reverse the list
popularity.reverse()
plt.barh(languages, popularity) 

plt.title("Most popular Language")
plt.xlabel("Number of People Who Use")
plt.ylabel("Programming Languages")

plt.tight_layout()

plt.show()