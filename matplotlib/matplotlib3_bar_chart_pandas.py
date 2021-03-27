import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter 

import pandas as pd 

plt.style.use('fivethirtyeight')

data = pd.read_csv("data.csv") # Shorter than previous and return a Dataframe
ids = data["Responder_id"]
lang_responses = data["LanguagesWorkedWith"]

print(data)
print(type(data)) # <class 'pandas.core.frame.DataFrame'>
print(ids)
print(lang_responses)
print(type(lang_responses)) # <class 'pandas.core.series.Series'>

language_counter = Counter()

for row in lang_responses:
    language_counter.update(row.split(";"))



languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])


languages.reverse() # reverse the list
popularity.reverse()
plt.barh(languages, popularity) 

plt.title("Most popular Language")
plt.xlabel("Number of People Who Use")
plt.ylabel("Programming Languages")

plt.tight_layout()

plt.show()