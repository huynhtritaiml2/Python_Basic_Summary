import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
import pandas as pd 

plt.style.use('fivethirtyeight')

'''
Pie chart is bad for more than 5 categories
'''

slices = [60,40,30,20] # Not need to be 100, plt.pie calculate percentage for us
labels = ["Sixty", "Forty", "Extra1", "Extra2"]
colors = ["blue", "red", "yellow", "green"]
#colors = ["#008fd5", "#fc4f30"]
plt.pie(slices, labels=labels, colors=colors, wedgeprops={"edgecolor": "black"})
# https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.patches.Wedge.html : WEDGE 

plt.title("My Awesone Pie Chart")
plt.tight_layout()
plt.show()

############################## 

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917, 31991, 27097, 23030, 20524, 18523, 18017, 7920, 7331, 7201, 5833]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', 'Bash/Shell/PowerShell', 'C#', 'PHP', 'C++', 'TypeScript', 'C', 'Other(s):', 'Ruby', 'Go', 'Assembly']

expode = [0, 0, 0, 0.5, 0]
plt.pie(slices, labels=labels, wedgeprops={"edgecolor": "black"})
plt.title("My Awesone Pie Chart")
plt.tight_layout()
plt.show()



##############################  NOT more than 5 categories

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']

explode = [0, 0, 0, 0.5, 0]
plt.pie(slices, labels=labels, explode=explode, wedgeprops={"edgecolor": "black"})
plt.title("My Awesone Pie Chart")
plt.tight_layout()
plt.show()



##############################  

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']

explode = [0, 0, 0, 0.1, 0]
plt.pie(slices, labels=labels, explode=explode, shadow = True, startangle=90,
        wedgeprops={"edgecolor": "black"})
plt.title("My Awesone Pie Chart")
plt.tight_layout()
plt.show()


##############################  

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']

explode = [0, 0, 0, 0.1, 0]
plt.pie(slices, labels=labels, explode=explode, shadow = True, startangle=90,
        autopct="%1.1f%%",
        wedgeprops={"edgecolor": "black"})
plt.title("My Awesone Pie Chart")
plt.tight_layout()
plt.show()