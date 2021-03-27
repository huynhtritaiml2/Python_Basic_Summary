import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]
js_dev_y = [37810, 43515, 46823, 49293, 53437,
            56373, 62375, 66674, 68745, 68746, 74583]
plt.bar(ages_x, dev_y, color="#444444", label="All devs") 
plt.bar(ages_x, py_dev_y, color="#008fd5", label="Python") # linewidth=1
plt.bar(ages_x, js_dev_y, color="#e5ae38",label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
#plt.grid(True) 
#plt.tight_layout() 
plt.show()
################ Problem: 3 style stack in other, and we can see all data
############### SOLUTION:
import numpy as np

x_indexes = np.arange(len(ages_x))
width = 0.25

plt.bar(x_indexes - width, dev_y, width=width, color="#444444", label="All devs") 
plt.bar(x_indexes, py_dev_y, width=width, color="#008fd5", label="Python")
plt.bar(x_indexes + width, js_dev_y, width=width, color="#e5ae38",label="JavaScript")

plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.xticks(ticks=x_indexes, labels=ages_x) # Make sure

plt.legend()
plt.show()