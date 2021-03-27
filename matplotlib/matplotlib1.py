import matplotlib.pyplot as plt 

# Median Developer Salaries by Age
dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] # list


dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

plt.plot(dev_x, dev_y)
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.show()

# Median Python Developer Salaries by Age
py_dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(py_dev_x, py_dev_y)
plt.title("Median Python Developer Salaries (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.show()

############## SIMILAR, Two line 
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x, dev_y)
plt.plot(ages_x, py_dev_y)
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.legend(["All devs", "Python"]) # PROBLEM: We must put the names in order :((

plt.show()

############################### BETTER VERSION: SOLUTION

ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x, dev_y, label="All devs")
plt.plot(ages_x, py_dev_y, label="Python")
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.legend()

plt.show()


'''
Format string:
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
'''
############################### BETTER VERSION: SOLUTION

ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x, dev_y, "k--", label="All devs")
plt.plot(ages_x, py_dev_y, "b", label="Python")
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.legend()

plt.show()

############################### BETTER VERSION: SOLUTION

ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x, dev_y, color="k", linestyle="--", marker=".", label="All devs")
plt.plot(ages_x, py_dev_y, color="b", marker="o", label="Python")
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.legend()

plt.show()

############################### BETTER VERSION: SOLUTION

ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x, dev_y, color="#444444", linestyle="--", marker=".", label="All devs") # Add Color by number 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", marker="o", label="Python") # Add color by Hexdecimal
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')

plt.legend()

plt.show()




# Median JavaScript Developer Salaries by Age
js_dev_y = [37810, 43515, 46823, 49293, 53437,
            56373, 62375, 66674, 68745, 68746, 74583]


plt.plot(ages_x, dev_y, color="#444444", linestyle="--", marker=".", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", marker="o", linewidth=3, label="Python") # linewidth=1
plt.plot(ages_x, js_dev_y, color="#adad3b", marker="o", linewidth=3,label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
plt.show()

#################### BETTER:
print(plt.style.available)
'''
['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 
'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 
'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
'''
plt.style.use('fivethirtyeight')

plt.plot(ages_x, dev_y, color="#444444", linestyle="--", marker=".", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", marker="o", linewidth=3, label="Python") 
plt.plot(ages_x, js_dev_y, color="#adad3b", marker="o", linewidth=3,label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
plt.grid(True) # grid should above tight_layout()
plt.tight_layout() # Plot better only each size of screen
plt.show()

#################### : change style
plt.style.use('ggplot')

plt.plot(ages_x, dev_y, color="#444444", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", label="Python") 
plt.plot(ages_x, js_dev_y, color="#adad3b", label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
#plt.grid(True) # grid should above tight_layout()
plt.tight_layout() # Plot better only each size of screen
plt.show()

#################### stype by method
#plt.style.use('ggplot')
plt.xkcd() # Similar to comic, manga 

plt.plot(ages_x, dev_y, color="#444444", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", label="Python") 
plt.plot(ages_x, js_dev_y, color="#adad3b", label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
#plt.grid(True) # grid should above tight_layout()
plt.tight_layout() # Plot better only each size of screen
plt.show()


#################### Save Graph
#plt.style.use('ggplot')
plt.xkcd() # Similar to comic, manga 

plt.plot(ages_x, dev_y, color="#444444", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", label="Python") 
plt.plot(ages_x, js_dev_y, color="#adad3b", label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
#plt.grid(True) # grid should above tight_layout()
plt.tight_layout() # Plot better only each size of screen

plt.savefig("plot.png")
plt.show()




# Ages 18 to 55
ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666,
            84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708, 108423, 101407, 112542, 122870, 120000]

js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000,
            78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000, 91660, 99240, 108000, 105000, 104000]

dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752, 77232,
         78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988, 100000, 108923, 105000, 103117]


#print(plt.style.available)
#plt.style.use('ggplot')
plt.xkcd() # Similar to comic, manga 

plt.plot(ages_x, dev_y, color="#444444", label="All devs") 
plt.plot(ages_x, py_dev_y, color="#5a7d9a", label="Python") 
plt.plot(ages_x, js_dev_y, color="#adad3b", label="JavaScript") 
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD')
plt.legend()
#plt.grid(True) # grid should above tight_layout()
plt.tight_layout() # Plot better only each size of screen

plt.savefig("plot.png")
plt.show()
