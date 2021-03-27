import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('seaborn')

data = pd.read_csv('data6.csv')
ages = data['Age']
dev_salaries = data['All_Devs']
py_salaries = data['Python']
js_salaries = data['JavaScript']


plt.plot(ages, py_salaries, label='Python')
plt.plot(ages, js_salaries, label='JavaScript')

plt.plot(ages, dev_salaries, color='#444444', linestyle='--', label='All Devs')

plt.legend()

plt.title('Median Salary (USD) by Age')
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')

plt.tight_layout()

plt.show()

###################################################
'''
fig, ax = plt.subplots()
print(ax) # AxesSubplot(0.125,0.11;0.775x0.77)
print(fig) # Figure(800x550
'''
###################################################
'''
fig, ax = plt.subplots(nrows=2, ncols=2)
print(ax) # 
#[[<AxesSubplot:> <AxesSubplot:>]
# [<AxesSubplot:> <AxesSubplot:>]]
print(fig) # Figure(800x550
'''
###################################################
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
print(ax1) # AxesSubplot(0.125,0.53;0.775x0.35)
print(ax2) # AxesSubplot(0.125,0.11;0.775x0.35)


print(fig) # Figure(800x550


ax1.plot(ages, py_salaries, label='Python')
ax2.plot(ages, js_salaries, label='JavaScript')

ax2.plot(ages, dev_salaries, color='#444444', linestyle='--', label='All Devs')

ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
ax1.set_xlabel('Ages')
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
ax2.set_title('Median Salary (USD) by Age')
ax2.set_xlabel('Ages')
ax2.set_ylabel('Median Salary (USD)')

plt.tight_layout()

plt.show()


###################################################
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
print(ax1) # AxesSubplot(0.125,0.53;0.775x0.35)
print(ax2) # AxesSubplot(0.125,0.11;0.775x0.35)


print(fig) # Figure(800x550


ax1.plot(ages, py_salaries, label='Python')
ax2.plot(ages, js_salaries, label='JavaScript')

ax2.plot(ages, dev_salaries, color='#444444', linestyle='--', label='All Devs')

ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
# ax1.set_xlabel('Ages') # Only need one labels
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
#ax2.set_title('Median Salary (USD) by Age')
ax2.set_xlabel('Ages')
#ax2.set_ylabel('Median Salary (USD)') # Only need one labels

plt.tight_layout()

plt.show()


###################################################
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
print(ax1) # AxesSubplot(0.125,0.53;0.775x0.35)
print(ax2) # AxesSubplot(0.125,0.11;0.775x0.35)


print(fig) # Figure(800x550


ax1.plot(ages, py_salaries, label='Python')
ax2.plot(ages, js_salaries, label='JavaScript')

ax2.plot(ages, dev_salaries, color='#444444', linestyle='--', label='All Devs')

ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
# ax1.set_xlabel('Ages') # Only need one labels
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
#ax2.set_title('Median Salary (USD) by Age')
ax2.set_xlabel('Ages')
#ax2.set_ylabel('Median Salary (USD)') # Only need one labels

plt.tight_layout()

plt.show()


################################################### Draw 2 separate graph 
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.plot(ages, py_salaries, label='Python')
ax2.plot(ages, js_salaries, label='JavaScript')

ax2.plot(ages, dev_salaries, color='#444444', linestyle='--', label='All Devs')

ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
# ax1.set_xlabel('Ages') # Only need one labels
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
#ax2.set_title('Median Salary (USD) by Age')
ax2.set_xlabel('Ages')
#ax2.set_ylabel('Median Salary (USD)') # Only need one labels

plt.tight_layout()

plt.show()


fig1.savefig("fig1.png")
fig2.savefig("fig2.png")