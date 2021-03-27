# Unique element 
# Unorder 
d = {"apple":4, "pear": 3}
print(d["apple"]) # 4

# NOTE: We cannot use index d[0]
# we can add many type of data into dictionary 
d[(2,3)] = 5
print(d) # {'apple': 4, 'pear': 3, (2, 3): 5}

# Note: The way to change name of key/element
v = d["apple"]
d["newName"] = v 
del d["apple"]
print(d) # {'pear': 3, (2, 3): 5, 'newName': 4}

d.clear()
print(d) # {} 



# Ex: count number character appear in the string 
s = "hellomynameis"

count = {}
for ch in s:
    if ch in count:
        count[ch] = count[ch] + 1 
    else:
        count[ch] = 1
print(count) # {'h': 1, 'e': 2, 'l': 2, 'o': 1, 'm': 2, 'y': 1, 'n': 1, 'a': 1, 'i': 1, 's': 1}

for i in count:
    print(i)
'''
h
e
l
o
m
y
n
a
i
s
'''

for i in count.items():
    print(i)


'''
('h', 1)
('e', 2)
('l', 2)
('o', 1)
('m', 2)
('y', 1)
('n', 1)
('a', 1)
('i', 1)
('s', 1)
'''

for val in count.values():
    print(val)
'''
1
2
2
1
2
1
1
1
1
1
'''
for i, val in count.items():
    print(i, val)
'''h 1
e 2
l 2
o 1
m 2
y 1
n 1
a 1
i 1
s 1
'''

# Ex: Nhập số và in ra chữ
'''
input_str = input("Number: ")
digit = {
    '1' : "One",
    '2' : "Two",
    '3' : "Three",
    '4' : "Four"
}

for ch in input_str:
    print(digit[ch])
'''

