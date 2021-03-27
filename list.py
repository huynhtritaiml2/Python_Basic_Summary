# Mutable = changeable 
# ordered 
numbers = [5, 6, 8, 9, 6, 1, 4, 3, 5, 5]
numbers.append(20)
numbers.insert(0,10)
numbers[0:0] = 10 # insert at position 0
numbers.remove(6) # loại bỏ số 6 đầu tiên 
numbers.pop() # in ra số cuối và remove it
for number in numbers:
    print(number , '*' * number)

print(numbers.index(8)) # 2 : Vị trí thứ 2 
print(numbers.count(5)) # 3
print(50 in numbers) # False 
numbers.sort() # Tăng dần 
numbers.reverse() # Tăng dần sau đó giảm dần 
print(numbers)
numbers2 = numbers.copy()
print(numbers2)
numbers.clear()
print(numbers)
#number.
print(type(numbers)) # <class 'list'>




'''
NOTE:
1. 'list' object has no attribute 'size
'''