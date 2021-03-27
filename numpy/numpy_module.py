import numpy as np

arr = np.array([1, 2, 3])
print(arr) # [1 2 3] 
print(type(arr)) # <class 'numpy.ndarray'>
print(arr.shape) # (3,)

arr = np.array([[1,2,3],[4,5,6]])
print(arr)
'''
[[1 2 3]
 [4 5 6]]

NOTE: np.array printed in order, rows and columns
comapre to list :)) Fucking in one line 
'''
li = [[1,2,3],[4,5,6]]
print(li) # [[1, 2, 3], [4, 5, 6]]

arr = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
print(arr[0]) # [1 2 3 4]
print(arr[0][1]) # 2 # Cách này thì ổn
print(arr[0, 1]) # 2 
# print(arr[1:][0]) # Too many error : Output không lường trước được, tránh cách này 
print(arr[0:, 2:])
'''
[[ 3  4]
 [ 6  7]
 [ 9 10]]

NOTE: 0:2 là chỉ từ 0->1 :)) FUCK 
'''

arr = np.array([[1,2,3],[4,5,6]])
print(arr.shape) # (2, 3)
print(arr.size) # 6 : total number in np.array 

li = ["jhello", 1, True]
print(li) # ['jhello', 1, True]
arr = np.array(["jhello", 1, True])
print(arr) 
'''
['jhello' '1' 'True']
NOTE: all data in np.array must in same type, 
if not np.array() will change to string by default 
SHOULD the same type :)) Just remember it 
'''

arr = np.array([1,2,3])
print(arr.ndim) # 1

arr = np.array([[1,2,3],[4,5,6]])
print(arr.ndim) # 2

print(arr.data) # <memory at 0x7f527867bc70> Data at that address

print(arr[0]) # [1 2 3]
print(type(arr[0])) # <class 'numpy.ndarray'>
# arr[0].append(4) # Error # 'numpy.ndarray' object has no attribute 'append'

print(np.append(arr[0], 99)) # [ 1  2  3 99]
print(arr)
'''
[[1 2 3]
 [4 5 6]]
NOTE: np.append() return a new np.array, so it not change the original array
So, we need to 
'''
arr = np.append(arr[0], 99)
print(arr) # [ 1  2  3 99]

arr = np.delete(arr, 2) 
'''
# np.array, deleted index 
NOTE: Return a np.array


'''
print(arr) # [ 1  2 99]
