x = ( 2 + 3)*10 -3
print(x) # 47

'''
parenthesis
expotentiation 2**3
multiplication or division * / , //: lấy nguyên, %: lấy dư 
addition or substraction + - 
'''
print(round(2.9)) # 3
print(round(2.4)) # 2
print(round(2.5)) # 2
print(abs(-3.6)) # 3.6
print(round(-3.6)) # -4
print(round(-3.5)) # -4
print(round(-3.4)) # 3


import math 
print(math.ceil(9.9)) # 10 , ceil: trần 
print(math.floor(9.9)) # 9 , fllor: sàn 

print(math.e) # 2.718281828459045
print(math.pi) # 3.141592653589793
print(math.tau) # 6.283185307179586 == 2*pi
print(math.inf) # inf
print(math.nan) # nan == Not A Number 
print(float('inf')) # inf
print(float('nan')) # nan == Not A Number 

n = 4
k = 3
# print(math.comb(n, k))# nCk  :::   n! / (k! * (n - k)!) Python 3.8
# print(math.perm(n, k))# nAk  :::   n! / (n - k)! Python 3.8

print(math.fabs(-3.9)) # 3.9, absulute of x == |x|
print(math.factorial(5)) # 5! = 1*2*3*4*5 = 120
print(math.fmod(125.3, 23.3)) # math.fmod work with float number better than % 
print(125.3 % 23.3) # Should use for Integer number
print(sum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])) # 0.9999999999999999
print(math.fsum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])) # 1.0 

'''
math.pow(x, y) # similar to 
math.exp(3) # Both better than 
math.e ** 3 
pow(math.e, 3)
'''
print(math.log(32)) # 36 = e^3.4657359027997265
print(math.log(32, 2)) # 32 = 2^5
'''
math.log2(x)
math.log10(x)
'''

math.sqrt(125) # square root of 125 = 5

'''
math.sin(x)
math.cos(x)
math.tan(x)
math.degrees(x)
math.radian(x)
'''
print(math.degrees(math.pi/2)) # 90.0
print(math.radians(90)) # 1.5707963267948966

'''
còn nhiều cái khó hiểu 
https://docs.python.org/3/library/math.html 
'''