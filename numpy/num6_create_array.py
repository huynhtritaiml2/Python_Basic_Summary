import numpy as np

'''
np.array : create array from list or tupple

'''

# np.array
a = np.array([1,2,3,4])
print(a) # [1 2 3 4]
a = np.array([[1,2],[3,4]])
print(a)
'''
[[1 2]
 [3 4]]
'''

a = np.array([1,2,3,4], dtype="complex")
print(a)
'''
[1.+0.j 2.+0.j 3.+0.j 4.+0.j]
'''

# Arange()
a = np.arange(1, 11)
print(a) # [ 1  2  3  4  5  6  7  8  9 10]

a = np.arange(1, 11.)
print(a) # [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

a = np.arange(1, 11, dtype='float')
print(a) # [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

a = np.arange(1, 11., 2)
print(a) # [1. 3. 5. 7. 9.]



a = np.arange(20, dtype='complex')
print(a)
'''
[ 0.+0.j  1.+0.j  2.+0.j  3.+0.j  4.+0.j  5.+0.j  6.+0.j  7.+0.j  8.+0.j
  9.+0.j 10.+0.j 11.+0.j 12.+0.j 13.+0.j 14.+0.j 15.+0.j 16.+0.j 17.+0.j
 18.+0.j 19.+0.j]
'''


# zeros() and ones()

a = np.zeros(5)
print(a) # [0. 0. 0. 0. 0.]
print(a.shape) # (5,)

a = np.zeros(5, dtype = "int")
print(a) # [0 0 0 0 0]

a = np.zeros((2, 3))
print(a) 
'''
[[0. 0. 0.]
 [0. 0. 0.]]
'''

# empty() : create random number
a = np.empty(6)
print(a) 
'''
[0. 0. 0. 0. 0. 0.]
'''

a = np.empty([3,4], dtype="int")
print(a) 
'''
[[    139839394356192     139839394356192                   0                    0]
 [2338042659922867310 7453010373645639725 2334102053416169504  8749496146326612324]
 [7310309138933707808 8313962067349693812  723441436278466336  2314885530818453536]]
'''

# linespace()
a = np.linspace(1, 100, num = 5)
print(a) 
'''
[  1.    25.75  50.5   75.25 100.  ]
'''

a = np.linspace(1, 100, num = 5, endpoint=False)
print(a) 
'''
 1.  20.8 40.6 60.4 80.2]
'''

a = np.linspace(1, 100, num = 4, retstep=True)
print(a) 
'''
(array([  1.,  34.,  67., 100.]), 33.0)
33.0 is step size in this array
'''

a = np.linspace(1, 100, num = 4, dtype="int")
print(a) 
'''
[  1  34  67 100]
'''

a = np.linspace(1, 100)
print(a) 
'''
# Because num = 50  (default value)
[  1.           3.02040816   5.04081633   7.06122449   9.08163265
  11.10204082  13.12244898  15.14285714  17.16326531  19.18367347
  21.20408163  23.2244898   25.24489796  27.26530612  29.28571429
  31.30612245  33.32653061  35.34693878  37.36734694  39.3877551
  41.40816327  43.42857143  45.44897959  47.46938776  49.48979592
  51.51020408  53.53061224  55.55102041  57.57142857  59.59183673
  61.6122449   63.63265306  65.65306122  67.67346939  69.69387755
  71.71428571  73.73469388  75.75510204  77.7755102   79.79591837
  81.81632653  83.83673469  85.85714286  87.87755102  89.89795918
  91.91836735  93.93877551  95.95918367  97.97959184 100.        ]
'''


###################### eye () : return identity matrix, return 2-d array
a = np.eye(5)
print(a)
'''
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

a = np.eye(2,3)
print(a)
'''
[[1. 0. 0.]
 [0. 1. 0.]]
'''

a = np.eye(4, k=-1)
print(a)
'''
[[0. 0. 0. 0.]
 [1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]]
'''


a = np.eye(5, k=2)
print(a)
'''
[[0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
'''

a = np.eye(5, dtype=int)
print(a)
'''
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
'''

###################### identity() ######################
a = np.identity(5)
print(a)
'''
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

a = np.identity(5)
print(a)
'''
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
'''

###################### random() ######################
a = np.random.rand(5)
print(a)
'''
[0.01684513 0.32216275 0.77200679 0.83359611 0.65692758]
'''


a = np.random.rand(4, 5)
print(a)
'''
[[0.6247457  0.63742426 0.67900036 0.67345428 0.38334097]
 [0.95076688 0.14846041 0.43997978 0.38284814 0.68853742]
 [0.43316164 0.85395033 0.4722514  0.77016734 0.39095732]
 [0.88257136 0.4518936  0.00451042 0.02546654 0.23674873]]
'''


a = np.random.randn(5)
print(a)
'''
[-1.28474517 -0.79007414  0.99653832  1.95972196  1.2496082 ]
'''

a = np.random.randn(4, 5)
print(a)
'''
[[ 0.38088477  0.76371801  0.1136683  -0.06428778 -0.26866346]
 [ 0.67859925  0.2766521  -0.47273692  1.45301591  1.84028557]
 [ 1.96635563 -1.19508412 -1.02455879 -1.42029748  1.45873097]
 [ 0.05544593  1.26368551  0.7567487   0.05423605 -0.7346949 ]]
'''


a = np.random.ranf(5)
print(a)
'''
[0.5624476  0.51112089 0.89635294 0.94278581 0.95336725]
'''

#a = np.random.ranf(5, 2) # ERROR
print(a)
'''
    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    
    Returns
    -------
    out : float or ndarray of floats
        Array of random floats of shape `size` (unless ``size=None``, in which
        case a single float is returned).
    
    Examples
    --------
    >>> np.random.random_sample()
    0.47108547995356098
    >>> type(np.random.random_sample())
    <type 'float'>
    >>> np.random.random_sample((5,))
    array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])
    
    Three-by-two array of random numbers from [-5, 0):
    
    >>> 5 * np.random.random_sample((3, 2)) - 5
    array([[-3.99149989, -0.52338984],
           [-2.99091858, -0.79479508],
           [-1.23204345, -1.75224494]])
'''


a = np.random.randint(5)
print(a)
'''
3
'''

a = np.random.randint(4, 5) # ERROR
print(a)
'''
4
'''

a = np.random.randint(5, size = 10)
print(a)
'''
[3 4 3 3 2 0 3 2 0 4]
'''

a = np.random.randint(4, 5, size = 10) 
print(a)
'''
[4 4 4 4 4 4 4 4 4 4]
'''

a = np.random.randint(5, size = (5, 2))
print(a)
'''
[[2 0]
 [3 2]
 [0 4]
 [4 4]
 [1 4]]
'''

a = np.random.randint(4, 5, size = (5, 2)) 
print(a)
'''
[[4 4]
 [4 4]
 [4 4]
 [4 4]
 [4 4]]
'''



