# Hands-on Python Libraries

## NumPy

### Example 01: Code for initializing NumPy array

```python
import numpy as np
# Create a rank 1 array
a = np.array([0, 1, 2])
print(type(a))
# this will print the dimension of the array
print(a.shape)
print(a[0])
print(a[1])
print(a[2])
# Change an element of the array
a[0] = 5
print(a)
# Create a rank 2 array
b = np.array([[0,1,2],[3,4,5]])
print(b.shape)
print(b)
print(b[0, 0], b[0, 1], b[1, 0])
```

**Output : **

```python
<class 'numpy.ndarray'>
(3,)
0
1
2
[5 1 2]
(2, 3)
[[0 1 2]
 [3 4 5]]
0 1 3

```

**Learning :**

* Setting up jupyter notebook
* Creating NumPy arrays



### Example 02: Creating NumPy array

```python
# Create a 3x3 array of all zeros
a = np.zeros((3,3))
print(a)
# Create a 2x2 array of all ones
b = np.ones((2,2))
print(b)
# Create a 3x3 constant array

c = np.full((3,3), 7)
print(c)
# Create a 3x3 array filled with random values
d = np.random.random((3,3))
print(d)
# Create a 3x3 identity matrix
e = np.eye(3)
print(e)
# convert list to array
f = np.array([2, 3, 1, 0])
print(f)
# arange() will create arrays with regularly incrementing values
g = np.arange(20)
print(g)
# note mix of tuple and lists
h = np.array([[0, 1,2.0],[0,0,0],(1+1j,3.,2.)])
print(h)
# create an array of range with float data type
i = np.arange(1, 8, dtype=np.float)
print(i)
# linspace() will create arrays with a specified number of items which are
# spaced equally between the specified beginning and end values j= np.linspace(start, stop, num)
j = np.linspace(2, 4, 5)
print(j)

```

**Output :**

```python
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
[[1. 1.]
 [1. 1.]]
[[7 7 7]
 [7 7 7]
 [7 7 7]]
[[0.83497924 0.36108516 0.8451762 ]
 [0.49592597 0.28751869 0.23165085]
 [0.83766161 0.71813656 0.26373574]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[2 3 1 0]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
[[0.+0.j 1.+0.j 2.+0.j]
 [0.+0.j 0.+0.j 0.+0.j]
 [1.+1.j 3.+0.j 2.+0.j]]
[1. 2. 3. 4. 5. 6. 7.]
[2.  2.5 3.  3.5 4. ]
```

**Learnings :**

* Generating numpy arrays with in-built function like `zero`, `ones`, `eye` , etc.

### Example 3: NumPy data types

```python
# Let numpy choose the datatype
x = np.array([0, 1])
y = np.array([2.0, 3.0])
# Force a particular datatype
z = np.array([5, 6], dtype=np.int64)
print(x.dtype, y.dtype, z.dtype)
```

**Output :**

```python
int64 float64 int64
```

**Learnings :**

* Finding type of a `numpy` structure using `dtype`

### Example 4: Python program to demonstrate indexing in numpy

```python
import numpy as np
# An exemplar array
arr = np.array([[-1, 2, 0, 4],
[4, -0.5, 6, 0],
[2.6, 0, 7, 8],
[3, -7, 4, 2.0]])
# Slicing array
temp = arr[:2, ::2]
print ("Array with first 2 rows and alternate columns(0 and 2):\n", temp)
# Integer array indexing example
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
print ("\nElements at indices (0, 3), (1, 2), (2, 1),(3, 0):\n", temp)
# boolean array indexing example
cond = arr > 0 # cond is a boolean array
temp = arr[cond]
print ("\nElements greater than 0:\n", temp)
```

**Output : **

```python
Array with first 2 rows and alternate columns(0 and 2):
 [[-1.  0.]
 [ 4.  6.]]

Elements at indices (0, 3), (1, 2), (2, 1),(3, 0):
 [4. 6. 0. 3.]

Elements greater than 0:
 [2.  4.  4.  6.  2.6 7.  8.  3.  4.  2. ]
```

**Learnings : **

* Slicing, filtering and indexing 

### Example 5 : Operation on single array

```python
# Python program to demonstrate
# basic operations on single array
import numpy as np
a = np.array([1, 2, 5, 3])
# add 1 to every element
print ("Adding 1 to every element:", a+1)
# subtract 3 from each element
print ("Subtracting 3 from each element:", a-3)
# multiply each element by 10
print ("Multiplying each element by 10:", a*10)
# square each element
print ("Squaring each element:", a**2)
# modify existing array
a *= 2
print ("Doubled each element of original array:", a)
# transpose of array
a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])
print ("\nOriginal array:\n", a)
print ("Transpose of array:\n", a.T)
```

**Output :**

```python
Adding 1 to every element: [2 3 6 4]
Subtracting 3 from each element: [-2 -1  2  0]
Multiplying each element by 10: [10 20 50 30]
Squaring each element: [ 1  4 25  9]
Doubled each element of original array: [ 2  4 10  6]

Original array:
 [[1 2 3]
 [3 4 5]
 [9 6 0]]
Transpose of array:
 [[1 3 9]
 [2 4 6]
 [3 5 0]]
```

**Learnings :**

* Applying arithmetic operations to entire array
* Transposing array

### Example 6 : Unary operator

```python
# Python program to demonstrate
# unary operators in numpy
import numpy as np
arr = np.array([[1, 5, 6],
[4, 7, 2],
[3, 1, 9]])
# maximum element of array
print ("Largest element is:", arr.max())
print ("Row-wise maximum elements:",arr.max(axis = 1))
# minimum element of array
print ("Column-wise minimum elements:",arr.min(axis = 0))
# sum of array elements
print ("Sum of all array elements:",arr.sum())
# cumulative sum along each row
print ("Cumulative sum along each row:\n",arr.cumsum(axis = 1))
```

**Output :**

```python
Largest element is: 9
Row-wise maximum elements: [6 7 9]
Column-wise minimum elements: [1 1 2]
Sum of all array elements: 38
Cumulative sum along each row:
 [[ 1  6 12]
 [ 4 11 13]
 [ 3  4 13]]

```

**Learnings :**

* Finding max, min along different axis
* Finding Cumulative sum

### Example 7 : Binary Operations

```python
# Python program to demonstrate
# binary operators in Numpy
import numpy as np
a = np.array([[1, 2],
[3, 4]])
b = np.array([[4, 3],
[2, 1]])
# add arrays
print ("Array sum:\n", a + b)
# multiply arrays (elementwise multiplication)
print ("Array multiplication:\n", a*b)
# matrix multiplication
print ("Matrix multiplication:\n", a.dot(b))
```

**Output :**

```python
Array sum:
 [[5 5]
 [5 5]]
Array multiplication:
 [[4 6]
 [6 4]]
Matrix multiplication:
 [[ 8  5]
 [20 13]]
```

**Learnings :**

* Applying binary operations

### Example 8: Universal Function

```python
# Python program to demonstrate
# universal functions in numpy
import numpy as np
# create an array of sine values
a = np.array([0, np.pi/2, np.pi])
print ("Sine values of array elements:", np.sin(a))
# exponential values
a = np.array([0, 1, 2, 3])
print ("Exponent of array elements:", np.exp(a))
# square root of array values
print ("Square root of array elements:", np.sqrt(a))
```

**Output : **

```python
Sine values of array elements: [0.0000000e+00 1.0000000e+00 1.2246468e-16]
Exponent of array elements: [ 1.          2.71828183  7.3890561  20.08553692]
Square root of array elements: [0.         1.         1.41421356 1.73205081]
```

**Learnings :**

* Using `numpy` functions like `sin`, `exp`, `sqrt`

### Example 9: Matrix Inverse

```python
##simple python program to find the inverse of an array (without exception handling)
import numpy as np
arr = np.array([[-1, 2, 0, 4],
[4, -0.5, 6, 0],
[2.6, 0, 7, 8],
[3, -7, 4, 2.0]])
inverse = np.linalg.inv(arr)
inverse
```

**Output :**

```python
array([[ 4.05707196,  1.98511166, -2.28287841,  1.01736973],
       [ 0.50620347,  0.30272953, -0.24813896, -0.01985112],
       [-2.66253102, -1.13151365,  1.50124069, -0.67990074],
       [ 1.01116625,  0.34491315, -0.44665012,  0.26426799]])
```

**Learnings :**

* Finding Inverse of a NumPy array



## Pandas

### Example 1: Creating Pandas Series

```python
# creating a series by passing a list of values, and a custom index label.
#Note that the labeled index reference for each row and it can have duplicate
#values
import pandas as pd
import numpy as np
s = pd.Series([1,2,3,np.nan,5,6], index=['A','B','C','D','E','F'])
print(s)
```

**Output :**

```python
A    1.0
B    2.0
C    3.0
D    NaN
E    5.0
F    6.0
dtype: float64
```

**Learnings :**

* Creating Series in Pandas

### Example 2: Create Pandas DataFrame

```python
data = {'Gender': ['F', 'M', 'M'],'Emp_ID': ['E01', 'E02','E03'], 'Age': [25, 27, 25]}
# We want the order the columns, so lets specify in columns parameter
df = pd.DataFrame(data, columns=['Emp_ID','Gender', 'Age'])
print(df) #output all data
print(df.Emp_ID) #output all Emp_ID(entire column)
print(df.Emp_ID[0]) #output first Emp_IDd
```

**Output :**

```python
 Emp_ID Gender  Age
0    E01      F   25
1    E02      M   27
2    E03      M   25
0    E01
1    E02
2    E03
Name: Emp_ID, dtype: object
E01
```

**Learnings :**

* Creating Pandas DataFrame
* Accessing Individual elements in the data frame using dot operator

### Example 3. Reading, Writing data from csv files

```python
import pandas as pd
df=pd.read_csv('mtcars.csv') # from csv
# writing from dataframe to files
# Index = False parameter will not write the index values, default is True
df.to_csv('mtcars_new.csv', index=False)
#df.to_csv('Data/mtcars_new.txt', sep='\t', index=False)
#df.to_excel('Data/mtcars_new.xlsx',sheet_name='Sheet1', index = False)
```

**Learnings :**

* Uploading csv file to jupyter
* Reading and Writing on jupyter



### Example 4: Basic statistics on DataFrame

```python
df = pd.read_csv('iris.csv')
df.describe()
```

**Output :**

|       | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |
| ----- | ------------ | ----------- | ------------ | ----------- |
| count | 11.000000    | 11.000000   | 11.000000    | 11.000000   |
| mean  | 6.081818     | 3.181818    | 3.927273     | 1.245455    |
| std   | 0.948492     | 0.318805    | 2.042102     | 0.847778    |
| min   | 4.700000     | 2.700000    | 1.300000     | 0.200000    |
| 25%   | 5.250000     | 3.000000    | 1.550000     | 0.300000    |
| 50%   | 6.300000     | 3.200000    | 4.700000     | 1.500000    |
| 75%   | 6.950000     | 3.250000    | 5.500000     | 1.850000    |
| max   | 7.300000     | 3.900000    | 6.300000     | 2.500000    |

**Learnings :**

* Getting the statical summary of the dataset 

### Example 5 : Creating covariance on dataframe

```python
df.con()
```

**Output :**

|              | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |
| ------------ | ------------ | ----------- | ------------ | ----------- |
| Sepal.Length | 0.899636     | -0.102364   | 1.732545     | 0.634909    |
| Sepal.Width  | -0.102364    | 0.101636    | -0.339455    | -0.126091   |
| Petal.Length | 1.732545     | -0.339455   | 4.170182     | 1.679636    |
| Petal.Width  | 0.634909     | -0.126091   | 1.679636     | 0.718727    |

**Learnings :**

* Finding covariance of a data frame

### Example 6 : Creating correlation matrix on DataFrame

```python
df.corr()
```

**Output :**

|              | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |
| ------------ | ------------ | ----------- | ------------ | ----------- |
| Sepal.Length | 1.000000     | -0.338523   | 0.894486     | 0.789580    |
| Sepal.Width  | -0.338523    | 1.000000    | -0.521410    | -0.466527   |
| Petal.Length | 0.894486     | -0.521410   | 1.000000     | 0.970188    |
| Petal.Width  | 0.789580     | -0.466527   | 0.970188     | 1.000000    |

**Learnings : **

* Finding correlation matrix of the data frame



## MatPlotLib

### Example 1: Create Plot on variable

```python
# simple bar and scatter plot
import numpy as np
from matplotlib import pyplot as plt
x = np.arange(5) # assume there are 5 students
y = (20, 35, 30, 35, 27) # their test scores
plt.bar(x,y) # Bar plot
# need to close the figure using show() or close(), if not closed any follow
#up plot commands will use same figure.
plt.show() # Try commenting this an run
plt.scatter(x,y) # scatter plot
plt.show()
```

**Output :**

![image-20200905211201468](/home/adesh/.config/Typora/typora-user-images/image-20200905211201468.png)

**Learnings :**

* Setting up `MatplotLib`
* Using different plotting techniques (bar, scatter)

### Example 2: Creating plot on DataFrame

```python
#Example 02: Creating plot on dataframe
import pandas as pd
df = pd.read_csv('iris.csv')
df.hist()# Histogram
df.plot() # Line Graph

df.boxplot() # Box plot:
```

**Output : **

![image-20200905211511198](/home/adesh/.config/Typora/typora-user-images/image-20200905211511198.png)

**Learnings :**

* Plotting histogram and box plot

