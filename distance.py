import numpy as np


size1 = int(input("Size of first array:"))

a = np.empty(size1)
for i in range(len(a)):
    x = float(input("Element:"))
    a[i]=x
print(np.floor(a))

size2 = int(input("Size of second array:"))

b = np.empty(size2)
for i in range(len(b)):
    x = float(input("Element:"))
    b[i]=x
print(np.floor(b))

q = int(input("Value of q:"))

result = 0
for i in range(len(a)):
    result += ((abs(a[i] - b[i])) ** q)
print(result)
result = (result) ** 1/q

print(result)