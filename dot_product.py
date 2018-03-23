# Compute the dot product between two matricies

import numpy as np
import random

num_columns = 2
num_rows = 3

# each row is denoted by [row1, row2, row3, ...]
# each column is denoted by each row, row1 = [col1, col2, col3, ...]

# dot product is the elementwise linear combination of two matricies
# # for A = [[a, b], [c, d]], X = [[1, 2], [3, 4]]
# # A, X = 2 column by 2 rows = 2 x 2 matrix
# # --> A * X.T = [[a, b], [c, d]] * [[1,3], [2,4]]
# # --> A * X.T = [[(1a + 3b), (2a + 4b)], [(1c + 3d), (2c + 4d)]]
A = np.array([[1,2,3],[3,4,3]])
X = np.array([[5,6,3],[7,8,3]])

def dot_product(a, x):
    outs = [[] for i in range(len(a))]
    rows = len(a[0])
    for col in range(len(a)):
        for i in range(rows):
            #print(sum(a[col] * x[i]))
            outs[col].append(sum(np.multiply(a[col], x[i])))
    return outs



print('Numpy dot |', np.dot(A, X.T))
print('Comp. dot |', dot_product(A, X.T))
