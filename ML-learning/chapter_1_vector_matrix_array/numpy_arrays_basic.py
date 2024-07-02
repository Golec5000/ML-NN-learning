import numpy as np

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])

# Create a 1D array as column vector
col = np.array([
    [1],
    [2],
    [3]
])

# Create matrix
matrix_1 = np.array([
    [1, 2],
    [3, 4]
])

# Create sparse matrix
from scipy import sparse

matrix_2 = np.array([
    [0, 0],
    [0, 1],
    [3, 0]
])

matrix_sparse = sparse.csr_matrix(matrix_2)
print(matrix_sparse)

large_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

large_matrix_sparse = sparse.csr_matrix(large_matrix)
print(large_matrix_sparse)

# np aloccates memory for the array

vector = np.zeros(5)
print(vector)

# np matrix allocation

matrix = np.full((2, 3), 5)
print(matrix)

# np matrix allocation with random numbers
matrix = np.random.random((2, 3))
print(matrix)

# get value from matrix/vector

vector = np.array([1, 2, 3, 4, 5])
print(vector[0])  # 1
print(vector[:])  # [1 2 3 4 5] -> all elements
print(vector[0:2])  # [1 2] -> first two elements
print(vector[3:])  # [4 5] -> last two elements
print(vector[:3])  # [1 2 3] -> first three elements
print(vector[-1])  # 5 -> last element

print(vector[::-1])  # [5 4 3 2 1] -> reverse vector

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix[1, 1])  # 5
print(matrix[:, 1])  # [2 5 8] -> second column
print(matrix[0:2, :])  # [[1 2 3]
#  [4 5 6]] -> first two rows

# matrix description

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix.shape)  # (3, 3) -> 3 rows, 3 columns
print(matrix.size)  # 9 -> total number of elements
print(matrix.ndim)  # 2 -> matrix dimensionality

# matrix operations

matrix_a = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2]
])

print(matrix_a)

add_100 = lambda i: i + 100 # lambda function -> add 100 to element

vectorized_add_100 = np.vectorize(add_100)  # vectorize function -> apply to all elements in matrix

print(vectorized_add_100(matrix_a))


# np min max

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(np.max(matrix))  # 9
print(np.min(matrix))  # 1

print(np.max(matrix, axis=0))  # [7 8 9] -> max in each column
print(np.max(matrix, axis=1))  # [3 6 9] -> max in each row

# np avg var std

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(np.mean(matrix))  # 5.0 -> mean of all elements (average)
print(np.var(matrix))  # 6.666666666666667 -> variance
print(np.std(matrix))  # 2.581988897471611 -> standard deviation

print(np.mean(matrix, axis=0))  # [4. 5. 6.] -> mean in each column

# np reshape

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

print(matrix.reshape(2, 6))  # [[1 2 3 4 5 6] [7 8 9 0 0 0]] -> reshape to 2x6

print(matrix.reshape(1, -1))  # [[1 2 3 4 5 6 7 8 9 10 11 12]] -> reshape to 1x12

print(matrix.reshape(12))  # [1 2 3 4 5 6 7 8 9 10 11 12] -> reshape to 1x12

# np transpose

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix.T)  # [[1 4 7] [2 5 8] [3 6 9]] -> transpose matrix

# np flatten

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix.flatten())  # [1 2 3 4 5 6 7 8 9] -> flatten matrix to 1D array -> np.reshape(matrix, -1)

# np rank

matrix = np.array([
    [1, 1, 1],
    [1, 1, 10],
    [1, 1, 15]
])

print(np.linalg.matrix_rank(matrix))  # 2 -> rank of matrix

# np diagonal

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix.diagonal())  # [1 5 9] -> get diagonal elements

# np trace

print(matrix.trace())  # 15 -> sum of diagonal elements -> np.sum(matrix.diagonal())

# np dot product

vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print(np.dot(vector_a, vector_b))  # 32 -> dot product of two vectors
print(vector_a @ vector_b)  # 32 -> dot product of two vectors
# np add subtract multiply

matrix_a = np.random.random((3,3))

matrix_b = np.random.random((3,3))

print(np.add(matrix_a, matrix_b))  # add two matrices -> matrix_a + matrix_b
print(np.subtract(matrix_a, matrix_b))  # subtract two matrices -> matrix_a - matrix_b
print(np.multiply(matrix_a, matrix_b))  # multiply two matrices -> matrix_a @ matrix_b

# np inverse

matrix = np.array(
    [
        [1, 4],
        [2, 5]
    ]

)
inveted_matrix = np.linalg.inv(matrix)
print(inveted_matrix)  # inverse of matrix

print(matrix @ inveted_matrix)  # [[1. 0.] [0. 1.]] -> identity matrix

# np generate random numbers

print(np.random.random(3))  # [0.14022471 0.96360618 0.37601032] -> 3 random numbers between 0 and 1
print(np.random.randint(0, 11, 3))  # [7 2 9] -> 3 random integers between 0 and 10
print(np.random.normal(0.0, 1.0, 3))  # [ 0.14404357 -0.50132438  0.27480168] -> 3 random numbers from normal distribution
print(np.random.logistic(0.0, 1.0, 3))  # [-0.07336607 -0.07336607  0.07336607] -> 3 random numbers from logistic distribution
