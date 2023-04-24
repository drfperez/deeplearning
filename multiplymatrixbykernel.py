import numpy as np

# Function to parse matrix-like input from user
def parse_matrix_input(prompt):
    matrix_str = input(prompt)
    matrix_list = matrix_str.strip().split(';')
    matrix = []
    for row in matrix_list:
        matrix.append([float(elem) for elem in row.strip().split()])
    return np.array(matrix)

# Get input for matrix
print("Enter the matrix:")
matrix = parse_matrix_input("Use spaces to separate elements within a row, and semicolons to separate rows: ")
print("Matrix:")
print(matrix)

# Get input for kernel
print("Enter the kernel:")
kernel = parse_matrix_input("Use spaces to separate elements within a row, and semicolons to separate rows: ")
print("Kernel:")
print(kernel)

# Perform matrix multiplication
result = np.zeros((matrix.shape[0], kernel.shape[1]))  # Initialize result matrix with zeros
print("Intermediate Calculations:")
for i in range(matrix.shape[0]):
    for j in range(kernel.shape[1]):
        intermediate_result = np.dot(matrix[i], kernel[:, j])
        result[i, j] = np.sum(intermediate_result)
        print("Step {}: Multiply row {} of matrix with column {} of kernel: {}".format(i * kernel.shape[1] + j + 1, i+1, j+1, intermediate_result))

# Display the final result
print("Result:")
print(result)
