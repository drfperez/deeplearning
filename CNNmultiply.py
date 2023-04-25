import numpy as np

# Input data
input_data = np.array(input("Enter input data separated by spaces for each row, and use semicolons to separate rows: ").split(";"))
input_data = np.array([row.split() for row in input_data], dtype=float)

# Kernel (Filter)
kernel = np.array(input("Enter kernel (filter) separated by spaces for each row, and use semicolons to separate rows: ").split(";"))
kernel = np.array([row.split() for row in kernel], dtype=float)

# Padding and stride
padding = 0
stride = 1

# Shape of input data and kernel
input_height, input_width = input_data.shape
kernel_height, kernel_width = kernel.shape

# Output feature map shape
output_height = (input_height + 2 * padding - kernel_height) // stride + 1
output_width = (input_width + 2 * padding - kernel_width) // stride + 1

# Initialize output feature map with zeros
output_feature_map = np.zeros((output_height, output_width))

# Perform convolution operation
for i in range(output_height):
    for j in range(output_width):
        # Extract input patch
        input_patch = input_data[i:i+kernel_height, j:j+kernel_width]
        # Perform element-wise multiplication between input patch and kernel
        intermediate_result = np.multiply(input_patch, kernel)
        # Sum the intermediate result
        output_feature_map[i, j] = np.sum(intermediate_result)

print("Intermediate Calculations:")
print("Step 1: Multiply input patch with kernel:")
print("Input Patch:")
print(input_patch)
print("Kernel:")
print(kernel)
print("Intermediate Result:")
print(intermediate_result)
print("Step 2: Add intermediate result to output feature map:")
print("Output Feature Map:")
print(output_feature_map)
