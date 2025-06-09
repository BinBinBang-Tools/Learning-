'''
Tensors
Tensor is a kind of special data which is similar with the arrays and matrics.
We can use tensors as the parameters in models and help encode the inputs and outputs of the model.
'''

import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
1.initializing

Tensors can be initialized in some ways:
a.Directly from data
b.From a NumPy array
c.From another tensor
d.With random or constant values
'''

# Directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data) #retains the properties of x_data
print(f"Ones Tensor:\n{x_ones}\n")
x_rand = torch.rand_like(x_data,dtype=torch.float) #overrides the datatype of x_data
print(f"Ones Tensor:\n{x_rand}\n")

# With random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

'''
2.Attributes of a Tensor

Tensor attributes describe their:
a.shape
b.datatype
c.device on which they are stored
'''

tensor = torch.rand(3,4)
# tensor = torch.rand(3, 4, device="cuda")  # if use CUDA

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor  is stored on: {tensor.device}")

'''
3.Operations

There are many tensor operations,such as arithmetic,linear algebra,matrix manipulation and so on.
By default,tensors are created on the CPU,we can use .to function move to the CUDA.
'''

# move out the tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

# standard numpy-like indexing and slicing
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")  # ":" means all elements
print(f"Last column: {tensor[...,-1]}")  # "..." means all dimensions before, "-1" means the last one
tensor[:,1] = 0  # assignment
print(tensor)

# joining tensors
# use torch.cat funtion can concatenate a sequence of tensors along a given dimension.
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

# arithmetic operations
# use 3 ways to show how to finish the matrix multiplication
#"tensor.T" returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor,tensor.T,out=y3)
# use 3 ways to show how to finish the element-wise multiplication(dot product)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)

# single-element tensors
# use item() can convert it to a python numerical(int or float)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item,type(agg_item))

# in-place operations
# use the new results cover old data
# use "_" as a signal (copy_(),t_(),etc.) as usual
print(f"{tensor}\n")
tensor.add_(5) # value +5
print(tensor)

'''
4.Bridge to the Numpy array

Tensors on the CPU and NumPy arrays can share their underlying memory locations,
and changing one will change the other.
'''

# Tensor to Numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# use "_"
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
# use "_"
np.add(n,1,out=n)
print(f"t: {t}")
print(f"n: {n}")