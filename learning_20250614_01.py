'''
A complete project example
'''

import torch
import math

'''
1.Creating Tensors

Tensors are the central data abstraction in PyTorch, we can use some simple ways to create it.
1-dimension tensor can be named vector.
2-dimension tensor can be named matrix.
More dimension is still tensors.
'''

x = torch.empty(3,4)
print(type(x))
print(x)

# Besides empty ones,we can also use other numbers to create
zeros = torch.zeros(2,3)
print(zeros)

ones = torch.ones(2,3)
print(ones)

# 1729 is a seed number, to ensure everytime we can get the same results.
# the results are random values between 0 and 1.
torch.matmul_seed(1729)
random = torch.rand(2,3)
print(random)

'''
2.Tensor shapes

Everytime when we need to do something about tensors,we need to ensure the tensors are of the same shape,
We can use the "torch.*_like" method to control they stay in the same dimensions and each dimension contains same cells.
'''

x = torch.empty(2,2,3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

# we can also create a tensor by covering it to specify its data directly from a PyTorch collection

# PyTorch automatically rounds the output value to 4 decimal places (but the actual storage accuracy remains unchanged)
some_constants = torch.tensor([3.1415926,2.71828],[1.61803,0.0072897])
print(some_constants)

some_integers = torch.tensor((2,3,5,7,11,13,17,19))
print(some_integers)

more_integers = torch.tensor((2,4,6),[3,6,9])
print(more_integers)

'''
3.Tensor Data Types

We can appoint the shape of data when we create tensors
'''

a = torch.ones((2,3),dtype=torch.int16)
print(a)

b = torch.rand((2,3),dtype=torch.float64)*20.
print(b)

c = b.to(torch.int32)
print(c)

'''
4.Math and Logic with PyTorch Tensors

We can perform mathematical operations on tensors like regular arrays
'''

ones = torch.zeros(2,2)+1
twos = torch.ones(2,2)*2
threes = (torch.ones(2,2)*7-1)/2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

# we can also do the same thing between 2 tensors
powers2 = twos ** torch.tensor([[1,2],[3,4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)

'''
5.Exception:Tensor Broadcasting

It allows tensors of different shapes to perform element wise operations (such as addition, subtraction, multiplication, and division). 
The core idea of broadcasting is to make the shapes of two tensors compatible by copying data, without the need for explicit copying of data.
'''

rand = torch.rand(2,4)
doubled = rand * (torch.ones(1,4)*2)

print(rand)
print(doubled)

# some examples
a = torch.ones(4,3,2)
b = a * torch.rand(3,2)
print(b)
c = a * torch.rand(3,1)
print(c)
d = a * torch.rand(1,2)
print(d)

'''
6.More Math with Tensors

There are more method to do.
'''

# Common functions
a = torch.rand(2,4) * 2 - 1
print('Common functions:')
print(torch.abs(a)) # absolute value
print(torch.ceil(a)) # round up
print(torch.floor(a)) # round down
print(torch.clamp(a,-0.5,0.5)) # clip values to [-0.5,0.5]

# Trigonometric functions and inverses
angles = torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4]) # radian angles
sines = torch.sin(angles) # compute sine
inverses = torch.asin(sines) # compute arcsine
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# Bitwise operations
print('\nBitwise XOR')
b = torch.tensor([1,5,11])
c = torch.tensor([2,7,10])
print(torch.bitwise_xor(b,c)) # element-wise XOR

# Comparison operations(with broadcasting)
print('\nBroadcasted equality comparison:')
d = torch.tensor([[1.,2.],[3.,4.]]) #  target tensor
e = torch.ones(1,2) # shape (1,2) will broadcast to (2,2)
print(torch.eq(d,e)) # elementary-wise equality

# Reduction operations
print('\nReduction ops:')
print(torch.max(d)) # maximum value(returns tensor)
print(torch.max(d).item()) # extract scalar value
print(torch.mean(d)) # mean value
print(torch.std(d)) # standard deviation
print(torch.prod(d)) # product of all elements
print(torch.unique(torch.tensor([1,2,1,2,1,2]))) # unique elements

# Linear algebra operations
v1 = torch.tensor([1.,0.,0.]) # x unit vector
v2 = torch.tensor([0.,1.,0.]) # y unit vector
m1 = torch.rand(2,2) # random matrix
m2 = torch.tensor([[3.,0.],[0.,3.]]) # 3x identity matrix

print('\nVectors and Matrices:')
print(torch.linalg.cross(v2,v1)) # cross product (v2*v1 = -z vector)
print(m1) # original random matrix
m3 = torch.linalg.matmul(m1,m2) # matrix multiplication
print(m3)
print(torch.linalg.svd(m3))

# There are some problems in outputs

'''
7.Altering Tensors in Place

Sometimes we need do some calculations but don't want to create more unnecessary space.
We can change it in place.
'''

a = torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4])
print('a: ')
print(a)
print(torch.sin(a))
print(a) # a has no changed

b = torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4])
print('\nb: ')
print(b)
print(torch.sin_(b))
print(b) # b has changed

# For example:
a = torch.ones(2,2)
b = torch.rand(2,2)
print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)

# Another example
a = torch.rand(2,2)
b = torch.rand(2,2)
c = torch.zeros(2,2)
old_id = id(c) # get the memory address identifier of object c
print(c)
d = torch.matmul(a,b,out=c)
print(c)

# "assert" method can test tensors whether at the same address
assert c is d  # test c & d are same object, not just containing equal values
assert  id(c) == old_id  # make sure that our new c is the same object as the old one

torch.rand(2,2,out=c)
print(c)
assert id(c) == old_id
'''
8.Copying Tensors

As with any object in Python, assigning a tensor to a variable makes the variable a label of the tensor, and does not copy it.
If we want a separate copy of data to work,we can use the method: "clone()"
When we need to process "forward()",we have to consider the "autograd".
If the source tensor has autograd,enabled then so will the clone.
'''

a = torch.ones(2,2)
b = a

a[0][1] = 561  # we change a
print(b)  # b is also altered

a = torch.ones(2,2)
b = a.clone()

assert b is not a
print(torch.eq(a,b))
a[0][1] = 561 # a changes
print(b) # b is still ones

a = torch.rand(2,2,requires_grad=True)
print(a)
b = a.clone()
print(b)
c = a.detach().clone()
print(c)
print(a)
'''
9.Moving to Accelerator

We can use CUDA to accelerate the whole processing
'''

if torch.accelerator.is_available():
    print('We have an accelerator!')
else:
    print('Sorry,CPU only.')

# After ensuring the existence of CUDA,we can move data to it.
if torch.accelerator.is_available():
    gpu_rand = torch.rand(2,2,device=torch.accelerator.current_accelerator())
    print(gpu_rand)
else:
    print('Sorry,CPU only.')

# Data are created in CPU by default, so we need to choose "device" to create in CUDA.
# We can use "torch.accelerator.device_count()" to inquiry the numbers of accelerators,and we can designate which one to do our tasks.
my_device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device('cpu')
)
print('Device: {}'.format(my_device))

x = torch.rand(2,2,device=my_device)
print(x)

# We can transfer an existed tensor to another device by ".to()" method.
y = torch.rand(2,2)
y = y.to(my_device)

# We need to ensure all tensors in the same device(cpu or gpu)
x = torch.rand(2,2)
y = torch.rand(2,2,device='cuda')
z = x + y
'''
10.Changing the Number of Dimensions

In PyTorch, models typically require the input to be a batched tensor, where the first dimension is batch size (batch_2).
If the input is a single image (such as (3, 226, 226)), and the model expects (N, 3, 226, 226), then it is necessary to adapt by increasing the dimension (adding batch dimension).

'''

# Method of unsqueeze(add dimension)
a = torch.rand(3,226,226)
b = a.unsqueeze(0) # add a dimension of extent 1. and adds it as a new zeroth dimension
print(a.shape)
print(b.shape)

c = torch.rand(1,1,1,1,1)
print(c)

# Method of squeeze(reduce a dimension of size 1)
a = torch.rand(1,20)
print(a.shape)
print(a)

b= a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2,2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

# We can use "unsqueeze()" method to ease broadcasting
a = torch.ones(4,3,2)
c = a * torch.rand( 3,1)
print(c)

a = torch.ones(4,3,2)
b = torch.rand( 3)
c = b.unsqueeze(1)
print(c.shape)
print(a*c)

# squeeze() and unsqueeze() all owns their "_()" method
batch_me = torch.rand(3,226,226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)

# Sometimes we need change the shape of a tensor more radically,while still preserving the number of elements and contents.
# We can use "reshape()" to do it.
output3d = torch.rand(6,20,20)
print(output3d.shape)

input1d = output3d.reshape(6*20*20)
print(input1d.shape)

print(torch.reshape(output3d,(6*20*20,)).shape)


'''
11.Numpy Bridge

PyTorch tensors can be translate to NumPy arrays.
'''

import numpy as np

numpy_array = np.ones((2,3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)

pytorch_rand = torch.rand(2,3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)

numpy_array[1,1] = 23
print(pytorch_tensor)

pytorch_rand[1,1] = 17
print(numpy_rand)