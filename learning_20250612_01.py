'''
Back Propagation

When training the neural networks,we can use "torch.autograd" block to complete the task of back propagation.
"torch.autograd" supports automatic computation of gradient for any computational graph.
'''

# Consider a one-layer neural network:
import torch
x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)


'''
1.Tensors,Functions and Computational graph

Codes can be show in a computational graph
'''

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

'''
2.Computing Gradients

Optimizing weights needto compute derivatives by loss function.
We can call the "loss.backward()" ,it will retrieve the values from w.grad and b.grad
'''

loss.backward()
print(w.grad)
print(b.grad)

'''
3.Disabling Gradient Tracking

Sometimes we only need to do forward,so we can use "torch.no_grad" or "detach()"to stop tracking computations.
reasons:
a.mark some parameters as frozen parameters
b.speed up computations when we only want to forward pass
'''

# using "torch.no_grad()"
z = torch.matmul(x,w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w)+b
print(z.requires_grad)

# or use "detach()"
z = torch.matmul(x,w)+b
z_det = z.detach()
print(z_det.requires_grad)

'''
4.Tensor Gradients and Jacobian Products

Sometimes we need to deal with the output function as an arbitrary tensor,maybe we can use Jacobian product to process it.
'''

inp = torch.eye(4,5,requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.one_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")