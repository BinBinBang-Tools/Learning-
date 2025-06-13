'''
Saving and Load the model

Try to persist model state with saving,loading and running model predictions
'''

import torch
import torchvision.models as models

'''
1.Saving and Loading Model Weights

PyTorch models store the learned parameters in "state_dict",an  internal state dictionary.
We can use it to persist by "torch.save" method.

when we need to load weights, we have to create a same model,and then,use the "load_state_dict()" to load the parameter.
we can do it by "weights_only=True"
'''

model = models.vgg16(weight='IMAGENET1K_V1')
torch.save(model.state_dict(),'model_weights.pth')

model = models.vgg16() #there are creating a new unreained model
model.load_state_dict(torch.load('model_weights.pth',weights_only=True))
model.eval()

'''
2.Saving and Loading Models with Shapes

We can save the model's construct and parameters or only parameters.
And we can load existing models.
'''

torch.save(model, 'model.pth')

model = torch.load('model.pth',weights_only=False)