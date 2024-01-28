import torch 

class CustomReLU(torch.nn.Module):
    def __init__(self, leak=None, sub=None):
        super().__init__()
        self.leak, self.sub = leak, sub

    def forward(self, x):
        if self.leak is None:
            x = torch.nn.functional.relu(x) 
        else:
            x = torch.nn.functional.leaky_relu(x, self.leak) 
        if self.sub is not None: x -= self.sub
        return x