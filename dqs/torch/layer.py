import torch

def sigsoftmax(input):
    numerator = torch.exp(input) * torch.sigmoid(input)
    denominator = torch.sum(numerator, 1)
    return numerator/denominator.view(-1,1)

class SigSoftmax:
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return sigsoftmax(input)
