from torch.autograd import Variable


import torch
x = Variable(torch.randn(1, 1), requires_grad=True)
with torch.autograd.profiler.profile(use_cuda=True, enabled=True) as prof:
     y = x ** 2
     y.backward()
# NOTE: some columns were removed for brevity
print(prof)


if __name__ == '__main__':
    pass



