import torch

quant = torch.quantization.QuantStub()
dequant = torch.quantization.DeQuantStub()

l = torch.nn.Linear(4, 4, bias=False)
x = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)
