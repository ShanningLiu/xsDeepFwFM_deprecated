import torch
import numpy as np
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig


class EmbeddingWithLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=50, embedding_dim=12)
        self.fc = torch.nn.Linear(5, 5)
        self.emb.qconfig = float_qparams_weight_only_qconfig
        self.qconfig = default_qconfig # torch.quantization.get_default_qconfig('fbgemm')
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, indices, linear_in):
        a = self.emb(indices)
        x = self.quant(linear_in)
        quant = self.fc(x)
        return a, self.dequant(quant)


# create a model instance
model_fp32 = EmbeddingWithLinear()

# input data
#indices_fp32 = torch.LongTensor(2, 1).random_(0, 10)
indices_fp32 = torch.empty(2, 1, dtype=torch.long)
input_fp32 = torch.randn(2, 5)

# original result
model_fp32.eval()
res = model_fp32(indices_fp32, input_fp32)
print(res)

# prepare model
model_fp32_prepared = torch.quantization.prepare(model_fp32)
print(model_fp32_prepared)

# calibration
model_fp32_prepared(indices_fp32, input_fp32)

#convertion
model_int8 = torch.quantization.convert(model_fp32_prepared)
res = model_int8(indices_fp32, input_fp32)
print(res)
