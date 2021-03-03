import torch
import numpy as np
from torch.nn.quantized import QFunctional
from torch.quantization import QuantStub, DeQuantStub, default_qconfig
import torch.nn.functional as F
import torch.nn as nn


class EmbeddingWithLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=512, embedding_dim=10)
        self.emb.qconfig = torch.quantization.float_qparams_weight_only_qconfig
        self.emb_bag = torch.nn.EmbeddingBag(num_embeddings=512, embedding_dim=10, mode='sum')
        self.emb_bag.qconfig = torch.quantization.float_qparams_weight_only_qconfig
        self.layers = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )
        #self.emb.qconfig = float_qparams_weight_only_qconfig
        #self.emb_bag.qconfig = float_qparams_weight_only_qconfig

        self.qconfig = default_qconfig #torch.quantization.get_default_qconfig('fbgemm')
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):
        #z = self.emb_bag(input.T.contiguous())

        x = self.emb(input)
        #x = torch.sum(x, dim=0)

        fm_input = x
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        x = self.quant(x)
        x = torch.sigmoid(self.layers(x))
        x = self.dequant(x)

        return cross_term


# create a model instance
model_fp32 = EmbeddingWithLinear()
input_fp32 = torch.randint(512, (2048, 39))
#input_fp32 = torch.randn(1, 39, 10)

model_fp32.eval()
_ = model_fp32(input_fp32) # warmup
with torch.autograd.profiler.profile() as prof:
        _ = model_fp32(input_fp32)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))

model_fp32_prepared = torch.quantization.prepare(model_fp32)
model_fp32_prepared(input_fp32)

model_int8 = torch.quantization.convert(model_fp32_prepared)

#model_int8 = torch.quantization.quantize(model_fp32, run_fn=model_fp32.forward, run_args=input_fp32, mapping=None, inplace=False)

_ = model_int8(input_fp32) # warmup
with torch.autograd.profiler.profile() as prof:
    _ = model_int8(input_fp32)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))

prof.export_chrome_trace("trace.json")


'''
Findings:
- EmbeddingBags always faster (non quant AND with quant)
- EmbeddingBags better for quantization
- PyTorch 1.7.1 faster than 1.6
- default qconfig faster than fbgemm in this test case
- warmup important for both models
- deepfwfm benchmark with cpp uses only 1 sample
'''