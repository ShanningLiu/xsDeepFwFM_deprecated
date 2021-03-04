import numpy as np
import torch
import torch.nn.functional as F
from model.Datasets import Dataset, get_dataset
from model.QREmbeddingBag import QREmbeddingBag


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, quantize=False, batch_norm=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self.quantize = quantize
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        if self.quantize:
            x = self.quant(x)
            return self.dequant(self.mlp(x))
        else:
            return self.mlp(x)


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_linear=False):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.use_linear = use_linear
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        if self.use_linear:
            x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        else:
            x = self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DeepFieldWeightedFactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_lw=False, use_fwlw=False, use_emb_bag=False, use_qr_emb=False, quantize_dnn=False, batch_norm=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, quantize=quantize_dnn, batch_norm=batch_norm)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x = [self.fwfm.embeddings[i](torch.unsqueeze(x[:, i], 1)) for i in range(self.num_fields)]
        else:
            embed_x = [self.fwfm.embeddings[i](x[:, i]) for i in range(self.num_fields)]

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        else:
            x = fwfm_second_order + self.mlp(torch.cat(embed_x, 1))

        return torch.sigmoid(x.squeeze(1))


class FieldWeightedFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, use_emb_bag=False, use_qr_emb=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        if use_emb_bag or use_qr_emb:
            self.embeddings = self.create_emb(embed_dim, field_dims, use_qr_emb)
        else:
            self.embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(field_dim, embed_dim) for field_dim in field_dims
            ])
            for embedding in self.embeddings:
                torch.nn.init.xavier_uniform_(embedding.weight.data)
        self.field_cov = torch.nn.Linear(self.num_fields, self.num_fields, bias=False)

    def create_emb(self, m, ln, qr=False, qr_operation="mult", qr_collisions=1, qr_threshold=200):
        emb_l = torch.nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if qr and n > qr_threshold:
                EE = QREmbeddingBag(n, m, qr_collisions,
                                    operation=qr_operation, mode="sum", sparse=False)
            else:
                EE = torch.nn.EmbeddingBag(n, m, mode="sum", sparse=False)
                torch.nn.init.xavier_uniform_(EE.weight.data)

            emb_l.append(EE)

        return emb_l

    def forward(self, x):
        outer_fm = torch.einsum('kij,lij->klij', x, x)
        outer_fwfm = torch.einsum('klij,kl->klij', outer_fm, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
        fwfm_second_order = (torch.sum(torch.sum(outer_fwfm, 0), 0) - torch.sum(torch.einsum('kkij->kij', outer_fwfm), 0)) * 0.5

        return fwfm_second_order


class FieldWeightedFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, use_lw=False, use_fwlw=False, use_emb_bag=False, use_qr_emb=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x = [self.fwfm.embeddings[i](torch.unsqueeze(x[:, i], 1)) for i in range(self.num_fields)]
        else:
            embed_x = [self.fwfm.embeddings[i](x[:, i]) for i in range(self.num_fields)] # TODO most computation here?

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order
        else:
            x = fwfm_second_order

        return torch.sigmoid(x.squeeze(1))


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, use_lw=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.use_lw = use_lw
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_lw:
            x = self.linear(x) + self.fm(self.embedding(x))
        else:
            x = self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
