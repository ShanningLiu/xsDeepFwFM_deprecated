import numpy as np
import torch
import torch.nn.functional as F
from model.Datasets import Dataset, get_dataset


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
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
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
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_lw=False, use_fwlw=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_offset = x + x.new_tensor(self.fwfm.offsets).unsqueeze(0)
        embed_x = [self.fwfm.embeddings[i](x_offset[:, i]) for i in range(self.num_fields)]

        fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
        fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        elif self.use_fwlw:
            x = fwfm_first_order + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        else:
            x = fwfm_second_order + self.mlp(torch.cat(embed_x, 1))

        return torch.sigmoid(x.squeeze(1))


class FieldWeightedFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)

        self.embed_dim = embed_dim
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.field_cov = torch.nn.Linear(self.num_fields, self.num_fields, bias=False)

    def forward(self, x):
        # compute outer product, outer_fm: 39x39x2048x10
        outer_fm = torch.einsum('kij,lij->klij', x, x)
        outer_fwfm = torch.einsum('klij,kl->klij', outer_fm, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
        fwfm_second_order = (torch.sum(torch.sum(outer_fwfm, 0), 0) - torch.sum(torch.einsum('kkij->kij', outer_fwfm), 0)) * 0.5

        return fwfm_second_order


class FieldWeightedFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, use_lw=False, use_fwlw=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim)
        #TODO shallow dropout?

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_offset = x + x.new_tensor(self.fwfm.offsets).unsqueeze(0)
        embed_x = [self.fwfm.embeddings[i](x_offset[:, i]) for i in range(self.num_fields)]

        fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
        fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order
        elif self.use_fwlw:
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
