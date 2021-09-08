import torch


class PairNorm(torch.nn.Module):
    """ PairNorm: Tackling Oversmoothing in GNNs https://arxiv.org/abs/1909.12223
    """

    def __init__(self, s=1, epsilon=1e-6, subtract_mean=True, **kwargs):
        super(PairNorm, self).__init__()
        self.epsilon = epsilon
        self.subtract_mean = subtract_mean
        self.s = s

    def forward(self, inputs):
        # TODO: Normalize per batch instance similarly as in QuerySAT

        # input size: cells x feature_maps
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            mean = torch.mean(inputs, dim=0, keepdim=True)
            inputs -= mean

        variance = torch.mean(torch.square(inputs), dim=1, keepdim=True)
        return self.s * inputs * torch.rsqrt(variance + self.epsilon)


class NodeNorm(torch.nn.Module):
    """ Understanding and Resolving Performance
    Degradation in Graph Convolutional Networks - https://arxiv.org/pdf/2006.07107.pdf
    """

    def __init__(self, p=2, epsilon=1e-6, **kwargs):
        super(NodeNorm, self).__init__()
        self.epsilon = epsilon
        self.p = p

    def forward(self, inputs):
        std = torch.std(inputs, dim=-1, keepdim=True)
        std = torch.pow(std, 1 / self.p)
        return inputs * torch.reciprocal(std + self.epsilon)
