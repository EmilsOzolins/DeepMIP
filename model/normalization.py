import torch


class PairNorm(torch.nn.Module):
    """ PairNorm: Tackling Oversmoothing in GNNs https://arxiv.org/abs/1909.12223
    """

    def __init__(self, epsilon=1e-6, subtract_mean=False, **kwargs):
        super(PairNorm, self).__init__()
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean

    def forward(self, inputs):
        """
        :param graph: graph level adjacency matrix
        :param count_in_graph: element count in each graph
        :param inputs: input tensor variables or clauses state
        """
        # TODO: Normalize per batch instance similarly as in QuerySAT

        # input size: cells x feature_maps
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            mean = torch.mean(inputs, dim=0, keepdim=True)
            inputs -= mean

        variance = torch.mean(torch.square(inputs), dim=1, keepdim=True)
        return inputs * torch.rsqrt(variance + self.epsilon)
