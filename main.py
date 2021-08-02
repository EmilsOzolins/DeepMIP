import itertools

import numpy
from comet_ml import Experiment
import torch.sparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.sudoku_binary import BinarySudokuDataset
from model.mip_network import MIPNetwork


def batch_graphs(batch):
    features = [x for x, _ in batch]

    # TODO: Here I return input Sudoku, but in future change it to solution
    labels = [x for _, x in batch]

    adj_matrices = [x.coalesce() for x, _ in features]
    const_values = [x for _, x in features]

    indices = []
    values = []

    var_offset = 0
    const_offset = 0

    for mat in adj_matrices:
        ind = mat.indices()

        ind[0, :] += var_offset
        ind[1, :] += const_offset

        var_offset = torch.max(ind[0, :]) + 1
        const_offset = torch.max(ind[1, :]) + 1

        indices.append(ind)
        values.append(mat.values())

    batch_adj = torch.sparse_coo_tensor(torch.cat(indices, dim=-1), torch.cat(values, dim=-1))

    return (batch_adj, torch.cat(const_values, dim=-1)), torch.cat(labels, dim=-1)


def relu1(inputs):
    return torch.clamp(inputs, 0, 1)


def rows_accuracy(inputs):
    """ Expect 3D tensor with dimensions [batch_size, 9, 9] with integer values
    """

    batch, r, c = inputs.size()
    result = torch.ones([batch, r], device=inputs.device)
    for i in range(1, 10, 1):
        value = torch.sum(torch.eq(inputs, i).int(), dim=-1)
        value = torch.clamp(value, 0, 1)
        result = result * value

    result = torch.mean(result.float(), dim=-1)
    return torch.mean(result)


def columns_accuracy(inputs):
    """ Expect 3D tensor with dimensions [batch_size, 9, 9] with integer values
    """
    batch, r, c = inputs.size()
    result = torch.ones([batch, r], device=inputs.device)
    for i in range(1, 10, 1):
        value = torch.sum(torch.eq(inputs, i).int(), dim=-2)
        value = torch.clamp(value, 0, 1)
        result = result * value

    result = torch.mean(result.float(), dim=-1)
    return torch.mean(result)


def sub_square_accuracy(inputs):
    pass


def givens_accuracy(inputs, givens):
    mask = torch.clamp(givens, 0, 1)
    el_equal = torch.eq(mask * inputs, givens) * mask
    per_batch = torch.mean(el_equal.float(), dim=[-2, -1])
    return torch.mean(per_batch)


def range_accuracy(inputs):
    geq = torch.greater_equal(inputs, 1)
    leq = torch.less_equal(inputs, 9)
    result = torch.logical_and(geq, leq)
    return torch.mean(torch.mean(result.float(), dim=[-2, -1]))


def full_accuracy(inputs, givens):
    pass


def discrete_accuracy(inputs):
    pass


class AverageMetric:

    def __init__(self) -> None:
        self._value = 0
        self._count = 0

    @property
    def result(self):
        return self._value

    @property
    def numpy_result(self):
        return self._value.detach().cpu().numpy()

    def update(self, new_value) -> None:
        self._count += 1
        self._value += (new_value - self._value) / self._count


if __name__ == '__main__':
    numpy.set_printoptions(precision=3)

    experiment = Experiment(
        disabled=False
    )

    batch_size = 4

    dataset = BinarySudokuDataset("binary/sudoku.csv")
    splits = [10000, 10000, len(dataset) - 20000]
    test, validation, train = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=batch_graphs)
    validation_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=batch_graphs)

    bit_count = 1
    steps = 10000

    network = MIPNetwork(bit_count).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    # TODO: Move this to model
    powers_of_two = torch.tensor([2 ** k for k in range(0, bit_count)], dtype=torch.float32,
                                 device=torch.device('cuda:0'))

    current_step = 0
    while current_step < steps:

        with experiment.train():
            network.train()
            loss_avg = AverageMetric()
            for (adj_matrix, b_values), givens in itertools.islice(train_dataloader, 1000):
                optimizer.zero_grad()
                assignments = network.forward(adj_matrix, b_values)

                loss = 0
                last_assignment = None
                for asn in assignments:
                    last_assignment = torch.sum(powers_of_two * asn, dim=-1, keepdim=True)

                    l = relu1(torch.squeeze(torch.sparse.mm(adj_matrix.t(), last_assignment)) - b_values)
                    loss += torch.sum(l)

                loss /= len(assignments)
                loss_avg.update(loss)

                loss.backward()
                optimizer.step()
                current_step += 1
                experiment.log_metric("loss", loss)

            print(f"Step {current_step} avg loss: ", loss_avg.numpy_result)

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()

            range_avg = AverageMetric()
            givens_avg = AverageMetric()
            rows_avg = AverageMetric()
            columns_avg = AverageMetric()

            for (adj_matrix, b_values), givens in itertools.islice(validation_dataloader, 100):
                assignment = network.forward(adj_matrix, b_values)[-1]
                assignment = torch.sum(powers_of_two * assignment, dim=-1)
                assignment = torch.round(assignment)

                assignment = torch.reshape(assignment, [batch_size, 9, 9, 9])
                assignment = torch.argmax(assignment, dim=-1) + 1

                reshaped_givens = torch.reshape(givens, [batch_size, 9, 9])

                range_avg.update(range_accuracy(assignment))
                givens_avg.update(givens_accuracy(assignment, reshaped_givens))
                rows_avg.update(rows_accuracy(assignment))
                columns_avg.update(columns_accuracy(assignment))

            print(f"[step={current_step}],"
                  f"[range_acc={range_avg.numpy_result}],"
                  f"[givens_acc={givens_avg.numpy_result}],"
                  f"[rows_acc={rows_avg.numpy_result}],"
                  f"[col_acc={columns_avg.numpy_result}]")

            # Login in comet.ml dashboard
            experiment.log_metric("range_acc", range_avg.result)
            experiment.log_metric("givens_acc", givens_avg.result)
            experiment.log_metric("rows_acc", rows_avg.result)
            experiment.log_metric("columns_acc", columns_avg.result)

    with experiment.test():
        network.eval()
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=batch_graphs)

        range_avg = AverageMetric()
        givens_avg = AverageMetric()
        rows_avg = AverageMetric()
        columns_avg = AverageMetric()

        for (adj_matrix, b_values), givens in test_dataloader:
            assignment = network.forward(adj_matrix, b_values)[-1]
            assignment = torch.sum(powers_of_two * assignment, dim=-1)
            assignment = torch.round(assignment)

            assignment = torch.reshape(assignment, [batch_size, 9, 9, 9])
            assignment = torch.argmax(assignment, dim=-1) + 1

            reshaped_givens = torch.reshape(givens, [batch_size, 9, 9])

            range_avg.update(range_accuracy(assignment))
            givens_avg.update(givens_accuracy(assignment, reshaped_givens))
            rows_avg.update(rows_accuracy(assignment))
            columns_avg.update(columns_accuracy(assignment))

        print("\n\n\n------------------ TESTING ------------------\n")
        print("Range accuracy: ", range_avg.numpy_result)
        print("Givens accuracy: ", givens_avg.numpy_result)
        print("Rows accuracy: ", rows_avg.numpy_result)
        print("Columns accuracy: ", columns_avg.numpy_result)

        # Login in comet.ml dashboard
        experiment.log_metric("range_acc", range_avg.result)
        experiment.log_metric("givens_acc", givens_avg.result)
        experiment.log_metric("rows_acc", rows_avg.result)
        experiment.log_metric("columns_acc", columns_avg.result)
