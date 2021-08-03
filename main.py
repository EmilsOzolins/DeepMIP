import itertools
import time
from pathlib import Path

import torch.sparse
from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import hyperparams as params
from data.sudoku_binary import BinarySudokuDataset
from model.mip_network import MIPNetwork


def batch_graphs(batch):
    mip_instances, givens, labels = list(zip(*batch))

    edge_indices = []
    edge_values = []
    constrain_values = []

    var_offset = 0
    const_offset = 0

    for mip in mip_instances:
        ind = mip.edge_indices
        ind[0, :] += var_offset
        ind[1, :] += const_offset

        var_offset += mip.next_var_index
        const_offset += mip.next_constraint_index

        edge_indices.append(ind)
        edge_values.append(mip.edge_values)
        constrain_values.append(mip.constraints_values)

    mip = torch.cat(edge_indices, dim=-1), torch.cat(edge_values, dim=-1), torch.cat(constrain_values, dim=-1)
    return mip, torch.cat(givens, dim=-1), torch.cat(labels, dim=-1)


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

    experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml

    experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    experiment.log_code(folder=str(Path().resolve()))

    dataset = BinarySudokuDataset("binary/sudoku.csv")
    splits = [10000, 10000, len(dataset) - 20000]
    test, validation, train = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs,
                                  num_workers=8, prefetch_factor=4, persistent_workers=True)
    validation_dataloader = DataLoader(validation, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs,
                                       num_workers=8, prefetch_factor=4, persistent_workers=True)

    network = MIPNetwork(
        output_bits=params.bit_count,
        feature_maps=params.feature_maps,
        pass_steps=params.recurrent_steps
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    # TODO: Move this to model
    powers_of_two = torch.tensor([2 ** k for k in range(0, params.bit_count)], dtype=torch.float32,
                                 device=torch.device('cuda:0'))

    current_step = 0
    while current_step < params.train_steps:

        with experiment.train():
            start = time.time()

            network.train()
            loss_avg = AverageMetric()
            for (edge_indices, edge_values, b_values), givens, labels in itertools.islice(train_dataloader, 1000):
                adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, device=torch.device('cuda:0'))
                b_values = b_values.cuda()
                givens = givens.cuda()
                labels = labels.cuda()

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

            print(f"Step {current_step} avg loss: {loss_avg.numpy_result:.4f} elapsed time {time.time() - start:0.3f}s")

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()

            range_avg = AverageMetric()
            givens_avg = AverageMetric()
            rows_avg = AverageMetric()
            columns_avg = AverageMetric()

            for (edge_indices, edge_values, b_values), givens, labels in itertools.islice(validation_dataloader, 100):
                adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, device=torch.device('cuda:0'))
                b_values = b_values.cuda()
                givens = givens.cuda()
                labels = labels.cuda()

                assignment = network.forward(adj_matrix, b_values)[-1]
                assignment = torch.sum(powers_of_two * assignment, dim=-1)
                assignment = torch.round(assignment)

                assignment = torch.reshape(assignment, [params.batch_size, 9, 9, 9])
                assignment = torch.argmax(assignment, dim=-1) + 1

                reshaped_givens = torch.reshape(givens, [params.batch_size, 9, 9])

                range_avg.update(range_accuracy(assignment))
                givens_avg.update(givens_accuracy(assignment, reshaped_givens))
                rows_avg.update(rows_accuracy(assignment))
                columns_avg.update(columns_accuracy(assignment))

            print(f"[step={current_step}]",
                  f"[range_acc={range_avg.numpy_result:.4f}]",
                  f"[givens_acc={givens_avg.numpy_result:.4f}]",
                  f"[rows_acc={rows_avg.numpy_result:.4f}]",
                  f"[col_acc={columns_avg.numpy_result:.4f}]")

            # Login in comet.ml dashboard
            experiment.log_metric("range_acc", range_avg.result)
            experiment.log_metric("givens_acc", givens_avg.result)
            experiment.log_metric("rows_acc", rows_avg.result)
            experiment.log_metric("columns_acc", columns_avg.result)

    with experiment.test():
        network.eval()
        test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs)

        range_avg = AverageMetric()
        givens_avg = AverageMetric()
        rows_avg = AverageMetric()
        columns_avg = AverageMetric()

        for (edge_indices, edge_values, b_values), givens, labels in test_dataloader:
            adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, device=torch.device('cuda:0'))
            b_values = b_values.cuda()
            givens = givens.cuda()
            labels = labels.cuda()

            assignment = network.forward(adj_matrix, b_values)[-1]
            assignment = torch.sum(powers_of_two * assignment, dim=-1)
            assignment = torch.round(assignment)

            assignment = torch.reshape(assignment, [params.batch_size, 9, 9, 9])
            assignment = torch.argmax(assignment, dim=-1) + 1

            reshaped_givens = torch.reshape(givens, [params.batch_size, 9, 9])

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
