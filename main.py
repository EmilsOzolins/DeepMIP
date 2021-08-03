import itertools
import time
from pathlib import Path

import torch.sparse
from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import hyperparams as params
from data.sudoku_binary import BinarySudokuDataset
from metrics.average_metrics import AverageMetric
from metrics.sudoku_metrics import SudokuMetric
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


if __name__ == '__main__':

    experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml

    experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    experiment.log_code(folder=str(Path().resolve()))

    dataset = BinarySudokuDataset("binary/sudoku.csv")
    splits = [10000, 10000, len(dataset) - 20000]
    test, validation, train = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs,
                                  num_workers=4, prefetch_factor=4, persistent_workers=True)
    validation_dataloader = DataLoader(validation, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs,
                                       num_workers=4, prefetch_factor=4, persistent_workers=True)

    network = MIPNetwork(
        output_bits=params.bit_count,
        feature_maps=params.feature_maps,
        pass_steps=params.recurrent_steps
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    # TODO: Move this to model
    powers_of_two = torch.as_tensor([2 ** k for k in range(0, params.bit_count)], dtype=torch.float32,
                                    device=torch.device('cuda:0'))

    current_step = 0
    while current_step < params.train_steps:

        with experiment.train():
            start = time.time()
            network.train()
            torch.enable_grad()
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
                loss_avg.update({"loss": loss})

                loss.backward()
                optimizer.step()
                current_step += 1

                experiment.log_metric("loss", loss)

            print(
                f"Step {current_step} avg loss: {loss_avg.numpy_result['loss']:.4f} elapsed time {time.time() - start:0.3f}s")

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()
            torch.no_grad()
            sudoku_metric = SudokuMetric()

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
                reshaped_labels = torch.reshape(labels, [params.batch_size, 9, 9])

                sudoku_metric.update(assignment, reshaped_givens, reshaped_labels)

            results = sudoku_metric.numpy_result
            print(f"[step={current_step}]",
                  f"[range_acc={results['range_acc']:.4f}]",
                  f"[givens_acc={results['givens_acc']:.4f}]",
                  f"[rows_acc={results['rows_acc']:.4f}]",
                  f"[col_acc={results['col_acc']:.4f}]")

            # Login in comet.ml dashboard
            experiment.log_metric("range_acc", results['range_acc'])
            experiment.log_metric("givens_acc", results['givens_acc'])
            experiment.log_metric("rows_acc", results['rows_acc'])
            experiment.log_metric("columns_acc", results['col_acc'])

    with experiment.test():
        network.eval()
        torch.no_grad()
        test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=True, collate_fn=batch_graphs,
                                  num_workers=4, prefetch_factor=4, persistent_workers=True)

        sudoku_metric = SudokuMetric()

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
            reshaped_labels = torch.reshape(labels, [params.batch_size, 9, 9])
            sudoku_metric.update(assignment, reshaped_givens, reshaped_labels)

        results = sudoku_metric.numpy_result

        print("\n\n\n------------------ TESTING ------------------\n")
        print("Range accuracy: ", results['range_acc'])
        print("Givens accuracy: ", results['givens_acc'])
        print("Rows accuracy: ", results['rows_acc'])
        print("Columns accuracy: ", results['col_acc'])

        # Login in comet.ml dashboard
        experiment.log_metric("range_acc", results['range_acc'])
        experiment.log_metric("givens_acc", results['givens_acc'])
        experiment.log_metric("rows_acc", results['rows_acc'])
        experiment.log_metric("columns_acc", results['col_acc'])
