import itertools
import time
from pathlib import Path

import torch.sparse
from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import config
import hyperparams as params
from data.sudoku_binary import BinarySudokuDataset
from metrics.average_metrics import AverageMetric
from metrics.discreate_metric import DiscretizationMetric
from metrics.sudoku_metrics import SudokuMetric
from model.mip_network import MIPNetwork
from utils.data import batch_graphs


def main():
    experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml
    experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    experiment.log_code(folder=str(Path().resolve()))

    dataset = BinarySudokuDataset("binary/sudoku.csv")
    splits = [10000, 10000, len(dataset) - 20000]
    test_data, val_data, train_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    train_dataloader = create_data_loader(train_data)
    validation_dataloader = create_data_loader(val_data)

    network = MIPNetwork(
        output_bits=params.bit_count,
        feature_maps=params.feature_maps,
        pass_steps=params.recurrent_steps
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=params.learning_rate)

    current_step = 0
    train_steps = 1000

    while current_step < params.train_steps:
        with experiment.train():
            network.train()
            torch.enable_grad()
            loss_res, elapsed_time, disc_metric = train(train_steps, experiment, network, optimizer, train_dataloader)
            current_step += train_steps
            print(f"Step {current_step} avg loss: {loss_res['loss']:.4f} elapsed time {elapsed_time:0.3f}s")
            print(f"[discrete_vs_continuous={disc_metric['discrete_vs_continuous']:.4f}]",
                  f"[discrete_variables={disc_metric['discrete_variables']:.4f}]",
                  f"[max_diff_to_discrete={disc_metric['max_diff_to_discrete']:.4f}]",
                  )

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()
            torch.no_grad()
            results = evaluate_model(network, itertools.islice(validation_dataloader, 100))

            # TODO: Get rid of this logging, it is not important
            print(f"[range_acc={results['range_acc']:.4f}]",
                  f"[givens_acc={results['givens_acc']:.4f}]",
                  f"[rows_acc={results['rows_acc']:.4f}]",
                  f"[col_acc={results['col_acc']:.4f}]",
                  f"[square_acc={results['square_acc']:.4f}]",
                  f"[full_acc={results['full_acc']:.4f}]",
                  )

            # Login in comet.ml dashboard
            experiment.log_metric("range_acc", results['range_acc'])
            experiment.log_metric("givens_acc", results['givens_acc'])
            experiment.log_metric("rows_acc", results['rows_acc'])
            experiment.log_metric("columns_acc", results['col_acc'])
            experiment.log_metric("square_acc", results['square_acc'])
            experiment.log_metric("full_acc", results['full_acc'])

    with experiment.test():
        network.eval()
        torch.no_grad()
        test_dataloader = create_data_loader(test_data)

        results = evaluate_model(network, test_dataloader)

        print("\n\n\n------------------ TESTING ------------------\n")
        print("Range accuracy: ", results['range_acc'])
        print("Givens accuracy: ", results['givens_acc'])
        print("Rows accuracy: ", results['rows_acc'])
        print("Columns accuracy: ", results['col_acc'])
        print("Sub-squares accuracy: ", results['square_acc'])
        print("Full accuracy: ", results['full_acc'])

        # Login in comet.ml dashboard
        experiment.log_metric("range_acc", results['range_acc'])
        experiment.log_metric("givens_acc", results['givens_acc'])
        experiment.log_metric("rows_acc", results['rows_acc'])
        experiment.log_metric("columns_acc", results['col_acc'])
        experiment.log_metric("square_acc", results['square_acc'])
        experiment.log_metric("full_acc", results['full_acc'])


def train(train_steps, experiment, network, optimizer, train_dataloader):
    start = time.time()
    loss_avg = AverageMetric()
    disc_metric = DiscretizationMetric()

    for (edge_indices, edge_values, b_values, size), givens, labels in itertools.islice(train_dataloader, train_steps):
        adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, size=size, device=torch.device('cuda:0'))
        b_values = b_values.cuda()

        optimizer.zero_grad()
        binary_assignments, decimal_assignments = network.forward(adj_matrix, b_values)

        loss = 0
        for asn in decimal_assignments:
            l = torch.relu(torch.squeeze(torch.sparse.mm(adj_matrix.t(), asn)) - b_values)
            # l = torch.square(l)
            loss += torch.sum(l)

        loss /= len(decimal_assignments)
        loss_avg.update({"loss": loss})
        disc_metric.update(torch.squeeze(binary_assignments[-1]))

        loss.backward()
        optimizer.step()

        experiment.log_metric("loss", loss)

    return loss_avg.numpy_result, time.time() - start, disc_metric.numpy_result


def create_data_loader(test):
    if config.debugging_enabled:
        return DataLoader(test,
                          batch_size=params.batch_size,
                          shuffle=True,
                          collate_fn=batch_graphs,
                          drop_last=True
                          )
    else:
        return DataLoader(test,
                          batch_size=params.batch_size,
                          shuffle=True,
                          collate_fn=batch_graphs,
                          num_workers=4,
                          prefetch_factor=4,
                          persistent_workers=True,
                          drop_last=True
                          )


def evaluate_model(network, test_dataloader):
    sudoku_metric = SudokuMetric()
    for (edge_indices, edge_values, b_values, size), givens, labels in test_dataloader:
        adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, size=size, device=torch.device('cuda:0'))
        b_values = b_values.cuda()
        givens = givens.cuda()
        labels = labels.cuda()

        binary_assignments, decimal_assignments = network.forward(adj_matrix, b_values)

        assignment = decimal_assignments[-1]
        assignment = torch.round(assignment)

        assignment = torch.reshape(assignment, [params.batch_size, 9, 9, 9])
        assignment = torch.argmax(assignment, dim=-1) + 1

        reshaped_givens = torch.reshape(givens, [params.batch_size, 9, 9])
        reshaped_labels = torch.reshape(labels, [params.batch_size, 9, 9])

        sudoku_metric.update(assignment, reshaped_givens, reshaped_labels)

    # print(torch.reshape(binary_assignments[-1], [params.batch_size, 9, 9, 9]).detach().cpu().numpy())
    # print(assignment.cpu().numpy()[-1, ...])
    # print(reshaped_givens.cpu().numpy()[-1, ...])

    return sudoku_metric.numpy_result


if __name__ == '__main__':
    main()
