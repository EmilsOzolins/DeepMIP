import itertools
import os
import time
from pathlib import Path

import torch.sparse
from comet_ml import Experiment
from torch.utils.data import DataLoader, IterableDataset

import config
import hyperparams as params
from data.kanapsack import BinaryKnapsackDataset
from data.sudoku import BinarySudokuDataset
from metrics.average_metrics import AverageMetric
from metrics.discrete_metrics import DiscretizationMetric
from model.mip_network import MIPNetwork
from utils.data import batch_data
from utils.visualize import format_metrics


def main():
    experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml
    experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    experiment.log_code(folder=str(Path().resolve()))

    # TODO: Move dataset selection to separate resolver and add flag in config
    sudoku_test_data = "binary/sudoku_test.csv"
    sudoku_train_data = "binary/sudoku_train.csv"
    sudoku_val_data = "binary/sudoku_validate.csv"

    test_dataset = BinarySudokuDataset(sudoku_test_data)
    train_dataset = BinarySudokuDataset(sudoku_train_data)
    val_dataset = BinarySudokuDataset(sudoku_val_data)

    # test_dataset = IntegerSudokuDataset(sudoku_test_data)
    # train_dataset = IntegerSudokuDataset(sudoku_train_data)
    # val_dataset = IntegerSudokuDataset(sudoku_val_data)

    # test_dataset = BinaryKnapsackDataset(8, 8)
    # train_dataset = BinaryKnapsackDataset(8, 8)
    # val_dataset = BinaryKnapsackDataset(8, 8)

    train_dataloader = create_data_loader(train_dataset)
    validation_dataloader = create_data_loader(val_dataset)

    network = MIPNetwork(
        output_bits=train_dataset.required_output_bits,
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
            print(format_metrics(current_step, {**disc_metric, **loss_res, "elapsed_time": elapsed_time}))
            experiment.log_metrics({**disc_metric, **loss_res, "elapsed_time": elapsed_time})

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()
            torch.no_grad()
            results = evaluate_model(network, validation_dataloader, val_dataset, eval_iterations=100)

            print(format_metrics(current_step, results))
            experiment.log_metrics(results)

    with experiment.test():
        network.eval()
        torch.no_grad()
        test_dataloader = create_data_loader(test_dataset)

        results = evaluate_model(network, test_dataloader, test_dataset)

        print("\n\n\n------------------ TESTING ------------------\n")
        print(format_metrics(params.train_steps, results))
        experiment.log_metrics(results)


def train(train_steps, experiment, network, optimizer, train_dataloader):
    start = time.time()
    loss_avg = AverageMetric()
    disc_metric = DiscretizationMetric()

    for batched_data in itertools.islice(train_dataloader, train_steps):
        edge_indices, edge_values, b_values, size = batched_data["mip"]
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
        disc_metric.update(torch.squeeze(decimal_assignments[-1]))

        loss.backward()
        optimizer.step()

        experiment.log_metric("loss", loss)

    return loss_avg.numpy_result, time.time() - start, disc_metric.numpy_result


def create_data_loader(dataset):
    if config.debugging_enabled:
        return DataLoader(dataset,
                          batch_size=params.batch_size,
                          shuffle=not isinstance(dataset, IterableDataset),
                          collate_fn=batch_data,
                          drop_last=True
                          )
    else:
        return DataLoader(dataset,
                          batch_size=params.batch_size,
                          shuffle=not isinstance(dataset, IterableDataset),
                          collate_fn=batch_data,
                          num_workers=os.cpu_count(),
                          prefetch_factor=4,
                          persistent_workers=True,
                          drop_last=True
                          )


def evaluate_model(network, test_dataloader, dataset, eval_iterations=None):
    dataset.create_metrics()
    iterable = itertools.islice(test_dataloader, eval_iterations) if eval_iterations else test_dataloader

    for batched_data in iterable:
        edge_indices, edge_values, b_values, size = batched_data["mip"]
        adj_matrix = torch.sparse_coo_tensor(edge_indices, edge_values, size=size, device=torch.device('cuda:0'))
        b_values = b_values.cuda()

        binary_assignments, decimal_assignments = network.forward(adj_matrix, b_values)
        dataset.evaluate_model_outputs(binary_assignments[-1], decimal_assignments[-1], batched_data)

    return dataset.get_metrics()


if __name__ == '__main__':
    main()
