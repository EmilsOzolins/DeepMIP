import itertools
import os
import time
from pathlib import Path

from comet_ml import Experiment
import torch.sparse
from torch.utils.data import DataLoader, IterableDataset

import config
import hyperparams as params
from data.kanapsack import BinaryKnapsackDataset
from metrics.average_metrics import AverageMetric
from metrics.discrete_metrics import DiscretizationMetric
from metrics.mip_metrics import MIPMetrics
from model.mip_network import MIPNetwork
from utils.data import batch_data
from utils.visualize import format_metrics


def main():
    experiment = Experiment(disabled=False)  # Set to True to disable logging in comet.ml
    experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    experiment.log_code(folder=str(Path().resolve()))

    # TODO: Move dataset selection to separate resolver and add flag in config
    sudoku_test_data = "binary/sudoku_test.csv"
    sudoku_train_data = "binary/sudoku_train.csv"
    sudoku_val_data = "binary/sudoku_validate.csv"

    # test_dataset = BinarySudokuDataset(sudoku_test_data)
    # train_dataset = BinarySudokuDataset(sudoku_train_data)
    # val_dataset = BinarySudokuDataset(sudoku_val_data)

    # test_dataset = IntegerSudokuDataset(sudoku_test_data)
    # train_dataset = IntegerSudokuDataset(sudoku_train_data)
    # val_dataset = IntegerSudokuDataset(sudoku_val_data)

    test_dataset = BinaryKnapsackDataset(2, 10)
    train_dataset = BinaryKnapsackDataset(2, 10)
    val_dataset = BinaryKnapsackDataset(2, 10)

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

    device = torch.device('cuda:0')

    for batched_data in itertools.islice(train_dataloader, train_steps):

        vars_const_edges, vars_const_values, constr_b_values, size = batched_data["mip"]["constraints"]
        vars_constr_graph = torch.sparse_coo_tensor(vars_const_edges, vars_const_values, size=size, device=device)
        constr_b_values = constr_b_values.cuda()

        obj_edge_indices, obj_edge_values, size = batched_data["mip"]["objective"]
        obj_adj_matrix = torch.sparse_coo_tensor(obj_edge_indices, obj_edge_values, size=size, device=device)

        const_inst_edges, const_inst_values, size = batched_data["mip"]["consts_per_graph"]
        const_inst_graph = torch.sparse_coo_tensor(const_inst_edges, const_inst_values, size=size, device=device)

        vars_inst_edges, vars_inst_values, size = batched_data["mip"]["vars_per_graph"]
        vars_inst_graph = torch.sparse_coo_tensor(vars_inst_edges, vars_inst_values, size=size, device=device)

        optimizer.zero_grad()
        binary_assignments, decimal_assignments = network.forward(vars_constr_graph, constr_b_values)

        loss = 0
        total_loss_o = 0
        total_loss_c = 0
        for asn in decimal_assignments:
            loss_c = torch.relu(torch.sparse.mm(vars_constr_graph.t(), asn) - torch.unsqueeze(constr_b_values, dim=-1))
            loss_c = torch.square(loss_c)
            loss_c = torch.sparse.mm(const_inst_graph.t(), loss_c)

            loss_o = torch.sparse.mm(obj_adj_matrix.t(), asn)

            total_loss_c += torch.mean(loss_c)
            total_loss_o += torch.mean(loss_o)  # Calculate mean over graphs
            loss += torch.mean(loss_c + loss_o * 0.1)

        steps_taken = len(decimal_assignments)

        total_loss_o /= steps_taken
        total_loss_c /= steps_taken
        loss /= steps_taken

        loss_avg.update({"loss": loss, "loss_opt": total_loss_o, "loss_const": total_loss_c})
        disc_metric.update(torch.squeeze(decimal_assignments[-1]))

        loss.backward()
        optimizer.step()

        experiment.log_metric("loss", loss)
        experiment.log_metric("loss_opt", total_loss_o)
        experiment.log_metric("loss_const", total_loss_c)

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

    metrics = MIPMetrics()

    device = torch.device('cuda:0')

    for batched_data in iterable:
        vars_const_edges, vars_const_values, constr_b_values, size = batched_data["mip"]["constraints"]
        vars_constr_graph = torch.sparse_coo_tensor(vars_const_edges, vars_const_values, size=size, device=device)
        constr_b_values = constr_b_values.cuda()

        obj_edge_indices, obj_edge_values, size = batched_data["mip"]["objective"]
        obj_adj_matrix = torch.sparse_coo_tensor(obj_edge_indices, obj_edge_values, size=size, device=device)

        const_inst_edges, const_inst_values, size = batched_data["mip"]["consts_per_graph"]
        const_inst_graph = torch.sparse_coo_tensor(const_inst_edges, const_inst_values, size=size, device=device)

        vars_inst_edges, vars_inst_values, size = batched_data["mip"]["vars_per_graph"]
        vars_inst_graph = torch.sparse_coo_tensor(vars_inst_edges, vars_inst_values, size=size, device=device)

        binary_assignments, decimal_assignments = network.forward(vars_constr_graph, constr_b_values)
        dataset.evaluate_model_outputs(binary_assignments[-1], decimal_assignments[-1], batched_data)

        predictions = dataset.decode_model_outputs(binary_assignments[-1], decimal_assignments[-1])
        metrics.update(predictions, vars_constr_graph, constr_b_values, const_inst_graph)

    return {**dataset.get_metrics(), **metrics.numpy_result}


if __name__ == '__main__':
    main()
