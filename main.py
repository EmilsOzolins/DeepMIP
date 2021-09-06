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
from metrics.discrete_metrics import DiscretizationMetrics
from metrics.general_metrics import AverageMetrics, MetricsHandler
from model.mip_network import MIPNetwork
from utils.data import batch_data, MIPBatchHolder
from utils.visualize import format_metrics


def main():
    experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml
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

    test_dataset = BinaryKnapsackDataset(2, 20)
    train_dataset = BinaryKnapsackDataset(2, 20)
    val_dataset = BinaryKnapsackDataset(2, 20)

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
            loss_res, elapsed_time, disc_metric = train(train_steps, experiment, network,
                                                        optimizer, train_dataloader, train_dataset)
            current_step += train_steps
            print(format_metrics("train", current_step, {**disc_metric, **loss_res, "elapsed_time": elapsed_time}))
            experiment.log_metrics({**disc_metric, **loss_res, "elapsed_time": elapsed_time})

        # TODO: Implement saving to checkpoint - model, optimizer and steps
        # TODO: Implement training, validating and tasting from checkpoint

        with experiment.validate():
            network.eval()
            torch.no_grad()
            results = evaluate_model(network, validation_dataloader, val_dataset, eval_iterations=100)

            print(format_metrics("val", current_step, results))
            experiment.log_metrics(results)

    with experiment.test():
        network.eval()
        torch.no_grad()
        test_dataloader = create_data_loader(test_dataset)

        results = evaluate_model(network, test_dataloader, test_dataset, eval_iterations=100)

        print("\n\n\n------------------ TESTING ------------------\n")
        print(format_metrics("test", params.train_steps, results))
        experiment.log_metrics(results)


def train(train_steps, experiment, network, optimizer, train_dataloader, dataset):
    loss_avg = AverageMetrics()  # TODO: Think what to do with this. LossMetrics???
    metrics = MetricsHandler(DiscretizationMetrics(), *dataset.train_metrics)
    device = torch.device(config.device)

    start = time.time()
    for batched_data in itertools.islice(train_dataloader, train_steps):
        batch_holder = MIPBatchHolder(batched_data, device)

        optimizer.zero_grad()
        binary_assignments, decimal_assignments = network.forward(batch_holder, device)

        # TODO: Deal with this loss garbage
        loss = 0
        total_loss_o = 0
        total_loss_c = 0
        for asn in decimal_assignments:
            left_side = torch.sparse.mm(batch_holder.vars_const_graph.t(), asn)
            loss_c = torch.relu(left_side - torch.unsqueeze(batch_holder.const_values, dim=-1))
            # loss_c = torch.square(loss_c)
            loss_c = torch.sparse.mm(batch_holder.const_inst_graph.t(), loss_c)

            loss_o = torch.sparse.mm(batch_holder.vars_obj_graph.t(), asn)

            total_loss_c += torch.mean(loss_c)
            total_loss_o += torch.mean(loss_o)  # Calculate mean over graphs

            loss += torch.mean(loss_c + loss_o * 0.3)

        steps_taken = len(decimal_assignments)

        total_loss_o /= steps_taken
        total_loss_c /= steps_taken
        loss /= steps_taken

        prediction = dataset.decode_model_outputs(binary_assignments[-1], decimal_assignments[-1])
        loss_avg.update(loss=loss, loss_opt=total_loss_o, loss_const=total_loss_c)
        metrics.update(prediction=prediction, batch_holder=batch_holder, logits=binary_assignments[-1])

        loss.backward()
        optimizer.step()

        experiment.log_metric("loss", loss)
        experiment.log_metric("loss_opt", total_loss_o)
        experiment.log_metric("loss_const", total_loss_c)

    return loss_avg.numpy_result, time.time() - start, metrics.numpy_result


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
    iterable = itertools.islice(test_dataloader, eval_iterations) if eval_iterations else test_dataloader
    metrics = MetricsHandler(*dataset.test_metrics)
    device = torch.device(config.device)

    for batched_data in iterable:
        batch_holder = MIPBatchHolder(batched_data, device)

        binary_assignments, decimal_assignments = network.forward(batch_holder, device)

        prediction = dataset.decode_model_outputs(binary_assignments[-1], decimal_assignments[-1])
        metrics.update(prediction=prediction, batch_holder=batch_holder)

    return metrics.numpy_result


if __name__ == '__main__':
    main()
