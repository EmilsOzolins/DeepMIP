import itertools
import os
import time
import numpy as np

import torch.sparse
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

import config
import hyperparams as params
from data.kanapsack import BinaryKnapsackDataset
from data.load_balancing import LoadBalancingDataset
from metrics.discrete_metrics import DiscretizationMetrics
from metrics.general_metrics import AverageMetrics, MetricsHandler
from model.mip_network import MIPNetwork
from utils.data import batch_data, MIPBatchHolder
from utils.visualize import format_metrics
from datetime import datetime as dt

now = dt.now()
run_directory = config.model_dir + "/" + now.strftime("%Y%m%d-%H%M%S")
summary = SummaryWriter(run_directory)
global_step = 0


def main():
    # experiment = Experiment(disabled=True)  # Set to True to disable logging in comet.ml
    # experiment.log_parameters({x: getattr(params, x) for x in dir(params) if not x.startswith("__")})
    # experiment.log_code(folder=str(Path().resolve()))

    # TODO: Move dataset selection to separate resolver and add flag in config
    sudoku_test_data = "binary/sudoku_test.csv"
    sudoku_train_data = "binary/sudoku_train.csv"
    sudoku_val_data = "binary/sudoku_validate.csv"
    #
    # test_dataset = BinarySudokuDataset(sudoku_test_data)
    # train_dataset = BinarySudokuDataset(sudoku_train_data)
    # val_dataset = BinarySudokuDataset(sudoku_val_data)

    # test_dataset = IntegerSudokuDataset(sudoku_test_data)
    # train_dataset = IntegerSudokuDataset(sudoku_train_data)
    # val_dataset = IntegerSudokuDataset(sudoku_val_data)
    #
    test_dataset = BinaryKnapsackDataset(2, 20)
    train_dataset = BinaryKnapsackDataset(2, 20)
    val_dataset = BinaryKnapsackDataset(2, 20)

    # train_dataset = LoadBalancingDataset("/host-dir/mip_data/item_placement/train")
    # val_dataset = LoadBalancingDataset("/host-dir/mip_data/item_placement/valid")

    train_dataloader = create_data_loader(train_dataset)
    validation_dataloader = create_data_loader(val_dataset)

    network = MIPNetwork(
        output_bits=params.output_bits,
        feature_maps=params.feature_maps,
        pass_steps=params.recurrent_steps,
        summary = summary
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=params.learning_rate)

    current_step = 0
    train_steps = 1000

    while current_step < params.train_steps:
        # with experiment.train():
        network.train()
        torch.enable_grad()
        loss_res, elapsed_time, disc_metric = train(train_steps, network,
                                                    optimizer, train_dataloader, train_dataset)
        current_step += train_steps
        print(format_metrics("train", current_step, {**disc_metric, **loss_res, "elapsed_time": elapsed_time}))
        # experiment.log_metrics({**disc_metric, **loss_res, "elapsed_time": elapsed_time})

        for name, param in network.named_parameters():
            summary.add_histogram("grad/" + name, param.grad, current_step)
            summary.add_histogram("params/" + name, param.data, current_step)

        for k, v in loss_res.items():
            summary.add_scalar("loss/" + k, v, current_step)

        for k, v in disc_metric.items():
            summary.add_scalar("discrete/" + k, v, current_step)

        torch.save({
            'step': current_step,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, run_directory + "/model.pth")

        # TODO: Implement training, validating and tasting from checkpoint

        # with experiment.validate():
        network.eval()
        with torch.no_grad():
            results = evaluate_model(network, validation_dataloader, val_dataset, eval_iterations=100)

        print(format_metrics("val", current_step, results))
        # experiment.log_metrics(results)

        for k, v in results.items():
            summary.add_scalar("validate/" + k, v, current_step, new_style=True)

    summary.flush()
    summary.close()

    # with experiment.test():
    network.eval()
    torch.no_grad()
    test_dataloader = create_data_loader(test_dataset)

    results = evaluate_model(network, test_dataloader, test_dataset, eval_iterations=100)

    print("\n\n\n------------------ TESTING ------------------\n")
    print(format_metrics("test", params.train_steps, results))
    # experiment.log_metrics(results)


def train(train_steps, network, optimizer, train_dataloader, dataset):
    global global_step
    loss_avg = AverageMetrics()  # TODO: Think what to do with this. LossMetrics???
    metrics = MetricsHandler(DiscretizationMetrics(), *dataset.train_metrics)
    device = torch.device(config.device)

    start = time.time()
    for batched_data in itertools.islice(train_dataloader, train_steps):
        batch_holder = MIPBatchHolder(batched_data, device)

        optimizer.zero_grad(set_to_none=True)
        network.global_step = global_step
        outputs, logits = network.forward(batch_holder, device)

        # TODO: Deal with this loss garbage
        loss = 0
        total_loss_o = 0
        total_loss_c = 0
        for asn in outputs:
            l, loss_c, loss_o, best_logit_map = sum_loss(asn, batch_holder)
            # l = combined_loss(asn, batch_holder)
            loss += l
            total_loss_o += loss_o
            total_loss_c += loss_c

        steps_taken = len(outputs)

        total_loss_o /= steps_taken
        total_loss_c /= steps_taken
        loss /= steps_taken

        result = outputs[-1][:, best_logit_map:best_logit_map + 1]
        prediction = dataset.decode_model_outputs(result, batch_holder)
        loss_avg.update(loss=loss, loss_opt=total_loss_o, loss_const=total_loss_c)
        metrics.update(prediction=prediction, batch_holder=batch_holder, logits=result)
        summary.add_histogram("logits", logits, global_step)
        summary.add_histogram("values", result, global_step)
        global_step += 1

        loss.backward()
        optimizer.step()
        #
        # experiment.log_metric("loss", loss)
        # experiment.log_metric("loss_opt", total_loss_o)
        # experiment.log_metric("loss_const", total_loss_c)

    return loss_avg.numpy_result, time.time() - start, metrics.numpy_result


def combined_loss(asn, batch_holder):
    """
    Makes objective loss dependent on constraint loss.
    """
    left_side = torch.sparse.mm(batch_holder.vars_const_graph.t(), asn)
    vars_in_const = torch.sparse.sum(batch_holder.binary_vars_const_graph, dim=0).to_dense()
    vars_in_const = torch.unsqueeze(vars_in_const, dim=-1)

    loss_c = torch.relu(left_side - torch.unsqueeze(batch_holder.const_values, dim=-1))
    loss_c /= vars_in_const

    loss_per_var = torch.sparse.mm(batch_holder.binary_vars_const_graph, loss_c)

    # TODO: Maybe zero is too strict and small leak should be allowed?
    mask = torch.isclose(loss_per_var, torch.zeros_like(loss_per_var)).float()
    mask = mask * 2 - 1
    loss_per_var = mask * (loss_per_var + 1)

    # TODO: If no objective, optimize constraint loss directly
    obj_multipliers = torch.unsqueeze(batch_holder.objective_multipliers, dim=-1)

    graph_loss = torch.sparse.mm(batch_holder.vars_inst_graph.t(), obj_multipliers * asn * loss_per_var)
    return torch.mean(graph_loss)


def sum_loss(asn, batch_holder):
    eps = 1e-2  # TODO. Also eps in validation because there cn be equality constraints
    left_side = torch.sparse.mm(batch_holder.vars_const_graph.t(), asn)
    loss_c = torch.relu(eps + left_side - torch.unsqueeze(batch_holder.const_values, dim=-1))

    # todo: nicer
    abs_graph = torch.sparse_coo_tensor(batch_holder.vars_const_graph.indices(),
                                        torch.abs(batch_holder.vars_const_graph.values()),
                                        size=batch_holder.vars_const_graph.size(),
                                        device=batch_holder.vars_const_graph.device)
    scalers1 = torch.sparse.sum(abs_graph, dim=0).to_dense()
    scalers2 = torch.sparse.sum(batch_holder.binary_vars_const_graph, dim=0).to_dense()
    loss_c = loss_c * torch.unsqueeze(scalers2 / torch.maximum(scalers1, torch.ones_like(scalers1)), dim=-1)
    # bounds_loss_0 = torch.square(torch.relu(-asn))
    # bounds_loss_1 = torch.square(torch.relu(asn-1))

    loss_c = torch.square(loss_c)
    loss_c = torch.sparse.mm(batch_holder.const_inst_graph.t(), loss_c)
    #loss_c += torch.mean(bounds_loss_0) + torch.mean(bounds_loss_1)  # todo correct per graph loss
    loss_c = torch.sqrt(loss_c + 1e-6) - np.sqrt(1e-6)
    loss_o = torch.sparse.mm(batch_holder.vars_obj_graph.t(), asn)
    abs_graph_o = torch.sparse_coo_tensor(batch_holder.vars_obj_graph.indices(),
                                          torch.abs(batch_holder.vars_obj_graph.values()),
                                          size=batch_holder.vars_obj_graph.size(),
                                          device=batch_holder.vars_obj_graph.device)
    scalers1_o = torch.sparse.sum(abs_graph_o, dim=0).to_dense()
    ones_graph_o = torch.sparse_coo_tensor(batch_holder.vars_obj_graph.indices(),
                                           torch.ones_like(batch_holder.vars_obj_graph.values()),
                                           size=batch_holder.vars_obj_graph.size(),
                                           device=batch_holder.vars_obj_graph.device)
    scalers2_o = torch.sparse.sum(ones_graph_o, dim=0).to_dense()
    loss_o_scaled = loss_o * torch.unsqueeze(scalers2_o / torch.maximum(scalers1_o, torch.ones_like(scalers1_o)),
                                             dim=-1)

    per_graph_loss = loss_c + loss_o_scaled*0.001
    best_logit_map = torch.argmin(torch.sum(per_graph_loss, dim=0))

    logit_maps = per_graph_loss.size()[-1]
    costs = torch.square(torch.arange(1, logit_maps + 1, dtype=torch.float32, device=per_graph_loss.device))
    sorted_loss, _ = torch.sort(per_graph_loss, dim=-1, descending=True)
    per_graph_loss_avg = torch.sum(sorted_loss * costs, dim=-1) / torch.sum(costs)

    return torch.mean(per_graph_loss_avg), torch.mean(loss_c), torch.mean(loss_o), best_logit_map


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

        outputs, logits = network.forward(batch_holder, device)
        l, loss_c, loss_o, best_logit_map = sum_loss(outputs[-1], batch_holder)

        prediction = dataset.decode_model_outputs(outputs[-1][:, best_logit_map:best_logit_map + 1], batch_holder)
        metrics.update(prediction=prediction, batch_holder=batch_holder)

    return metrics.numpy_result


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
