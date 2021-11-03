import os
from datetime import datetime as dt

import numpy as np
import torch.sparse
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

import config
import hyperparams as params
from data.kanapsack import BinaryKnapsackDataset
from data.lp_knapsack import LPKnapsackDataset
from optimizers.adam_clip import Adam_clip
from utils.data_utils import batch_data, MIPBatchHolder, sparse_func, make_sparse_unit

now = dt.now()
run_directory = config.model_dir + "/" + now.strftime("%Y%m%d-%H%M%S")
summary = SummaryWriter(run_directory)
load_directory = None  # config.model_dir + "/" + '20211013-145400'
global_step = 0


def main():
    train_dataset = BinaryKnapsackDataset(2, 2)

    train_dataloader = create_data_loader(train_dataset)

    batched_data = next(iter(train_dataloader))
    batch_holder = MIPBatchHolder(batched_data, config.device)
    var_count, const_count = batch_holder.vars_inst_graph.size()

    variables = torch.ones([var_count, 1], device=config.device, requires_grad=True)
    optimizer = torch.optim.Adam([variables], lr=params.learning_rate)
    noise = torch.distributions.Normal(0, 1)

    for i in range(20000):
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = variables + noise.sample(variables.size()).cuda()
            output = torch.sigmoid(output)
            loss, total_loss_c, total_loss_o, best_logit_map = sum_loss([output], batch_holder)
            opt_obj = batch_holder.optimal_solution.cpu().numpy().flatten()
            relax_opt_obj = torch.sparse.mm(batch_holder.vars_obj_graph.t(),
                                            torch.unsqueeze(batch_holder.relaxed_solution,
                                                            dim=-1)).cpu().numpy().flatten()
            opt_found = torch.sparse.mm(batch_holder.vars_obj_graph.t(),
                                        torch.sigmoid(variables)).detach().cpu().numpy().flatten()
            print(
                f"[{i}] Loss={loss}; C_loss={total_loss_c}; O_loss={total_loss_o}; Opt_obj={np.mean(opt_obj)};"
                f" Rel_opt_obj={np.mean(relax_opt_obj)};"
                f" Found_obj={np.mean(opt_found)}")
            loss.backward()
            optimizer.step()

    print(batch_holder.optimal_solution.cpu().numpy().flatten())
    print(torch.sparse.mm(batch_holder.vars_obj_graph.t(),
                          torch.round(torch.sigmoid(variables))).detach().cpu().numpy().flatten())


def sum_loss_sumscaled(asn_list, batch_holder, eps=1e-3):
    sum_loss = 0.

    abs_graph = sparse_func(batch_holder.vars_const_graph, torch.square)
    scalers1 = torch.sqrt(torch.sparse.sum(abs_graph, dim=0).to_dense())
    scalers1 = torch.unsqueeze(torch.clamp(scalers1, min=1e-3), dim=-1)
    # unit_graph = make_sparse_unit(batch_holder.vars_const_graph)
    # scalers2 = torch.unsqueeze(torch.sparse.sum(unit_graph, dim=0).to_dense(), dim=-1)
    scalers1 = 1.0 / scalers1

    if batch_holder.vars_eq_const_graph._nnz() > 0:
        abs_graph = sparse_func(batch_holder.vars_eq_const_graph, torch.square)
        scalers1eq = torch.sqrt(torch.sparse.sum(abs_graph, dim=0).to_dense())
        scalers1eq = torch.unsqueeze(torch.clamp(scalers1eq, min=1e-3), dim=-1)
        scalers1eq = 1.0 / scalers1eq
    else:
        scalers1eq = 1.

    if batch_holder.vars_obj_graph._nnz() == 0:
        scalers1_o = 1.
    else:
        abs_graph_o = sparse_func(batch_holder.vars_obj_graph, torch.square)
        # unit_graph_o = make_sparse_unit(batch_holder.vars_obj_graph)
        scalers1_o = torch.sqrt(torch.sparse.sum(abs_graph_o, dim=0).to_dense())
        scalers1_o = torch.unsqueeze(torch.clamp(scalers1_o, min=1e-3), dim=-1)
        # scalers2_o = torch.unsqueeze(torch.sparse.sum(unit_graph_o, dim=0).to_dense(), dim=-1)
        scalers1_o = 1.0 / scalers1_o

    logit_maps = asn_list[0].size()[-1]
    costs = torch.square(torch.arange(1, logit_maps + 1, dtype=torch.float32, device=asn_list[0].device))

    if batch_holder.vars_eq_const_graph._nnz() > 0:
        eq_const_values = torch.unsqueeze(batch_holder.eq_const_values, dim=-1)
        eq_squared_coef_sum = torch.unsqueeze(torch.sparse.sum(batch_holder.vars_eq_const_graph ** 2, dim=0).to_dense(),
                                              dim=-1)
        unit_var_eq_const_graph = make_sparse_unit(batch_holder.vars_const_graph)
        eq_coef_weight = torch.unsqueeze(torch.sparse.sum(unit_var_eq_const_graph, dim=1), dim=-1).to_dense()

    const_values = torch.unsqueeze(batch_holder.const_values, dim=-1)
    squared_coef_sum = torch.unsqueeze(torch.sparse.sum(batch_holder.vars_const_graph ** 2, dim=0).to_dense(), dim=-1)
    unit_var_const_graph = make_sparse_unit(batch_holder.vars_const_graph)
    coef_weight = torch.unsqueeze(torch.sparse.sum(unit_var_const_graph, dim=1), dim=-1).to_dense()

    for asn in asn_list:
        # Calculate mean squared error for equality constraints
        left_side_eq = torch.sparse.mm(batch_holder.vars_eq_const_graph.t(), asn)
        loss_c_eq = torch.square(torch.unsqueeze(batch_holder.eq_const_values, dim=-1) - left_side_eq) * scalers1eq
        loss_c_eq = torch.sparse.mm(batch_holder.eq_const_inst_graph.t(), loss_c_eq)

        # Normalize equality constraints with single variable
        if batch_holder.vars_eq_const_graph._nnz() > 0:
            dif = (left_side_eq - eq_const_values) / eq_squared_coef_sum
            prediction_dif = torch.sparse.mm(batch_holder.vars_eq_const_graph, dif)
            prediction = asn - prediction_dif / torch.maximum(eq_coef_weight, torch.ones_like(asn))
        else:
            prediction = asn

        # Calculate loss for the rest of the constraints
        left_side = torch.sparse.mm(batch_holder.vars_const_graph.t(), prediction)
        loss_c = torch.relu((left_side - torch.unsqueeze(batch_holder.const_values, dim=-1)) * scalers1)

        # loss_c = torch.square(loss_c)
        loss_c = torch.sparse.mm(batch_holder.const_inst_graph.t(), loss_c)
        # loss_c += torch.mean(bounds_loss_0) + torch.mean(bounds_loss_1)  # todo correct per graph loss
        # loss_c = torch.sqrt(loss_c + 1e-6) - np.sqrt(1e-6) + loss_c_eq

        # dif = torch.relu(left_side - const_values) / squared_coef_sum
        # prediction_dif = torch.sparse.mm(batch_holder.vars_const_graph, dif)
        # prediction = prediction - prediction_dif / torch.maximum(coef_weight, torch.ones_like(prediction))

        loss_o = torch.sparse.mm(batch_holder.vars_obj_graph.t(), prediction)
        loss_o_scaled = loss_o * scalers1_o

        per_graph_loss = 150 * loss_c + loss_o_scaled

        sorted_loss, _ = torch.sort(per_graph_loss, dim=-1, descending=True)
        per_graph_loss_avg = torch.sum(sorted_loss * costs, dim=-1) / torch.sum(costs)

        sum_loss += torch.mean(per_graph_loss_avg)
        # sum_loss_c += torch.mean(loss_c)
        # sum_loss_o += torch.mean(loss_o)

    best_logit_map = torch.argmin(torch.sum(per_graph_loss, dim=0))

    return sum_loss, torch.mean(loss_c), torch.mean(loss_o), best_logit_map


def sum_loss(asn_list, batch_holder):
    return sum_loss_sumscaled(asn_list, batch_holder)


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
                          prefetch_factor=params.batch_size,
                          persistent_workers=True,
                          drop_last=True
                          )


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
