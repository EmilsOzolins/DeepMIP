from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict

import torch
from torch import Tensor

from data.ip_instance import IPInstance


def batch_data(batch: List[Dict]):
    """Data batching with support for batching sparse adjacency matrices"""
    batch_data = defaultdict(list)

    for data in batch:
        for key, item in data.items():
            batch_data[key].append(item)

    output_batch = dict()
    for key, data in batch_data.items():
        if isinstance(data[0], IPInstance):
            output_batch[key] = batch_as_mip(data)
        elif isinstance(data[0], Tensor):
            output_batch[key] = batch_as_tensor(data)
        else:
            raise NotImplementedError(f"Batching for {type(data[0])} is not implemented!")

    return output_batch


def batch_as_mip(mip_instances: Tuple[IPInstance]):
    """ Batches mip instances as one huge sparse graph.
    Graphs should be converted to sparse tensor after data is collected by workers.
    """
    edge_indices = []
    edge_values = []
    constrain_values = []
    var_offset = 0
    const_offset = 0
    size = [0, 0]

    objective_edge_indices = []
    objective_edge_values = []

    for graph_id, mip in enumerate(mip_instances):
        ind = mip.edge_indices
        ind[0, :] += var_offset
        ind[1, :] += const_offset

        size[0] += mip.size[0]
        size[1] += mip.size[1]

        obj_ind = mip.objective_edge_indices
        obj_ind[0, :] += var_offset
        obj_ind[1, :] += graph_id

        var_offset += mip.next_var_index
        const_offset += mip.next_constraint_index

        objective_edge_indices.append(obj_ind)
        objective_edge_values.append(mip.objective_edge_values)

        edge_indices.append(ind)
        edge_values.append(mip.edge_values)
        constrain_values.append(mip.constraints_values)

    constraints = torch.cat(edge_indices, dim=-1), torch.cat(edge_values, dim=-1), torch.cat(constrain_values, dim=-1), size
    objective = torch.cat(objective_edge_indices, dim=-1), torch.cat(objective_edge_values, dim=-1), [size[0], len(mip_instances)]

    return {"constraints": constraints, "objective": objective}


def batch_as_tensor(batch_data: Tuple[Tensor]):
    return torch.stack(batch_data, dim=0)
