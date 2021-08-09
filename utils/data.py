import torch


def batch_graphs(batch):
    mip_instances, givens, labels = list(zip(*batch))

    edge_indices = []
    edge_values = []
    constrain_values = []

    var_offset = 0
    const_offset = 0

    size = [0, 0]

    for mip in mip_instances:
        ind = mip.edge_indices
        ind[0, :] += var_offset
        ind[1, :] += const_offset

        size[0] += mip.size[0]
        size[1] += mip.size[1]

        var_offset += mip.next_var_index
        const_offset += mip.next_constraint_index

        edge_indices.append(ind)
        edge_values.append(mip.edge_values)
        constrain_values.append(mip.constraints_values)

    mip = torch.cat(edge_indices, dim=-1), torch.cat(edge_values, dim=-1), torch.cat(constrain_values, dim=-1), size
    return mip, torch.cat(givens, dim=-1), torch.cat(labels, dim=-1)
