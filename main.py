import torch.sparse
from torch.utils.data import DataLoader

from data.sudoku_binary import BinarySudokuDataset
from model.mip_network import MIPNetwork


def batch_graphs(batch):
    features = [x for x, _ in batch]

    # TODO: Here I return input Sudoku, but in future change it to solution
    labels = [x for _, x in batch]

    adj_matrices = [x.coalesce() for x, _ in features]
    const_values = [x for _, x in features]

    indices = []
    values = []

    var_offset = 0
    const_offset = 0

    for mat in adj_matrices:
        ind = mat.indices()

        ind[0, :] += var_offset
        ind[1, :] += const_offset

        var_offset = torch.max(ind[0, :]) + 1
        const_offset = torch.max(ind[1, :]) + 1

        indices.append(ind)
        values.append(mat.values())

    batch_adj = torch.sparse_coo_tensor(torch.cat(indices, dim=-1), torch.cat(values, dim=-1))

    return (batch_adj, torch.cat(const_values, dim=-1)), torch.cat(labels, dim=-1)


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


if __name__ == '__main__':
    batch_size = 4

    dataset = BinarySudokuDataset("binary/sudoku.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_graphs)

    bit_count = 1
    steps = 100000

    network = MIPNetwork(bit_count).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    powers_of_two = torch.tensor([2 ** k for k in range(0, bit_count)], dtype=torch.float32,
                                 device=torch.device('cuda:0'))

    average_loss = 0

    for step, ((adj_matrix, b_values), label) in zip(range(steps), dataloader):
        optimizer.zero_grad()

        assignments = network.forward(adj_matrix, b_values)
        # int_loss = torch.square(assignment) * torch.square(1 - assignment)

        loss = 0
        last_assignment = None
        for asn in assignments:
            last_assignment = torch.sum(powers_of_two * asn, dim=-1, keepdim=True)

            l = relu1(torch.squeeze(torch.sparse.mm(adj_matrix.t(), last_assignment)) - b_values)
            l = torch.sum(l)  # + torch.sum(int_loss)
            loss += l

        loss /= len(assignments)

        loss.backward()
        optimizer.step()

        average_loss += loss.detach()
        if step % 500 == 0:
            assignment = torch.round(torch.squeeze(last_assignment))
            assignment = torch.reshape(assignment, [batch_size, 9, 9, 9])
            assignment = torch.argmax(assignment, dim=-1) + 1

            reshaped_label = torch.reshape(label, [batch_size, 9, 9])

            print("Step: ", step, "Avg. Loss:", (average_loss / 500.).cpu().numpy())
            print("Range accuracy: ", range_accuracy(assignment).cpu().detach().numpy())
            print("Givens accuracy: ", givens_accuracy(assignment, reshaped_label.cuda()).cpu().detach().numpy())
            print("Rows accuracy: ", rows_accuracy(assignment).cpu().detach().numpy())
            print("Columns accuracy: ", columns_accuracy(assignment).cpu().detach().numpy())
            print("Main vars  ", assignment[0, ...].cpu().detach().int().numpy())
            print("Label      ", reshaped_label[0, ...].cpu().detach().int().numpy())
            average_loss = 0
