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
    m = torch.maximum(torch.zeros_like(inputs), inputs)
    return torch.minimum(torch.ones_like(m), m)


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

        assignment = network.forward(adj_matrix, b_values)
        int_loss = torch.square(assignment) * torch.square(1 - assignment)

        assignment = torch.sum(powers_of_two * assignment, dim=-1, keepdim=True)

        loss = relu1(torch.squeeze(torch.sparse.mm(adj_matrix.t(), assignment)) - b_values)
        loss = torch.sum(loss) + torch.sum(int_loss)

        loss.backward()
        optimizer.step()

        average_loss += loss.detach()
        if step % 500 == 0:
            assignment = torch.round(torch.squeeze(assignment))
            assignment = torch.reshape(assignment, [batch_size, 9, 9, 9])
            assignment = torch.argmax(assignment, dim=-1) + 1

            print("Step: ", step, "Avg. Loss:", (average_loss / 500.).cpu().numpy())
            print("Main vars  ", assignment[0, ...].cpu().detach().int().numpy())
            print("Label      ", torch.reshape(label, [batch_size, 9, 9])[0, ...].cpu().detach().int().numpy())
            average_loss = 0
