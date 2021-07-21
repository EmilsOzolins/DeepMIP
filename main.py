import torch.sparse
from torch.utils.data import DataLoader

from data.sudoku import SudokuDataset
from model.mip_network import MIPNetwork


def batch_graphs(batch):
    features = [x for x, _ in batch]
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

    return (batch_adj, torch.cat(const_values, dim=-1)), torch.stack(labels, dim=0)


if __name__ == '__main__':
    dataset = SudokuDataset("binary/sudoku.csv")
    # TODO: Implement batching
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=batch_graphs)

    bit_count = 4
    steps = 10000

    network = MIPNetwork(bit_count)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    powers_of_two = torch.tensor([2 ** k for k in range(0, bit_count)], dtype=torch.float32)

    average_loss = 0

    for step, ((adj_matrix, b_values), label) in zip(range(steps), dataloader):
        optimizer.zero_grad()

        assignment = network.forward(adj_matrix, b_values)
        assignment = torch.sum(powers_of_two * assignment, dim=-1, keepdim=True)

        loss = torch.relu(torch.squeeze(torch.sparse.mm(adj_matrix.t(), assignment)) - b_values)
        loss = torch.sum(loss)

        loss.backward()
        optimizer.step()

        average_loss += loss.detach()
        if step % 500 == 0:
            print("Step: ", step, "Avg. Loss:", (average_loss / 500.).numpy())
            print("Last output", torch.round(torch.squeeze(assignment)).detach().numpy())
            average_loss = 0
