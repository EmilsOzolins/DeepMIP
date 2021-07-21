import torch.sparse
from torch.utils.data import DataLoader

from data.sudoku import SudokuDataset
from model.mip_network import MIPNetwork

if __name__ == '__main__':
    dataset = SudokuDataset("binary/sudoku.csv")
    # TODO: Implement batching
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    bit_count = 4
    steps = 10000

    network = MIPNetwork(bit_count)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    powers_of_two = torch.tensor([2 ** k for k in range(0, bit_count)], dtype=torch.float32)

    for step, (feature, label) in zip(range(steps), dataloader):
        optimizer.zero_grad()
        adj_matrix, b_values = feature
        adj_matrix = adj_matrix.coalesce()
        adj_matrix = torch.sparse_coo_tensor(adj_matrix.indices()[1:, :], adj_matrix.values(), size=adj_matrix.size()[1:])

        assignment = network.forward(adj_matrix, b_values)
        assignment = torch.sum(powers_of_two * assignment, dim=-1, keepdim=True)

        loss = torch.relu(torch.squeeze(torch.sparse.mm(adj_matrix.t(), assignment)) - torch.squeeze(b_values, axis=0))
        loss = torch.sum(loss)

        loss.backward()
        optimizer.step()
        print(loss.detach().numpy(), torch.round(torch.squeeze(assignment)).detach().numpy())
