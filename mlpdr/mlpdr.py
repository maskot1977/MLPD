import torch.nn as nn


class MLPDR(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, p_dropout):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.relu(self.l1(x))
        # print(x)
        x = self.dropout(x)
        # print(x)
        x = self.l2(x)
        print(x)
        return x
