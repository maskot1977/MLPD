import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset



class MLPDR2(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, p_dropout1, p_dropout2):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden1)
        self.l2 = nn.Linear(n_hidden1, n_hidden2)
        self.l3 = nn.Linear(n_hidden2, n_output)
        self.relu = nn.ReLU()
        self.d1 = nn.Dropout(p_dropout1)
        self.d2 = nn.Dropout(p_dropout2)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.d1(x)
        x = self.relu(self.l2(x))
        x = self.d2(x)
        x = self.l3(x)
        return x


class Objective:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=100, shuffle=True
        )
        self.n_h1 = [1, 500]
        self.n_h2 = [1, 500]
        self.lr = [1e-10, 1e-1]
        self.p_dropout1 = [0, 0.99]
        self.p_dropout2 = [0, 0.99]
        self.n_epoch = [1, 500]
        self.best_model = None
        self.best_score = None
        self.train_loss_history = []
        self.test_loss_history = []
        self.n_trial = 0

    def __call__(self, trial):
        self.n_trial += 1
        n_epoch, = trial.suggest_int('n_epoch', self.n_epoch[0], self.n_epoch[1]),
        n_h1, = trial.suggest_int('n_hidden1', self.n_h1[0], self.n_h1[1]),
        n_h2, = trial.suggest_int('n_hidden2', self.n_h2[0], self.n_h2[1]),
        lr, = trial.suggest_loguniform('lr', self.lr[0], self.lr[1]),
        p_dropout1,  = trial.suggest_uniform('p_dropout1', self.p_dropout1[0], self.p_dropout1[1]),
        p_dropout2,  = trial.suggest_uniform('p_dropout2', self.p_dropout2[0], self.p_dropout2[1]),

        model = MLPDR2(self.x_train.shape[1], n_h1, n_h2, self.y_train.shape[1], p_dropout1, p_dropout2)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(n_epoch):
            total_loss = 0
            for x, y in self.train_loader:
                x = Variable(x)
                y = Variable(y)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.train_loss_history.append(total_loss)

        test_loss = criterion(model(X_test), Y_test)
        self.test_loss_history.append(loss.detach().numpy())

        if self.best_score is None or self.best_score > loss:
            self.best_score = loss
            self.best_model = model
            print("n_trial=", self.n_trial)
            print(loss)
            print(model)

        return loss 
