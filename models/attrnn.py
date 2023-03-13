import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class AttRNN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 d_model: int,
                 n_heads: int,
                 conv_size: int,
                 kernel_size: int = 3,
                 n_mels: int = 80,
                 len: int = 801,
                 device: str = 'cuda:0',
                 dropout=0.25):
        super(AttRNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_mels)
        self.conv1 = nn.Conv1d(n_mels, conv_size, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_size)
        self.conv2 = nn.Conv1d(conv_size, conv_size, kernel_size=kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(conv_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=len, hidden_size=64, batch_first=True, num_layers=2, bidirectional=True)
        self.lambd = LambdaLayer(lambda q: q[:, -1])
        self.fc1 = nn.Linear(64 * 2, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.fc3 = nn.Linear(d_model // 2, d_model // 4)
        self.fc4 = nn.Linear(d_model // 4, d_model // 8)
        self.out = nn.Linear(d_model // 8, num_classes)
        self.device = device

    def forward(self, x):
        bn = self.bn1(x)
        conv1 = self.relu(self.bn2(self.conv1(bn)))
        conv2 = self.relu(self.bn3(self.conv2(conv1)))
        lstm = self.lstm(conv2)[0]
        last = self.lambd(lstm)
        fc1 = self.fc1(last)
        attention = self.attention(fc1, fc1, fc1)[0]
        fc2 = self.dropout(self.relu(self.fc2(attention)))
        fc3 = self.dropout(self.relu(self.fc3(fc2)))
        fc4 = self.dropout(self.relu(self.fc4(fc3)))
        out = self.out(fc4)
        return out


def test():
    device = "cuda:1"
    net = AttRNN(num_classes=3, d_model=512, n_heads=4, conv_size=32, device=device).to(device)
    x = torch.rand((64, 80, 801)).to(device)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()
