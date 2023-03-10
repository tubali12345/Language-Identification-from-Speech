import math

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, device, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, dropout, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.dropout(self.norm1(attention + x))
        ff = self.ff(x)
        y = self.dropout(self.norm1(ff + x))
        return y


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, device, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_attention = nn.MultiheadAttention(d_model, heads, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerBlock(d_model, heads, device, forward_expansion, dropout)

    def forward(self, x):
        attention = self.masked_attention(x, x, x)[0]
        y = self.dropout(self.norm1(attention + x))
        out = self.transformer(y)
        return out


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=2048):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.device = device
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        return x + nn.Parameter(self.pe[:, :seq_len], requires_grad=False).to(self.device)


class ConvLayers(nn.Module):
    def __init__(self,
                 conv_dim: int,
                 kernel_size: int = 3,
                 n_mels: int = 80,
                 device: str = 'cuda:0'):
        super(ConvLayers, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_mels)
        self.conv1 = nn.Conv1d(n_mels, conv_dim, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(conv_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        bn = self.bn1(x)
        conv1 = self.relu(self.bn2(self.conv1(bn)))
        conv2 = self.relu(self.bn3(self.conv2(conv1)))
        return conv2


class MultipleDecoderModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 d_model: int,
                 heads: int,
                 no_layers: int,
                 device: str,
                 n_mels: int = 80,
                 len: int = 801,
                 forward_expansion: int = 4,
                 conv_size: int = 64,
                 dropout: float = 0.1):
        super(MultipleDecoderModel, self).__init__()
        self.conv_layer = ConvLayers(conv_size)
        self.trans_conv = nn.Linear(len, d_model)
        self.device = device
        self.layers = nn.Sequential()
        for i in range(no_layers):
            self.layers.add_module(f'decoder_layer{i + 1}',
                                   DecoderBlock(d_model, heads, device, forward_expansion, dropout))
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(d_model * conv_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        conv = self.conv_layer(x)
        trans_conv = self.trans_conv(conv)
        y = self.layers(trans_conv)
        out = self.fc_out(self.flatten(y))
        return out


def test():
    device = "cuda:0"
    net = MultipleDecoderModel(num_classes=4,
                               d_model=512,
                               heads=8,
                               no_layers=6,
                               device=device,
                               forward_expansion=4,
                               dropout=0.1).to(device)
    x = torch.rand((64, 80, 801)).to(device)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()
