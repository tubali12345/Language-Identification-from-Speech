import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim, num_lang, dropout=0.1):
        super(CNN_LSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # reduce max operation
        self.tdnn3 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.lstm6 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.tdnn7 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.tdnn8 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.atten_stat_pooling = AttensiveStatisticsPooling(inputdim=1500, outputdim=1500)
        self.fn9 = nn.Linear(3000, 512)
        self.fn10 = nn.Linear(512, 512)
        self.fn11 = nn.Linear(512, num_lang)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = torch.max(x, dim=1, keepdim=False).values
        x = F.relu(self.tdnn3(x))
        x = F.relu(self.tdnn4(x))
        x = F.relu(self.tdnn5(x))
        x, _ = self.lstm6(x.transpose(1, 2))
        x = F.relu(self.tdnn7(x.transpose(1, 2)))
        x = F.relu(self.tdnn8(x))
        stat = self.atten_stat_pooling(x)
        x = F.relu(self.fn9(stat))
        x = F.relu(self.fn10(x))
        output = self.fn11(x)
        return output


class AttensiveStatisticsPooling(nn.Module):
    def __init__(self, inputdim, outputdim, attn_dropout=0.0):
        super(AttensiveStatisticsPooling, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.attn_dropout = attn_dropout
        self.linear_projection = nn.Linear(inputdim, outputdim)
        self.v = torch.nn.Parameter(torch.randn(outputdim))

    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance

    def stat_attn_pool(self, inputs, attention_weights):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, dim=1)
        variance = self.weighted_sd(inputs, attention_weights, mean)
        stat_pooling = torch.cat((mean, variance), 1)
        return stat_pooling

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        lin_out = self.linear_projection(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.relu(lin_out.bmm(v_view).squeeze(2))
        attention_weights = F.softmax(attention_weights, dim=1)
        statistics_pooling_out = self.stat_attn_pool(inputs, attention_weights)
        return statistics_pooling_out
