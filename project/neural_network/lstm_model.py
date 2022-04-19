import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_cnt, device, batch_first=True, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if bidirectional:
            self.bidirectional_multiplier = 2
        else:
            self.bidirectional_multiplier = 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.fully_connected = nn.Linear(hidden_size * self.bidirectional_multiplier, output_cnt)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * self.bidirectional_multiplier,
            x.size(0),
            self.hidden_size
        ).to(self.device)
        c0 = torch.zeros(
            self.num_layers * self.bidirectional_multiplier,
            x.size(0),
            self.hidden_size
        ).to(self.device)

        output, _ = self.lstm(x, (h0, c0))
        output = self.fully_connected(output[:, -1, :])

        return output
