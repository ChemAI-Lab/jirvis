import torch
import torch.nn as nn
import os


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=batch_first
        )
        self.hidden_proj = nn.Linear(input_size, hidden_size)
        self.cell_proj = nn.Linear(input_size, hidden_size)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        # Pool encoder memory
        # PLEASE ADD FOR FUTURE -> Currently the code is pooling the memory by averaging across the sequence length dimension.
        # Future implementation should include options for using the first or last token.
        pooled = memory.mean(dim=1)  # [B, D]
        # pooled = memory[:, 0, :]  # Use the first token
        # # or
        # pooled = memory[:, -1, :]  # Use the last token

        # Project to initial hidden and cell state
        h_0 = (
            torch.tanh(self.hidden_proj(pooled))
            .unsqueeze(0)
            .repeat(self.num_layers, 1, 1)
        )
        c_0 = (
            torch.tanh(self.cell_proj(pooled))
            .unsqueeze(0)
            .repeat(self.num_layers, 1, 1)
        )

        output, (h_n, c_n) = self.lstm(tgt, (h_0, c_0))
        return output, (h_n, c_n)
