import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    # DİKKAT: input_dim 5'ten 7'ye çıktı
    def __init__(self, input_dim=7, hidden_size=64, num_layers=2, output_dim=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

class GRUModel(nn.Module):
    # DİKKAT: input_dim 5'ten 7'ye çıktı
    def __init__(self, input_dim=7, hidden_size=64, num_layers=2, output_dim=3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) 
        return out