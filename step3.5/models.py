import torch
import torch.nn as nn

# ==========================================
# LSTM MODELİ MİMARİSİ
# ==========================================
class LSTMModel(nn.Module):
    # Planda belirtilen hiperparametreler: input_dim=1, output_dim=1, hidden_size=32, num_layers=2
    def __init__(self, input_dim=1, hidden_size=32, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Katman: Ana LSTM bloğu
        # batch_first=True komutu, verinin bize (Paket Sayısı, Kayan Pencere, Özellik Sayısı) sırasında geleceğini söyler.
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        
        # 2. Katman: Çıkış Pimi (Linear)
        # LSTM'den çıkan karmaşık veriyi tek bir fiyat tahminine (output_dim=1) indirger.
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Her yeni veri paketi geldiğinde modelin kısa süreli (h0) ve uzun süreli (c0) hafızasını sıfırdan başlatıyoruz[cite: 1].
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Veriyi LSTM'den geçiriyoruz
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # 20 günlük pencerenin SADECE en sonundaki günün çıktısını alıp Linear katmana yolluyoruz
        out = self.fc(out[:, -1, :]) 
        return out


# ==========================================
# GRU MODELİ MİMARİSİ
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_size=32, num_layers=2, output_dim=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Katman: Ana GRU bloğu
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        
        # 2. Katman: Çıkış Pimi (Linear)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # GRU'nun uzun süreli hafıza (Cell State - c0) kanalı yoktur, sadece h0 vardır.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Veriyi GRU'dan geçiriyoruz
        out, hn = self.gru(x, h0.detach())
        
        out = self.fc(out[:, -1, :]) 
        return out