from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
import random
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from models_v8 import LSTMModel, GRUModel

def sistemi_sabitle(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

app = FastAPI(title="Fiyat Projeksiyon Motoru")

# SİPARİŞ LİSTESİNE AKILLI FREN EKLENDİ
class TahminIstegi(BaseModel):
    hisse_kodu: str = "THYAO.IS"
    pencere_boyutu: int = 20
    epoch_sayisi: int = 50
    akilli_fren: bool = True

# EĞİTİM FONKSİYONUNA ERKEN DURDURMA MANTIĞI EKLENDİ
def modeli_egit_api(model, X_train, y_train, epochs, akilli_fren):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    en_iyi_loss = float('inf')
    sabir = 0
    patience = 15
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        tahminler = model(X_train)
        loss = criterion(tahminler, y_train)
        loss.backward()
        optimizer.step()
        
        # Akıllı Fren Kontrolü
        if akilli_fren:
            guncel_loss = loss.item()
            if guncel_loss < en_iyi_loss:
                en_iyi_loss = guncel_loss
                sabir = 0
            else:
                sabir += 1
                if sabir >= patience:
                    break
    return model

@app.post("/tahmin_et")
def tahmin_yap(istek: TahminIstegi):
    try:
        sistemi_sabitle(42)
        
        df_hisse = yf.download(istek.hisse_kodu, period="2y")
        if df_hisse.empty:
            raise ValueError(f"{istek.hisse_kodu} için veri bulunamadı.")
            
        df_hisse = df_hisse.ffill().dropna()
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        close_degerleri = np.array(df_hisse['Close']).reshape(-1, 1)
        scaled_data = scaler.fit_transform(close_degerleri)
        
        X, y = [], []
        for i in range(len(scaled_data) - istek.pencere_boyutu - 3):
            X.append(scaled_data[i:(i + istek.pencere_boyutu)])
            y.append(scaled_data[(i + istek.pencere_boyutu):(i + istek.pencere_boyutu + 3), 0])
            
        X = torch.from_numpy(np.array(X)).type(torch.float32)
        y = torch.from_numpy(np.array(y)).type(torch.float32)
        
        lstm = LSTMModel(input_dim=1, hidden_size=64, num_layers=2, output_dim=3)
        gru = GRUModel(input_dim=1, hidden_size=64, num_layers=2, output_dim=3)
        
        # Eğitim fonksiyonuna "akilli_fren" parametresini gönderiyoruz
        lstm = modeli_egit_api(lstm, X, y, istek.epoch_sayisi, istek.akilli_fren)
        gru = modeli_egit_api(gru, X, y, istek.epoch_sayisi, istek.akilli_fren)
        
        son_pencere = scaled_data[-istek.pencere_boyutu:]
        canli_tensor = torch.from_numpy(son_pencere).type(torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            lstm_gelecek = lstm(canli_tensor).numpy()
            gru_gelecek = gru(canli_tensor).numpy()
            
        lstm_gelecek_tl = scaler.inverse_transform(lstm_gelecek)[0]
        gru_gelecek_tl = scaler.inverse_transform(gru_gelecek)[0]
        
        if np.isnan(lstm_gelecek_tl).any() or np.isnan(gru_gelecek_tl).any():
            raise ValueError("Eğitim sırasında dengesizlik oluştu. Lütfen Akıllı Fren'i aktif edin veya Epoch'u düşürün.")

        return {
            "hisse": istek.hisse_kodu,
            "lstm_tahminleri": lstm_gelecek_tl.tolist(),
            "gru_tahminleri": gru_gelecek_tl.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))