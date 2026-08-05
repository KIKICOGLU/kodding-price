import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

from data_ingestion_v2 import veri_cek_ve_gorsellestir
from data_preprocessing_v2 import veriyi_hazirla
from models_v2 import LSTMModel, GRUModel

def modeli_egit(model, X_train, y_train, model_adi, epochs=50):
    print(f"\n--- {model_adi} Eğitimi Başlıyor (Çok Değişkenli & 3 Günlük Tahmin) ---")
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tahminler = model(X_train)
        loss = criterion(tahminler, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Hata (Loss): {loss.item():.4f}")
            
    return model

if __name__ == "__main__":
    # 1. Veriyi çek
    df_hisse = veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS")
    
    # 2. Veriyi Hazırla (pencere=20 gün, tahmin=3 gün)
    X_train, X_test, y_train, y_test, model_scaler = veriyi_hazirla(df=df_hisse, pencere_boyutu=20, tahmin_adimi=3)
    
    # 3. PyTorch Tensörlerine Dönüşüm
    x_train_tensor = torch.from_numpy(X_train).type(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).type(torch.float32)
    
    x_test_tensor = torch.from_numpy(X_test).type(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).type(torch.float32)
    
    # 4. Modelleri Başlat (Girdi: 5 özellik, Çıktı: 3 gün)
    lstm_model = LSTMModel(input_dim=5, hidden_size=64, num_layers=2, output_dim=3)
    gru_model = GRUModel(input_dim=5, hidden_size=64, num_layers=2, output_dim=3)
    
    # 5. Eğitimi Başlat
    lstm_model = modeli_egit(lstm_model, x_train_tensor, y_train_tensor, "LSTM", epochs=50)
    gru_model = modeli_egit(gru_model, x_train_tensor, y_train_tensor, "GRU", epochs=50)

    # 6. TEST VE DEĞERLENDİRME
    print("\n--- Test ve Değerlendirme Başlıyor ---")
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        lstm_tahminler = lstm_model(x_test_tensor).numpy()
        gru_tahminler = gru_model(x_test_tensor).numpy()
    
    # -1 ve 1 aralığındaki 3 günlük tahmin dizilerini TL'ye çeviriyoruz
    lstm_tahminler_gercek = model_scaler.inverse_transform(lstm_tahminler)
    gru_tahminler_gercek = model_scaler.inverse_transform(gru_tahminler)
    y_test_gercek = model_scaler.inverse_transform(y_test_tensor.numpy())
    
    # Tüm 3 günün ortalama RMSE hatası
    lstm_rmse = math.sqrt(mean_squared_error(y_test_gercek, lstm_tahminler_gercek))
    gru_rmse = math.sqrt(mean_squared_error(y_test_gercek, gru_tahminler_gercek))
    
    print(f"\n--- Sonuçlar (3 Günlük Toplam RMSE Sapması) ---")
    print(f"LSTM Modeli Hatası: {lstm_rmse:.2f} TL")
    print(f"GRU Modeli Hatası:  {gru_rmse:.2f} TL")
    
    # GRAFİK ÇİZİMİ
    plt.figure(figsize=(14, 6))
    
    # Sadece dizinin 0. indeksini (T+1 yani yarının) tahminlerini grafiğe döküyoruz
    plt.plot(y_test_gercek[:, 0], label="Gerçek Fiyatlar (T+1)", color="black", linewidth=2)
    plt.plot(lstm_tahminler_gercek[:, 0], label="LSTM Tahmini (T+1)", color="blue", alpha=0.7)
    plt.plot(gru_tahminler_gercek[:, 0], label="GRU Tahmini (T+1)", color="red", alpha=0.7)
    
    plt.title("Çoklu Değişken (5 Özellik): Gerçek Fiyat vs Tahmin (1. Gün Çıktısı)", fontsize=14, fontweight='bold')
    plt.xlabel("Test Günleri", fontsize=12)
    plt.ylabel("Kapanış Fiyatı (TL)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()