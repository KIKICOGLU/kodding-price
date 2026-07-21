import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Kendi yazdığımız modüller
from data_ingestion import veri_cek_ve_gorsellestir
from data_preprocessing import veriyi_hazirla
from models import LSTMModel, GRUModel

# ==========================================
# EĞİTİM DÖNGÜSÜ FONKSİYONU
# ==========================================
def modeli_egit(model, X_train, y_train, model_adi, epochs=50):
    print(f"\n--- {model_adi} Modeli Eğitimi Başlıyor ---")
    
    # 1. Hata Fonksiyonu (MSE) ve Optimizasyon (Adam)[cite: 1]
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Modeli eğitim moduna alıyoruz
    model.train()
    
    # Eğitim Döngüsü[cite: 1]
    for epoch in range(epochs):
        # a. Eski hataları sıfırla (PID'deki Integral birikimini temizle)
        optimizer.zero_grad()
        
        # b. İleri besleme: Modelin tahmini (Kameradan anlık konumu oku)[cite: 1]
        tahminler = model(X_train)
        
        # c. Hatayı hesapla: Hedef ile Tahmin arasındaki fark[cite: 1]
        loss = criterion(tahminler, y_train)
        
        # d. Geri yayılım: Hatanın kaynağını bul (Hangi motor ne kadar saptırdı?)[cite: 1]
        loss.backward()
        
        # e. Ağırlıkları güncelle: Sistemi düzelt (Yeni PWM sinyallerini gönder)[cite: 1]
        optimizer.step()
        
        # Her 10 döngüde bir gidişatı ekrana yazdır[cite: 1]
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Hata (Loss): {loss.item():.4f}")
            
    print(f"{model_adi} Eğitimi Tamamlandı!")
    return model

if __name__ == "__main__":
    # 1. Veriyi çek
    df_hisse = veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS", grafik_ciz=False)
    
    # 2. Veriyi modele uygun hale getir
    X_train, X_test, y_train, y_test, model_scaler = veriyi_hazirla(df=df_hisse, pencere_boyutu=20, egitim_orani=0.8)
    
    # 3. PyTorch Tensörlerine Dönüşüm
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    x_train_tensor = torch.from_numpy(X_train).type(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).type(torch.float32).view(-1, 1)
    
    x_test_tensor = torch.from_numpy(X_test).type(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).type(torch.float32).view(-1, 1)
    
    # 4. Modelleri Başlat
    lstm_model = LSTMModel(input_dim=1, hidden_size=32, num_layers=2, output_dim=1)
    gru_model = GRUModel(input_dim=1, hidden_size=32, num_layers=2, output_dim=1)
    
    # 5. EĞİTİMİ BAŞLAT[cite: 1]
    # İki modeli de 50'şer tur (epoch) eğitiyoruz.
    lstm_model = modeli_egit(lstm_model, x_train_tensor, y_train_tensor, "LSTM", epochs=50)
    gru_model = modeli_egit(gru_model, x_train_tensor, y_train_tensor, "GRU", epochs=50)
    
    
    # ==========================================
    # 6. TEST VE DEĞERLENDİRME AŞAMASI
    # ==========================================
    print("\n--- Test ve Değerlendirme Başlıyor ---")
    
    # Modelleri test moduna alıyoruz (Ağırlık güncellemeyi kilitleriz)
    lstm_model.eval()
    gru_model.eval()
    
    # torch.no_grad() ile gereksiz hafıza kullanımını kapatıyoruz
    with torch.no_grad():
        lstm_tahminler = lstm_model(x_test_tensor).numpy()
        gru_tahminler = gru_model(x_test_tensor).numpy()
    
    # Tahminleri -1 ve 1 aralığından gerçek fiyatlara (TL) çeviriyoruz
    lstm_tahminler_gercek = model_scaler.inverse_transform(lstm_tahminler)
    gru_tahminler_gercek = model_scaler.inverse_transform(gru_tahminler)
    
    # Gerçek test verisini de aynı şekilde orijinal fiyatlara çeviriyoruz
    y_test_gercek = model_scaler.inverse_transform(y_test_tensor.numpy())
    
    # ==========================================
    # HATA HESAPLAMA (RMSE)[cite: 1]
    # ==========================================
    from sklearn.metrics import mean_squared_error
    import math
    
    lstm_rmse = math.sqrt(mean_squared_error(y_test_gercek, lstm_tahminler_gercek))
    gru_rmse = math.sqrt(mean_squared_error(y_test_gercek, gru_tahminler_gercek))
    
    print(f"\n--- Sonuçlar (RMSE - Ortalama Sapma Miktarı) ---")
    print(f"LSTM Modeli Hatası: {lstm_rmse:.2f} TL")
    print(f"GRU Modeli Hatası:  {gru_rmse:.2f} TL")
    
    if gru_rmse < lstm_rmse:
        print("Sonuç: GRU modeli bu veri setinde daha iyi performans gösterdi![cite: 1]")
    else:
        print("Sonuç: LSTM modeli bu veri setinde daha iyi performans gösterdi!")

    # ==========================================
    # GRAFİK ÇİZİMİ (Karşılaştırma)[cite: 1]
    # ==========================================
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 6))
    
    # Gerçek fiyatlar (Siyah çizgi)
    plt.plot(y_test_gercek, label="Gerçek Fiyatlar", color="black", linewidth=2)
    
    # Model Tahminleri
    plt.plot(lstm_tahminler_gercek, label="LSTM Tahmini", color="blue", alpha=0.7)
    plt.plot(gru_tahminler_gercek, label="GRU Tahmini", color="red", alpha=0.7)
    
    plt.title("Borsa Fiyat Tahmini: Gerçek Fiyatlar vs. Model Tahminleri", fontsize=14, fontweight='bold')
    plt.xlabel("Test Günleri (Zaman)", fontsize=12)
    plt.ylabel("Kapanış Fiyatı (TL)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
