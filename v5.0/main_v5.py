import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import copy
import os
import random
import torch
import numpy as np
# Version 5 imports

# DİKKAT: V4 dosyalarını import ediyoruz
from data_ingestion_v5 import veri_cek_ve_gorsellestir
from data_preprocessing_v5 import veriyi_hazirla
from models_v5 import LSTMModel, GRUModel

# =========================================================
# 🔒 SİSTEMİ SABİTLEME (REPRODUCIBILITY)
# =========================================================
def sistemi_sabitle(seed=42):
    # DİKKAT: Buradaki her satır içeriden (indented) başlıyor!
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Sistem deterministik (sabit) moda alındı (Seed: {seed})")
    
def modeli_egit(model, X_train, y_train, model_adi, epochs=50, patience=15, akilli_fren=True):
    durum = "Açık (Hızlı)" if akilli_fren else "Kapalı (Tam Odak)"
    print(f"\n--- {model_adi} Eğitimi Başlıyor (Akıllı Fren: {durum}) ---")
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tahminler = model(X_train)
        loss = criterion(tahminler, y_train)
        loss.backward()
        optimizer.step()
        
        guncel_hata = loss.item()
        
        if akilli_fren:
            if guncel_hata < best_loss:
                best_loss = guncel_hata
                patience_counter = 0 
                best_model_weights = copy.deepcopy(model.state_dict()) 
            else:
                patience_counter += 1 
                
            if patience_counter >= patience:
                print(f"🛑 Akıllı Fren Devrede! Eğitim {epoch+1}. epoch'ta kesildi.")
                break 

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Hata (Loss): {guncel_hata:.4f}")
            
    if akilli_fren and best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"✨ En iyi model durumu (Loss: {best_loss:.4f}) başarıyla geri yüklendi.")
        
    return model

if __name__ == "__main__":
    # 1. KOD BAŞLAR BAŞLAMAZ İLK İŞ SİSTEMİ KİLİTLEYEN FONKSİYONU ÇAĞIRIYORUZ
    sistemi_sabitle(42)
    
    # 2. SONRA UYGULAMA ARAYÜZÜ BAŞLIYOR
    print("\n" + "="*50)
    print(" 🚀 HİSSE SENEDİ TAHMİN MODELİ v5.0 (+RSI, +SMA)")
    print("="*50)
    print("Lütfen eğitim modunu seçin:")
    print("[1] Hızlı ve Verimli (Akıllı Fren AÇIK - Önerilen)")
    print("[2] Maksimum Odak (Akıllı Fren KAPALI - Kesin sonuç, uzun sürer)")

    # ... (Kodunun geri kalanı tamamen aynı kalacak) ...
    secim = input("Seçiminiz (1 veya 2): ")
    
    if secim == '2':
        fren_durumu = False
        print("\n=> Uyarı: Akıllı Fren KAPATILDI. Model belirlenen tüm adımları (epoch) tamamlayacak.")
    else:
        fren_durumu = True
        print("\n=> Akıllı Fren AÇIK. Model en verimli noktasında otomatik durdurulacak.")
    print("="*50 + "\n")

    df_hisse = veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS")
    X_train, X_test, y_train, y_test, model_scaler = veriyi_hazirla(df=df_hisse, pencere_boyutu=20, tahmin_adimi=3)
    
    x_train_tensor = torch.from_numpy(X_train).type(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).type(torch.float32)
    x_test_tensor = torch.from_numpy(X_test).type(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).type(torch.float32)
    
    # DİKKAT: input_dim=7 olarak güncellendi (OHLCV + SMA + RSI)
    lstm_model = LSTMModel(input_dim=7, hidden_size=64, num_layers=2, output_dim=3)
    gru_model = GRUModel(input_dim=7, hidden_size=64, num_layers=2, output_dim=3)
    
    lstm_model = modeli_egit(lstm_model, x_train_tensor, y_train_tensor, "LSTM", epochs=50, patience=15, akilli_fren=fren_durumu)
    gru_model = modeli_egit(gru_model, x_train_tensor, y_train_tensor, "GRU", epochs=50, patience=15, akilli_fren=fren_durumu)

    # --- TEST VE ÇİZİM KISMI ---
    print("\n--- Test ve Değerlendirme Başlıyor ---")
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        lstm_tahminler = lstm_model(x_test_tensor).numpy()
        gru_tahminler = gru_model(x_test_tensor).numpy()
    
    lstm_tahminler_gercek = model_scaler.inverse_transform(lstm_tahminler)
    gru_tahminler_gercek = model_scaler.inverse_transform(gru_tahminler)
    y_test_gercek = model_scaler.inverse_transform(y_test_tensor.numpy())
    
    lstm_rmse = math.sqrt(mean_squared_error(y_test_gercek, lstm_tahminler_gercek))
    gru_rmse = math.sqrt(mean_squared_error(y_test_gercek, gru_tahminler_gercek))
    
    print(f"\n--- Sonuçlar (3 Günlük Toplam RMSE Sapması) ---")
    print(f"LSTM Modeli Hatası: {lstm_rmse:.2f} TL")
    print(f"GRU Modeli Hatası:  {gru_rmse:.2f} TL")
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_gercek[:, 0], label="Gerçek Fiyatlar (T+1)", color="black", linewidth=2)
    plt.plot(lstm_tahminler_gercek[:, 2], label="LSTM Tahmini (T+1)", color="blue", alpha=0.7)
    plt.plot(gru_tahminler_gercek[:, 2], label="GRU Tahmini (T+1)", color="red", alpha=0.7)
    
    plt.title("Özellik Mühendisliği (7 Değişken): Gerçek Fiyat vs Tahmin", fontsize=14, fontweight='bold')
    plt.xlabel("Test Günleri", fontsize=12)
    plt.ylabel("Kapanış Fiyatı (TL)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
    # =========================================================
# =========================================================
    # 🔮 V5.0 - CANLI TAHMİN MOTORU (GELECEK 3 GÜN) - DÜZELTİLMİŞ
    # =========================================================
    print("\n" + "="*50)
    print(" 🔮 CANLI TAHMİN MOTORU AKTİF (Önümüzdeki 3 Gün)")
    print("="*50)
    
    from sklearn.preprocessing import MinMaxScaler
    
    # 1. Eğitimde kullanılan doğru cetveli (Scaler) yeniden inşa ediyoruz
    veri_uzunlugu = len(df_hisse)
    egitim_siniri = int(veri_uzunlugu * 0.8) # Veriyi %80 eğitim sınırından kesiyoruz
    
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    # DİKKAT: fit() sadece eğitim verisiyle yapılır! Yeni rekorları (300+ TL) cetvele dahil etmiyoruz.
    X_scaler.fit(df_hisse.values[:egitim_siniri])
    
    # 2. Tüm veriyi bu DOĞRU cetvelle ölçeklendirip en güncel 20 günü alıyoruz
    tum_veri_scaled = X_scaler.transform(df_hisse.values)
    son_20_gun_scaled = tum_veri_scaled[-20:]
    
    # PyTorch'un anlayacağı 3 boyutlu tensöre çevir (Batch=1, Zaman=20, Özellik=7)
    canli_tensor = torch.from_numpy(son_20_gun_scaled).type(torch.float32).unsqueeze(0)
    
    # 3. Geleceği Tahmin Et
    with torch.no_grad():
        lstm_gelecek = lstm_model(canli_tensor).numpy()
        gru_gelecek = gru_model(canli_tensor).numpy()
        
    # 4. Çıkan sayıları gerçek TL'ye çevir
    lstm_gelecek_tl = model_scaler.inverse_transform(lstm_gelecek)[0]
    gru_gelecek_tl = model_scaler.inverse_transform(gru_gelecek)[0]
    
    print("\nBorsanın açılacağı önümüzdeki ilk 3 gün için fiyat beklentileri:\n")
    
    gunler = ["1. Gün (Yarın)", "2. Gün (Öbür Gün)", "3. Gün (Sonraki Gün)"]
    
    for i in range(3):
        print(f"--- {gunler[i]} ---")
        print(f"LSTM Modeli: {lstm_gelecek_tl[i]:.2f} TL")
        print(f"GRU Modeli : {gru_gelecek_tl[i]:.2f} TL")
        print("-" * 25)
        
    print("\n=> Not: Bu değerler yatırım tavsiyesi değildir, yapay zekanın matematiksel çıkarımlarıdır! 🚀\n")
    