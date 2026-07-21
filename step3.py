import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. ADIM: VERİ ÇEKME FONKSİYONU
# ==========================================
def veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS", baslangic="2016-07-01", bitis="2026-07-01", grafik_ciz=False):
    print(f"[{hisse_kodu}] verileri Yahoo Finance üzerinden çekiliyor...\n")
    df = yf.download(hisse_kodu, start=baslangic, end=bitis)
    df.reset_index(inplace=True)
    
    eksik_veri = df.isnull().sum()
    if eksik_veri.sum() == 0:
        print("Veri setinde hiç eksik veri (NaN) bulunmuyor.")
    else:
        print(f"Dikkat: Veri setinde toplam {eksik_veri.sum()} adet eksik veri var.")

    if grafik_ciz:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label=f'{hisse_kodu} Kapanış Fiyatı', color='tab:blue', linewidth=1.5)
        plt.title(f'{hisse_kodu} - Zaman İçindeki Kapanış Fiyatı Değişimi', fontsize=14, fontweight='bold')
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel('Fiyat (TL)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return df

# ==========================================
# 2. ADIM: VERİ ÖN İŞLEME FONKSİYONU
# ==========================================
def veriyi_hazirla(df, pencere_boyutu=20, egitim_orani=0.8):
    print("\n--- Veri Ön İşleme (Preprocessing) Başlıyor ---")
    
    # 1. Öznitelik Seçimi
    kapanis_verisi = df[['Close']].values 
    
    # 2. Normalizasyon (-1 ile 1 arası)[cite: 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    olceklenmis_veri = scaler.fit_transform(kapanis_verisi)
    
    # 3. Kayan Pencere (Sliding Window)[cite: 1]
    X, y = [], []
    for i in range(pencere_boyutu, len(olceklenmis_veri)):
        X.append(olceklenmis_veri[i - pencere_boyutu : i, 0])
        y.append(olceklenmis_veri[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # 4. Eğitim ve Test Olarak Ayırma[cite: 1]
    bolme_indeksi = int(len(X) * egitim_orani)
    X_train, X_test = X[:bolme_indeksi], X[bolme_indeksi:]
    y_train, y_test = y[:bolme_indeksi], y[bolme_indeksi:]
    
    print("Veri hazırlığı tamamlandı!")
    print(f"Eğitim Verisi Boyutu: X={X_train.shape}, y={y_train.shape}")
    print(f"Test Verisi Boyutu: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

# ==========================================
# ANA ÇALIŞTIRMA BLOĞU (MAIN)
# ==========================================
if __name__ == "__main__":
    # 1. Veriyi çek
    df_hisse = veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS", grafik_ciz=False)
    
    # 2. Veriyi modele uygun hale getir
    X_train, X_test, y_train, y_test, model_scaler = veriyi_hazirla(df=df_hisse, pencere_boyutu=20, egitim_orani=0.8)