import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# veri çekme'!!!
hisse_kodu = "THYAO.IS"
print(f"[{hisse_kodu}] verileri Yahoo Finance üzerinden çekiliyor...\n")

# 10 yıllık veri indirme
df = yf.download(hisse_kodu, start="2016-07-01", end="2026-07-01")

# tarihler kenar başlığı (excelde soldaki 0,1,2,3) halinde gelir.buna düzeltme.
df.reset_index(inplace=True)

# 2. İlk 5 satır
print("--- Veri Setinin İlk 5 Satırı (df.head()) ---")
print(df.head())
print("\n" + "="*50 + "\n")

# 3. Eksik veri (NaN) kontrolü
print("--- Eksik Veri (NaN) Durumu ---")
eksik_veri = df.isnull().sum()
print(eksik_veri)

if eksik_veri.sum() == 0:
    print("\nVeri setinde hiç eksik veri (NaN) bulunmuyor.Temizlemeye gerek yok.")
else:
    print(f"\nDikkat: Veri setinde toplam {eksik_veri.sum()} adet eksik veri var.")
print("\n" + "="*50 + "\n")

# 4. grafik tablosu açma
plt.figure(figsize=(12, 6))

# X eksenine Tarih (Date), Y eksenine Kapanış (Close) çizgiye başlık, renk kalınlık ayarı
plt.plot(df['Date'], df['Close'], label=f'{hisse_kodu} Kapanış Fiyatı', color='tab:blue', linewidth=1.5)

# Grafik görüntü ayarları 
plt.title(f'{hisse_kodu} - Zaman İçindeki Kapanış Fiyatı Değişimi', fontsize=14, fontweight='bold')
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Fiyat (TL)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6) # Arka plana kılavuz çizgiler
plt.legend()

# ekrana ver
plt.tight_layout()
plt.show()