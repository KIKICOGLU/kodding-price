import yfinance as yf
import matplotlib.pyplot as plt

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