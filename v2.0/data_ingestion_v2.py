import pandas as pd
import yfinance as yf

def veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS", grafik_ciz=False):
    print(f"{hisse_kodu} verileri indiriliyor (Çok Değişkenli)...")
    
    # 10 yıllık veri
    # 10 yıllık yerine sadece son dönemi (daha stabil enflasyon dönemi) alıyoruz
    df = yf.download(hisse_kodu, start="2023-01-01", end="2024-07-01")
    
    # Sadece kapanış değil, 5 temel borsa değişkenini alıyoruz
    df_ozellikler = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Eksik (NaN) verileri temizle
    df_ozellikler.dropna(inplace=True)
    
    return df_ozellikler