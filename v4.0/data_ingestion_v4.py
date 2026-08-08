import pandas as pd
import yfinance as yf

def veri_cek_ve_gorsellestir(hisse_kodu="THYAO.IS", grafik_ciz=False):
    print(f"{hisse_kodu} verileri indiriliyor (Özellik Mühendisliği: +SMA, +RSI)...")
    
    # Enflasyon/Data Drift sorunu yaşamamak için 1.5 yıllık stabil dönemi alıyoruz
    df = yf.download(hisse_kodu, start="2023-01-01", end="2024-07-01")
    
    df_ozellikler = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # --- YENİ EKLENEN FİNANSAL İNDİKATÖRLER (ÖZELLİK MÜHENDİSLİĞİ) ---
    
    # 1. SMA (14 Günlük Basit Hareketli Ortalama)
    df_ozellikler['SMA_14'] = df_ozellikler['Close'].rolling(window=14).mean()
    
    # 2. RSI (14 Günlük Göreceli Güç Endeksi)
    delta = df_ozellikler['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df_ozellikler['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Yeni hesaplamalarda ilk 14 gün geçmiş veri olmadığı için "Boş (NaN)" çıkar. Onları siliyoruz.
    df_ozellikler.dropna(inplace=True)
    
    return df_ozellikler