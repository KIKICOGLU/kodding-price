import pandas as pd
import yfinance as yf

#veri çekme
hisse_kodu = "THYAO.IS"
baslangic_tarihi = "2014-07-01"
bitis_tarihi = "2024-07-01"

print(f"{hisse_kodu} verileri indiriliyor...")
df = yf.download(hisse_kodu, start=baslangic_tarihi, end=bitis_tarihi)

# kapanış değerlerini görmek için
df_kapanis = df[['Close']]
print("\nİlk 5 Günün Kapanış Fiyatları:")
print(df_kapanis.head())

#veriyi cvs kayıt
df.to_csv("thyao_10_yillik_veri.csv")
print("\nVeri başarıyla 'thyao_10_yillik_veri.csv' olarak kaydedildi!")

#ilk adım atılmıştır. İstenen kütüphanelerin kontrolleri yapılmış ardından yfinance kütüphanesi üzerinden THY'nin verileri çekilerek  doğrulama sağlanmıştır.