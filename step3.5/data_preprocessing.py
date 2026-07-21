import numpy as np
from sklearn.preprocessing import MinMaxScaler

def veriyi_hazirla(df, pencere_boyutu=20, egitim_orani=0.8):
    print("\n--- Veri Ön İşleme (Preprocessing) Başlıyor ---")
    
    kapanis_verisi = df[['Close']].values 
    
    # 2. Normalizasyon (-1 ile 1 arası)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    olceklenmis_veri = scaler.fit_transform(kapanis_verisi)
    
    # 3. Kayan Pencere (Sliding Window)
    X, y = [], []
    for i in range(pencere_boyutu, len(olceklenmis_veri)):
        X.append(olceklenmis_veri[i - pencere_boyutu : i, 0])
        y.append(olceklenmis_veri[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # 4. Eğitim ve Test Olarak Ayırma
    bolme_indeksi = int(len(X) * egitim_orani)
    X_train, X_test = X[:bolme_indeksi], X[bolme_indeksi:]
    y_train, y_test = y[:bolme_indeksi], y[bolme_indeksi:]
    
    print("Veri hazırlığı tamamlandı!")
    print(f"Eğitim Verisi Boyutu: X={X_train.shape}, y={y_train.shape}")
    print(f"Test Verisi Boyutu: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler