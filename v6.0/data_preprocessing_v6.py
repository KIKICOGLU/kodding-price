import numpy as np
from sklearn.preprocessing import MinMaxScaler

def veriyi_hazirla(df, pencere_boyutu=20, tahmin_adimi=3, egitim_orani=0.8):
    # Tüm matrisi (5 sütun) -1 ile 1 arasına ölçeklemek için
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    
    # Sadece Kapanış fiyatını sonradan TL'ye çevirebilmek için bağımsız bir scaler
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    veri = df.values
    veri_X = scaler_X.fit_transform(veri)
    
    # Kapanış (Close) sütununun indeksi 3'tür (Open=0, High=1, Low=2, Close=3, Volume=4)
    kapanis_indeksi = 3
    scaler_y.fit(veri[:, kapanis_indeksi].reshape(-1, 1))

    X, y = [], []
    
    # Kayan pencere (Sliding window)
    for i in range(len(veri_X) - pencere_boyutu - tahmin_adimi + 1):
        # X: 20 günlük 5 farklı değişken verisi
        X.append(veri_X[i:(i + pencere_boyutu)])
        # y: Sadece Kapanış fiyatının (indeks 3) önümüzdeki 3 günlük (tahmin_adimi) verisi
        y.append(veri_X[(i + pencere_boyutu):(i + pencere_boyutu + tahmin_adimi), kapanis_indeksi])
    
    X = np.array(X)
    y = np.array(y)
    
    # Veriyi Eğitim ve Test olarak ikiye ayırma
    egitim_siniri = int(len(X) * egitim_orani)
    
    X_train, X_test = X[:egitim_siniri], X[egitim_siniri:]
    y_train, y_test = y[:egitim_siniri], y[egitim_siniri:]
    
    return X_train, X_test, y_train, y_test, scaler_y