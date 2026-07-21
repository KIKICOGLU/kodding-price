# 📈 Borsa Fiyat Tahmini: PyTorch ile LSTM ve GRU Karşılaştırması

Bu proje, derin öğrenme (Deep Learning) teknikleri kullanarak zaman serisi verileri üzerinden borsa hisse senedi kapanış fiyatlarını tahmin etmeyi amaçlamaktadır. Proje kapsamında **PyTorch** kullanılarak iki farklı Tekrarlayan Sinir Ağı (RNN) mimarisi olan **LSTM (Long Short-Term Memory)** ve **GRU (Gated Recurrent Unit)** sıfırdan inşa edilmiş, eğitilmiş ve performansları gerçek veriler üzerinde kıyaslanmıştır.

---

## 🚀 Proje Adımları ve Mimari

Proje, sürdürülebilirlik ve kod okunabilirliği açısından modüler bir yapıda tasarlanmıştır. Sistem 4 ana adımdan (ve dosyadan) oluşur:

### Adım 1: Veri Toplama (`data_ingestion.py`)
* **Ne Yapıyor?** `yfinance` kütüphanesi kullanılarak Yahoo Finance API'sine bağlanılır.
* **Detay:** Belirtilen hisse senedinin (örn: THYAO.IS) geçmiş 10 yıllık borsa hareketleri indirilir. Eksik veya hatalı veriler (NaN) temizlenerek veri seti analize hazır hale getirilir.

### Adım 2: Veri Ön İşleme (`data_preprocessing.py`)
* **Ne Yapıyor?** Ham veriler makine öğrenmesi modelinin anlayabileceği matematiksel tensörlerin temeline oturtulur.
* **Detay:** 
  * Veriler ağın ağırlıklarını bozmaması için `MinMaxScaler` ile **-1 ile 1** arasına ölçeklenir (Normalization).
  * **Kayan Pencere (Sliding Window):** Algoritmanın geçmişe bakarak öğrenmesi için veriler 20 günlük paketler (pencereler) halinde dilimlenir. Sistem, geçmiş 20 güne bakarak 21. günü tahmin edecek şekilde (X ve y matrisleri) ayarlanır.
  * Veri kronolojik sıra bozulmadan %80 Eğitim (Train) ve %20 Test (Test) olarak ikiye ayrılır.

### Adım 3: Model İnşası (`models.py`)
* **Ne Yapıyor?** PyTorch `nn.Module` tabanlı yapay sinir ağı mimarileri kurulur.
* **Detay:**
  * **LSTM Modeli:** 3 kapılı (gate) yapısı ve uzun/kısa süreli hafıza hücreleriyle zaman serilerindeki uzun vadeli ilişkileri öğrenmek üzere tasarlanmıştır.
  * **GRU Modeli:** 2 kapılı, daha hafif yapısıyla sadece gizli durumu (hidden state) kullanarak işlem yapar.
  * *Hiperparametreler:* Her iki model de 1 girdi boyutu, 32 gizli nöron, 2 katman ve 1 çıktı boyutu ile yapılandırılmıştır.

### Adım 4: Eğitim ve Değerlendirme (`main.py`)
* **Ne Yapıyor?** Sistemin orkestrasyonunu sağlar.
* **Detay:** Matrisler PyTorch Tensörlerine (3 boyutlu) dönüştürülür. Modeller `Adam` optimizasyon algoritması ve `MSELoss` (Ortalama Kare Hatası) kullanılarak 50 epoch boyunca eğitilir. Eğitilen modeller test verisiyle sınanarak tahminleri grafik üzerine aktarılır.

---

## 📊 Test Sonuçları ve Kıyaslama

Projeyi çalıştırdığınızda modeller test verisi üzerinde sınanır ve **RMSE (Kök Ortalama Kare Hatası)** değerleri hesaplanır. 

**Model Kıyaslama Analizi:**
* Yapılan testler sonucunda **GRU modelinin**, LSTM'e kıyasla gerçek fiyat hareketlerini daha düşük hata payı (RMSE) ile yakaladığı gözlemlenmiştir.
* **Nedeni:** Kullanılan veri setinin tek değişkenli (sadece Kapanış fiyatı) ve nispeten küçük olması, kompleks (çok parametreli) LSTM yapısında hafif bir ezberlemeye (overfitting) yol açarken; daha sade bir mimariye sahip olan GRU modeli, ana trendi çok daha başarılı bir şekilde genellemiştir.

---

## 🛠️ Kurulum Rehberi

Projeyi kendi ortamınızda çalıştırmak için sırasıyla aşağıdaki adımları izleyin:

**1. Depoyu Klonlayın:**
```bash
git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
cd REPO_ADINIZ
```

2. Gerekli Kütüphaneleri Yükleyin:
Terminal veya komut satırında aşağıdaki komutu çalıştırarak bağımlılıkları indirin:
```bash
pip install torch pandas numpy scikit-learn matplotlib yfinance
```

3. Projeyi Çalıştırın:
Ana dosyayı çalıştırarak veri çekme, eğitim ve test süreçlerini başlatın:
```bash
python main.py
```

