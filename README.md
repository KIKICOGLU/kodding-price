📈 Fiyat Projeksiyon Motoru: Mikroservis Mimarisi ile Yapay Zeka (LSTM vs GRU)


Bu proje, Microsoft AI Innovators programı kapsamında derin öğrenme teknikleri kullanılarak borsa hisse senedi kapanış fiyatlarını tahmin etmek amacıyla sıfırdan geliştirilmiş, uçtan uca bir yazılım mimarisidir.

Proje standart bir makine öğrenmesi betiği olmaktan çıkarılmış; FastAPI ile Backend, Streamlit ile Frontend olarak ikiye ayrılmış ve Docker ile donanım bağımsız bir konteyner mimarisine taşınmıştır.

🚀 Sistem Mimarisi ve Kullanılan Teknolojiler
Sistem sürdürülebilirlik, ölçeklenebilirlik ve yüksek performans odaklı bir mikroservis yapısında tasarlanmıştır:

1. Backend Katmanı / API (api.py)
Teknoloji: FastAPI, PyTorch, yfinance, scikit-learn

Görev: Sistemin "Motor" kısmıdır. yfinance üzerinden finansal verileri çeker, ffill() ve dropna() algoritmalarıyla verideki "NaN" tatil boşluklarını temizler ve tensör boyutlandırmalarını yapar. Gelen istekler doğrultusunda LSTM ve GRU modellerini eğitip sonuçları JSON formatında arayüze servis eder.

2. Yapay Zeka Modelleri (models_v8.py)
LSTM (Long Short-Term Memory): Uzun vadeli trendleri hafızasında tutarak daha temkinli ve stabil projeksiyonlar çizen 3 kapılı model.

GRU (Gated Recurrent Unit): Daha hafif ve 2 kapılı yapısıyla piyasadaki ani hareketlere daha agresif ve hızlı tepki veren mimari.

Akıllı Fren (Early Stopping): Modelin veriyi ezberlemesini (overfitting) ve gradyan patlamalarını (Exploding Gradients) engellemek için sisteme özel bir adaptif durdurma mekanizması entegre edilmiştir.

3. Frontend Katmanı / Vitrin (appd.py)
Teknoloji: Streamlit, Plotly, Pandas, Requests

Görev: Kullanıcının sistemle etkileşime girdiği kurumsal finans paneli. HTTP/REST istekleri atarak API'den aldığı karmaşık matrisleri, Plotly kullanarak interaktif, yakınlaştırılabilir ve modern grafiklere dönüştürür.

4. DevOps ve Konteynerleştirme (Dockerfile & docker-compose.yml)
Görev: Sistemin Python sürümü veya kütüphane çakışması yaşamadan, dünyadaki herhangi bir bilgisayarda tek tıkla ve hatasız şekilde çalışmasını sağlayan izole ağ yapısı.

🛠️ Kurulum ve Çalıştırma Rehberi (Çok Kolay!)
Proje Docker mimarisine sahip olduğu için karmaşık kütüphane kurulumlarıyla veya sürüm çatışmalarıyla uğraşmanıza gerek yoktur. Sadece aşağıdaki adımları izlemeniz yeterlidir:

1. Depoyu Klonlayın:

Bash
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
cd REPO_ADINIZ
2. Docker Orkestrasyonunu Başlatın:
Bilgisayarınızda Docker (Docker Desktop) açıkken proje dizininde komut satırına sadece şu kodu yazın:

Bash
docker compose up --build
3. Sisteme Giriş Yapın:
Terminalde kurulum tamamlandığında, tarayıcınızı açın ve aşağıdaki adrese giderek kurumsal arayüze ulaşın:
👉 http://localhost:8501

📊 Öne Çıkan Mühendislik Yaklaşımları
Data Preprocessing & Zırhı: Resmi tatil günlerinin yarattığı veri boşlukları (NaN), sistemi çökertmemesi için ileriye dönük doldurma yöntemleriyle optimize edilmiştir.

Asenkron Mikroservis : Arayüz ve model eğitimi birbirinden ayrı portlarda (8501 ve 8000) çalıştırılarak işlem yükü izole edilmiştir.

Dinamik Hiperparametre Kontrolü: Kullanıcılar arayüz üzerinden modeli istedikleri Epoch sayısıyla veya "Akıllı Fren" aktif/pasif durumuyla özgürce test edebilirler.

