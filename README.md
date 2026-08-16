 Gelecek Hedefleri ve Yol Haritası 

📰 Doğal Dil İşleme (NLP) ile Duyarlılık Analizi (Sentiment Analysis):

Plan: Hisse fiyatları sadece geçmiş sayılara değil, küresel haberlere ve şirket duyurularına da bağlıdır. Gelecek sürümde sisteme FinBERT veya Azure Bilişsel Hizmetler (Cognitive Services) gibi NLP modelleri entegre edilerek; finansal haber metinlerinin, KAP bildirimlerinin ve sosyal medya verilerinin anlık olarak taranması ve piyasa duyarlılığının (pozitif/negatif algı) tahminleme ağırlıklarına katılması planlanmaktadır.

📊 Çok Değişkenli (Multi-Variate) Zaman Serisi Analizi:

Plan: Şu anki mimari tek değişkenli (sadece Kapanış Fiyatı - Close) çalışmaktadır. Gelecek aşamada; İşlem Hacmi (Volume), RSI, MACD gibi teknik indikatörler ile faiz kararları gibi makroekonomik verilerin de matrise dahil edilerek modelin çok değişkenli bir yapıya (Multi-variate LSTM/GRU) geçirilmesi hedeflenmektedir.

☁️ Bulut Tabanlı Otomasyon ve CI/CD (Cloud Deployment):

Plan: Mevcut Docker mimarisinin, yerel makinelerden çıkarılarak bulut bilişim platformlarına (Örn: Azure Container Apps) taşınması. GitHub Actions ile bir CI/CD pipeline'ı kurularak, koda yapılan her yeni eklemenin (push) otomatik olarak test edilip canlı sunucuda güncellenmesi sağlanacaktır.

💼 Portföy Optimizasyonu ve Risk Skoru:

Plan: Sadece tek bir hissenin yönünü tahmin etmek yerine, kullanıcının seçtiği birden fazla hisse senedini analiz ederek, modern portföy teorisine (Markowitz) göre optimum ağırlık dağılımı ve "Risk/Ödül" skorlaması sunan yeni bir API uç noktası (endpoint) geliştirilecektir.

Bu eklenti, projeni sadece bir "öğrenci denemesi" olmaktan çıkarıp, adeta bir fintek (FinTech) startup'ının iş planına dönüştürür.

