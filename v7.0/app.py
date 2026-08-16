import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Sayfa Yapılandırması (Geniş ekran, temiz sekme ismi)
st.set_page_config(page_title="Fiyat Projeksiyon Sistemi", layout="wide")

# Kurumsal CSS Stilleri
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 300; }
    .stMetric { border-left: 3px solid #0066cc; padding-left: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("Fiyat Projeksiyon Sistemi")
st.markdown("Derin Öğrenme Tabanlı Zaman Serisi Analizi | Microsoft AI Innovators Projesi")
st.divider()

# Ekranı iki ana sütuna bölüyoruz (Sol menü, sağ sonuç ekranı)
sol_panel, sag_panel = st.columns([1, 3])

with sol_panel:
    st.subheader("Model Parametreleri")
    hisse = st.text_input("Hisse Sembolü", value="THYAO.IS").upper()
    
    pencere = st.number_input("Gözlem Penceresi (Gün)", min_value=10, max_value=100, value=20, step=5)
    epoch = st.slider("Eğitim İterasyonu (Epoch)", min_value=10, max_value=200, value=50, step=10)
    
    akilli_fren = st.toggle("Akıllı Fren (Early Stopping)", value=True, help="Modelin gereksiz ezber yapmasını (overfitting) engeller.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    baslat_btn = st.button("Analizi Başlat", use_container_width=True, type="primary")

with sag_panel:
    if baslat_btn:
        with st.spinner(f"Sistem {hisse} verilerini işliyor ve ağ ağırlıklarını güncelliyor..."):
            api_url = "http://127.0.0.1:8000/tahmin_et"
            payload = {
                "hisse_kodu": hisse,
                "pencere_boyutu": int(pencere),
                "epoch_sayisi": int(epoch),
                "akilli_fren": akilli_fren
            }
            
            try:
                cevap = requests.post(api_url, json=payload)
                if cevap.status_code == 200:
                    veri = cevap.json()
                    
                    df_sonuc = pd.DataFrame({
                        "Periyot": ["T+1 (Yarın)", "T+2", "T+3"],
                        "LSTM (Temkinli)": veri["lstm_tahminleri"],
                        "GRU (Agresif)": veri["gru_tahminleri"]
                    })
                    
                    # 1. Gösterge Kartları (Metrikler)
                    st.subheader(f"Gelecek 3 Günlük Tahmin: {hisse}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("T+1 LSTM Beklentisi", f"{veri['lstm_tahminleri'][0]:.2f} ₺")
                    m2.metric("T+2 LSTM Beklentisi", f"{veri['lstm_tahminleri'][1]:.2f} ₺")
                    m3.metric("T+3 LSTM Beklentisi", f"{veri['lstm_tahminleri'][2]:.2f} ₺")
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # 2. Kurumsal Grafikler
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_sonuc["Periyot"], y=df_sonuc["LSTM (Temkinli)"], mode='lines+markers', name='LSTM Modeli', line=dict(color='#0066cc', width=2)))
                    fig.add_trace(go.Scatter(x=df_sonuc["Periyot"], y=df_sonuc["GRU (Agresif)"], mode='lines+markers', name='GRU Modeli', line=dict(color='#cc0000', width=2, dash='dot')))
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="",
                        yaxis_title="Kapanış Fiyatı (TRY)",
                        template="plotly_white",  # Temiz beyaz tema
                        margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Detaylı Veri Tablosu
                    with st.expander("Detaylı Veri Tablosunu Görüntüle"):
                        st.dataframe(df_sonuc.style.format({"LSTM (Temkinli)": "{:.2f} ₺", "GRU (Agresif)": "{:.2f} ₺"}), use_container_width=True)
                        
                else:
                    st.error(f"Sistem Hatası: {cevap.json().get('detail', 'Bilinmeyen bir hata oluştu.')}")
            except Exception as e:
                st.error("API sunucusuna ulaşılamıyor. Lütfen FastAPI servisinin çalıştığından emin olun.")
    else:
        st.info("Analizi başlatmak için sol panelden parametreleri yapılandırın ve 'Analizi Başlat' butonuna tıklayın.")