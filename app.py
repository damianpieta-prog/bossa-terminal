import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai 
from datetime import datetime, timedelta

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(page_title="CENTRUM DOWODZENIA", layout="wide", page_icon="🧠")

# ==========================================
# 📥 FUNKCJE GLOBALNE
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zAE2mUbcVwBfI78f7v3_4K20Z5ffXymyrIcqcyadF4M/export?format=csv&gid=0"

@st.cache_data(ttl=900)
def load_tickers():
    try:
        df = pd.read_csv(SHEET_URL)
        if df.empty: return []
        tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        clean_tickers = sorted(list(set([t.strip() for t in tickers if len(t) > 1])))
        return clean_tickers
    except: return []

# Funkcja pobierająca dane (uniwersalna)
def get_data_universal(ticker, period="5y"):
    if ticker == "DAX": ticker = "^GDAXI"
    if ticker == "WIG20": ticker = "WIG20.WA"
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if len(df) < 100: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Usuwamy strefy czasowe dla bezpieczeństwa obliczeń
        df.index = df.index.tz_localize(None)
        return df
    except: return None

# ==========================================
# 🎛️ MENU GŁÓWNE
# ==========================================
st.sidebar.title("🎛️ NAWIGACJA")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Wybierz moduł:", 
    [
        "🔍 SZYBKI AUDYT (One-Pager)", 
        "🚀 BOSSA Terminal (Skaner)", 
        "📈 Analiza Trendu (Regresja)", 
        "🛡️ Kalkulator Bezpiecznego Inwestora", 
        "👁️ Irydologia AI"
    ]
)
st.sidebar.markdown("---")

# ==========================================
# MODUŁ 1: SZYBKI AUDYT (ONE-PAGER) - NOWOŚĆ
# ==========================================
if app_mode == "🔍 SZYBKI AUDYT (One-Pager)":
    st.title("🔍 SZYBKI AUDYT AKTYWA")
    st.markdown("Zintegrowany panel decyzyjny: BOSSA + Regresja + Bezpieczeństwo na jednym wykresie.")

    tickers = load_tickers()
    col_input, col_date = st.columns([2, 1])
    
    with col_input:
        selected_ticker = st.selectbox("Wybierz spółkę:", tickers)
    with col_date:
        default_start = datetime.now() - timedelta(days=180)
        start_date = st.date_input("Początek trendu:", value=default_start)

    if st.button("🚀 PRZEŚWIETL SPÓŁKĘ", type="primary"):
        with st.spinner(f"Analizuję {selected_ticker} pod każdym kątem..."):
            df = get_data_universal(selected_ticker, period="5y")
            
            if df is not None:
                # --- 1. OBLICZENIA BOSSA ---
                close = df['Close']
                current_price = close.iloc[-1]
                rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
                ema200 = EMAIndicator(close, window=200).ema_indicator().iloc[-1]
                ema100 = EMAIndicator(close, window=100).ema_indicator().iloc[-1]
                
                # Sygnał BOSSA
                bossa_signal = "NEUTRAL / WAIT"
                sl_price = 0.0
                if rsi >= 65 and current_price > ema200:
                    bossa_signal = "🟢 MOŻLIWY BUY (Momentum)"
                    sl_price = current_price * (1 - 0.015) # SL 1.5%

                # --- 2. OBLICZENIA REGRESJI (Logarytmiczna) ---
                start_ts = pd.to_datetime(start_date)
                df_reg = df[df.index >= start_ts].copy()
                
                reg_status = "Brak danych"
                trend_pct = 0.0
                
                if len(df_reg) > 5:
                    y = df_reg['Close'].values
                    x = np.arange(len(y))
                    y_log = np.log(y)
                    slope, intercept = np.polyfit(x, y_log, 1)
                    
                    # Linie trendu
                    trend_log = slope * x + intercept
                    std_dev = np.std(y_log - trend_log)
                    
                    trend_line = np.exp(trend_log)
                    upper_2sd = np.exp(trend_log + 2*std_dev)
                    lower_2sd = np.exp(trend_log - 2*std_dev)
                    upper_1sd = np.exp(trend_log + 1*std_dev)
                    lower_1sd = np.exp(trend_log - 1*std_dev)
                    
                    curr_trend = trend_line[-1]
                    trend_pct = ((current_price - curr_trend)/curr_trend)*100
                    
                    if current_price > upper_2sd[-1]: reg_status = "🚨 EKSTREMALNIE DROGO (>2SD)"
                    elif current_price > upper_1sd[-1]: reg_status = "🔥 DROGO (>1SD)"
                    elif current_price < lower_2sd[-1]: reg_status = "💎 SUPER OKAZJA (<2SD)"
                    elif current_price < lower_1sd[-1]: reg_status = "💎 TANIO (<1SD)"
                    else: reg_status = "⚖️ W NORMIE"

                # --- 3. OBLICZENIA SAFE INVESTOR ---
                wma_200_val = df['Close'].rolling(window=1000).mean().iloc[-1] # ok. 200 tyg
                if pd.isna(wma_200_val): wma_200_val = df['Close'].min()
                safe_dist = ((current_price - wma_200_val)/wma_200_val)*100
                safe_txt = "BEZPIECZNIE" if safe_dist < 15 else "NEUTRALNIE"

                # ==========================================
                # DASHBOARD (WIZUALIZACJA)
                # ==========================================
                st.divider()
                
                # METRYKI
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cena", f"{current_price:.2f}")
                m2.metric("Sygnał BOSSA", bossa_signal, delta=f"RSI: {rsi:.1f}")
                m3.metric("Status Trendu", reg_status, delta=f"{trend_pct:.1f}% od środka", delta_color="inverse")
                m4.metric("Długi Termin", safe_txt, delta=f"{safe_dist:.1f}% od dna", delta_color="inverse")

                # WYKRES HYBRYDOWY
                st.subheader(f"📊 Mapa Taktyczna: {selected_ticker}")
                
                fig = go.Figure()

                # Tło - Kanały Regresji
                fig.add_trace(go.Scatter(x=df_reg.index, y=upper_2sd, mode='lines', name='+2 SD (Opór)', line=dict(color='red', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=df_reg.index, y=lower_2sd, mode='lines', name='-2 SD (Wsparcie)', line=dict(color='green', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=df_reg.index, y=upper_1sd, mode='lines', name='+1 SD', line=dict(color='orange', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df_reg.index, y=lower_1sd, mode='lines', name='-1 SD', line=dict(color='lightgreen', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df_reg.index, y=trend_line, mode='lines', name='TREND (Środek)', line=dict(color='blue', width=2)))

                # Cena
                fig.add_trace(go.Scatter(x=df_reg.index, y=df_reg['Close'], mode='lines', name='CENA', line=dict(color='black', width=3)))

                # Stop Loss (jeśli jest sygnał)
                if "BUY" in bossa_signal:
                    fig.add_hline(y=sl_price, line_dash="solid", line_color="red", annotation_text=f"STOP LOSS: {sl_price:.2f}", annotation_position="bottom right")

                fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Błąd pobierania danych.")

# ==========================================
# MODUŁ 2: BOSSA TERMINAL (SKANER)
# ==========================================
elif app_mode == "🚀 BOSSA Terminal (Skaner)":
    st.title("🚀 BOSSA TERMINAL")
    st.write("Skaner całego rynku w poszukiwaniu sygnałów.")
    
    RSI_MOMENTUM = 65
    ATR_MULTIPLIER = 2.5
    SL_NORMAL_PCT = 0.015
    SL_TIGHT_PCT = 0.006

    def calculate_bossa(df):
        close = df['Close']
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema17 = EMAIndicator(close, window=17).ema_indicator()
        ema100 = EMAIndicator(close, window=100).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        rsi = RSIIndicator(close, window=14).rsi()
        
        current_price = close.iloc[-1]
        
        signal = "WAIT"
        sl_price = 0.0
        
        # Prosta logika
        if rsi.iloc[-1] >= RSI_MOMENTUM and current_price > ema200.iloc[-1]:
            signal = "🟢 BUY"
            sl_price = current_price * (1 - SL_NORMAL_PCT)

        return {
            "Price": current_price, "RSI": rsi.iloc[-1], "Signal": signal, "SL": sl_price,
            "DataFrame": df, "EMA9": ema9, "EMA17": ema17, "EMA100": ema100, "EMA200": ema200
        }

    tickers = load_tickers()
    if st.button("🚀 Skanuj Rynek"):
        results = []
        prog = st.progress(0)
        
        for i, t in enumerate(tickers):
            prog.progress((i+1)/len(tickers))
            df = get_data_universal(t, period="2y")
            if df is not None:
                try:
                    res = calculate_bossa(df)
                    res['Ticker'] = t
                    if "BUY" in res['Signal']: results.append(res)
                except: pass
        prog.empty()
        
        if results:
            st.success(f"Znaleziono {len(results)} okazji.")
            for row in results:
                # WYSWIETLANIE BEZ ROZWIJANIA (OD RAZU WIDOCZNE)
                st.container()
                st.markdown(f"### {row['Ticker']} | Cena: {row['Price']:.2f}")
                
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.metric("Sygnał", row['Signal'])
                    st.metric("Stop Loss", f"{row['SL']:.2f}")
                    st.metric("RSI", f"{row['RSI']:.1f}")
                
                with c2:
                    df_chart = row['DataFrame'].tail(150)
                    fig = go.Figure()
                    # LINIOWY
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], mode='lines', line=dict(color='black', width=2), name='Cena'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=row['EMA9'].tail(150), line=dict(color='blue', width=1), name='EMA 9'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=row['EMA200'].tail(150), line=dict(color='gray', width=2), name='EMA 200'))
                    
                    fig.add_hline(y=row['SL'], line_dash="dash", line_color="red")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"bossa_{row['Ticker']}")
                st.divider()
        else:
            st.warning("Brak sygnałów kupna.")

# ==========================================
# MODUŁ 3: REGRESJA (ANALIZA TRENDU)
# ==========================================
elif app_mode == "📈 Analiza Trendu (Regresja)":
    st.title("📈 Analiza Trendu (Kanały 1SD - 3SD)")
    
    with st.sidebar:
        default_start = datetime.now() - timedelta(days=180)
        start_date = st.date_input("Początek trendu:", value=default_start)

    tickers = load_tickers()
    if st.button("🚀 Oblicz Kanały Regresji"):
        results_reg = []
        prog = st.progress(0)
        start_ts = pd.to_datetime(start_date)

        for i, t in enumerate(tickers):
            prog.progress((i+1)/len(tickers))
            df = get_data_universal(t, period="5y")
            if df is not None:
                try:
                    df_reg = df[df.index >= start_ts].copy()
                    if len(df_reg) > 5:
                        y = df_reg['Close'].values
                        x = np.arange(len(y))
                        y_log = np.log(y)
                        slope, intercept = np.polyfit(x, y_log, 1)
                        trend_log = slope * x + intercept
                        std_dev = np.std(y_log - trend_log)
                        
                        trend_line = np.exp(trend_log)
                        upper_2sd = np.exp(trend_log + 2*std_dev)
                        lower_2sd = np.exp(trend_log - 2*std_dev)
                        upper_1sd = np.exp(trend_log + 1*std_dev)
                        lower_1sd = np.exp(trend_log - 1*std_dev)
                        upper_3sd = np.exp(trend_log + 3*std_dev)
                        
                        curr = y[-1]
                        dist = ((curr - trend_line[-1])/trend_line[-1])*100
                        
                        extreme_note = ""
                        if curr > upper_2sd[-1]:
                            dist3 = ((upper_3sd[-1] - curr)/curr)*100
                            extreme_note = f"⚠️ UWAGA: Przebito 2SD! Do 3SD zostało {dist3:.1f}%"

                        results_reg.append({
                            "Ticker": t, "DistPct": dist, "Data": df_reg, "Trend": trend_line,
                            "U2": upper_2sd, "L2": lower_2sd, "U1": upper_1sd, "L1": lower_1sd,
                            "Note": extreme_note
                        })
                except: pass
        prog.empty()
        
        results_reg.sort(key=lambda x: x['DistPct'], reverse=True)
        
        for res in results_reg:
            st.container()
            header = f"{res['Ticker']} | Odchylenie: {res['DistPct']:.1f}%"
            if res['Note']: header += f" | {res['Note']}"
            
            st.markdown(f"#### {header}")
            
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Cena", f"{res['Data']['Close'].iloc[-1]:.2f}")
                st.metric("Odchylenie", f"{res['DistPct']:.1f}%")
                if res['Note']: st.error(res['Note'])
            
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['U2'], mode='lines', line=dict(color='red', width=1, dash='dash'), name='+2SD'))
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['L2'], mode='lines', line=dict(color='green', width=1, dash='dash'), name='-2SD'))
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['U1'], mode='lines', line=dict(color='orange', width=1, dash='dot'), name='+1SD'))
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['L1'], mode='lines', line=dict(color='lightgreen', width=1, dash='dot'), name='-1SD'))
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Trend'], mode='lines', line=dict(color='blue', width=2), name='Trend'))
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['Close'], mode='lines', line=dict(color='black', width=2), name='Cena'))
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"reg_{res['Ticker']}")
            st.divider()

# ==========================================
# MODUŁ 4: SAFE INVESTOR
# ==========================================
elif app_mode == "🛡️ Kalkulator Bezpiecznego Inwestora":
    st.title("🛡️ Kalkulator Bezpiecznego Inwestora")
    tickers = load_tickers()
    
    if st.button("🚀 Skanuj Rynek"):
        results = []
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            prog.progress((i+1)/len(tickers))
            df = get_data_universal(t, period="5y")
            if df is not None:
                try:
                    curr = df['Close'].iloc[-1]
                    wma200 = df['Close'].rolling(1000).mean().iloc[-1]
                    if pd.isna(wma200): wma200 = df['Close'].min()
                    
                    downside = curr - wma200
                    score = 0
                    if downside <= 0: score = 100 + abs(downside) # Okazja życia
                    else: score = (curr*1.1 - curr) / downside # RR
                    
                    results.append({"Ticker": t, "Price": curr, "Floor": wma200, "Score": score})
                except: pass
        prog.empty()
        
        results.sort(key=lambda x: x['Score'], reverse=True)
        
        for res in results:
            col = "green" if res['Price'] < res['Floor'] else "orange"
            st.markdown(f"### {res['Ticker']}")
            c1, c2 = st.columns([1, 2])
            c1.metric("Cena", f"{res['Price']:.2f}")
            c1.metric("Bezpieczne Dno", f"{res['Floor']:.2f}", delta=f"{res['Price']-res['Floor']:.2f}", delta_color="inverse")
            
            with c2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = res['Score'],
                    title = {'text': "Atrakcyjność"},
                    gauge = {'axis': {'range': [0, 5]}, 'bar': {'color': "black"},
                             'steps': [{'range': [0, 1], 'color': "#ff4b4b"}, {'range': [1, 5], 'color': "#21c354"}]}
                ))
                fig.update_layout(height=150, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig, use_container_width=True, key=f"safe_{res['Ticker']}")
            st.divider()

# ==========================================
# MODUŁ 5: IRYDOLOGIA AI
# ==========================================
elif app_mode == "👁️ Irydologia AI":
    st.title("👁️ Irydologia AI (System Wzorców Własnych)")
    
    # === TWOJE DANE API ===
    api_key = "AIzaSyB3CYXGVWsouSHuQRo8TF7mh_uT8BuHoQU"
    
    REFERENCE_FILES = [
        "konstytucja.jpeg",
        "teczowka.jpeg", 
        "twardowka.jpeg",
        "kryza.jpeg",
        "mapa_irydologiczna.jpg" 
    ]

    uploaded_file = st.file_uploader("Wgraj zdjęcie oka pacjenta...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        patient_img = Image.open(uploaded_file)
        c1, c2 = st.columns(2)
        with c1: st.image(patient_img, caption='Oko Pacjenta', use_column_width=True)
        with c2: st.info(f"System użyje {len(REFERENCE_FILES)} Twoich wzorców do analizy.")
        
        if st.button("🔍 URUCHOM ANALIZĘ"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            with st.spinner('AI studiuje Twoje mapy i analizuje pacjenta...'):
                try:
                    prompt_parts = []
                    prompt_parts.append("""
                    Jesteś ekspertem irydologii. Analizuj oko pacjenta PORÓWNUJĄC z WZORCAMI.
                    Użyj 'mapa_irydologiczna' do lokalizacji organów.
                    Zidentyfikuj znaki (zatoki, psora) i postaw diagnozę w punktach.
                    MATERIAŁY REFERENCYJNE:
                    """)
                    for filename in REFERENCE_FILES:
                        try:
                            img = Image.open(filename)
                            prompt_parts.append(f"WZORZEC/MAPA: {filename}")
                            prompt_parts.append(img)
                        except: pass
                    prompt_parts.append("A TERAZ PRZEANALIZUJ TO ZDJĘCIE PACJENTA:")
                    prompt_parts.append(patient_img)
                    
                    response = model.generate_content(prompt_parts)
                    st.success("Analiza zakończona!")
                    st.markdown("### 📋 Raport Irydologiczny")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")
