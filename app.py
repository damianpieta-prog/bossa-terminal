import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ==========================================
# KONFIGURACJA STRONY (Globalna)
# ==========================================
st.set_page_config(page_title="CENTRUM DOWODZENIA", layout="wide", page_icon="🧠")

# ==========================================
# 📥 FUNKCJE GLOBALNE (Dostępne dla każdej aplikacji)
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zAE2mUbcVwBfI78f7v3_4K20Z5ffXymyrIcqcyadF4M/export?format=csv&gid=0"

@st.cache_data(ttl=900)
def load_tickers():
    """Pobiera listę tickerów z Google Sheet"""
    try:
        df = pd.read_csv(SHEET_URL)
        if df.empty: return []
        tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        # Czyścimy i usuwamy duplikaty
        clean_tickers = sorted(list(set([t.strip() for t in tickers if len(t) > 1])))
        return clean_tickers
    except: return []

# ==========================================
# 🎛️ MENU GŁÓWNE (PASEK BOCZNY)
# ==========================================
st.sidebar.title("🎛️ NAWIGACJA")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Wybierz aplikację:", 
    ["🚀 BOSSA Terminal", "🛡️ Kalkulator Bezpiecznego Inwestora", "👁️ Irydologia AI"]
)
st.sidebar.markdown("---")

# ==========================================
# APLIKACJA 1: BOSSA TERMINAL (Daytrading)
# ==========================================
if app_mode == "🚀 BOSSA Terminal":
    
    # --- Konfiguracja lokalna ---
    RSI_MOMENTUM = 65
    ATR_MULTIPLIER = 2.5
    SL_NORMAL_PCT = 0.015
    SL_TIGHT_PCT = 0.006

    def get_data(ticker):
        if ticker == "DAX": ticker = "^GDAXI"
        if ticker == "WIG20": ticker = "WIG20.WA"
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if len(df) < 200: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except: return None

    def calculate_signals(df):
        close = df['Close']
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema17 = EMAIndicator(close, window=17).ema_indicator()
        ema100 = EMAIndicator(close, window=100).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        rsi = RSIIndicator(close, window=14).rsi()
        atr = AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
        
        df['EMA_9'] = ema9
        df['EMA_17'] = ema17
        df['EMA_100'] = ema100
        df['EMA_200'] = ema200

        y = close.tail(50).values
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        lin_reg = np.poly1d(coef)(x)
        std_dev = np.std(y - lin_reg)
        
        current_price = close.iloc[-1]
        keltner_upper = EMAIndicator(close, window=20).ema_indicator().iloc[-1] + (atr.iloc[-1] * ATR_MULTIPLIER)
        
        signal = "WAIT"
        risk_note = "Neutral"
        sl_price = 0.0
        
        is_trend = current_price > ema200.iloc[-1] and ema100.iloc[-1] > ema200.iloc[-1]
        is_momentum = rsi.iloc[-1] >= RSI_MOMENTUM
        
        if is_trend and is_momentum:
            if current_price > keltner_upper:
                signal = "⚠️ BUY (HIGH RISK)"
                risk_note = "Cena > ATR"
                sl_price = current_price * (1 - SL_TIGHT_PCT)
            else:
                signal = "🟢 BUY (MOMENTUM)"
                space = ((keltner_upper - current_price) / current_price) * 100
                risk_note = f"Zapas ATR: {space:.1f}%"
                sl_price = current_price * (1 - SL_NORMAL_PCT)

        return {
            "Price": current_price, "RSI": rsi.iloc[-1], "Signal": signal,
            "Risk Note": risk_note, "SL": sl_price,
            "DataFrame": df, "Reg_Last": lin_reg[-1], "Reg_Upper": lin_reg[-1] + (2*std_dev)
        }

    # --- Interfejs BOSSA ---
    st.title("🚀 BOSSA 3.3 TERMINAL")
    
    with st.sidebar:
        st.header("Ustawienia Terminala")
        capital = st.number_input("Kapitał (PLN/USD)", 10000, step=1000)
        risk_pct = st.slider("Ryzyko (%)", 0.5, 5.0, 1.0) / 100
        show_all = st.checkbox("Pokaż wszystkie (nawet WAIT)", False)
        show_crosses = st.checkbox("Pokaż przecięcia EMA", True)

    tickers = load_tickers()
    if not tickers:
        st.error("Brak tickerów. Sprawdź link do arkusza.")
        st.stop()

    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(tickers):
        status.text(f"Pobieram: {t}...")
        progress.progress((i+1)/len(tickers))
        df = get_data(t)
        if df is not None:
            try:
                res = calculate_signals(df)
                res['Ticker'] = t
                results.append(res)
            except: pass
    
    progress.empty()
    status.empty()
    
    res_df = pd.DataFrame(results)
    
    if not show_all:
        final_df = res_df[res_df['Signal'].str.contains("BUY")]
    else:
        final_df = res_df

    if not final_df.empty:
        for idx, row in final_df.iterrows():
            with st.expander(f"{row['Ticker']} | {row['Signal']}", expanded=True):
                c1, c2, c3 = st.columns([1,2,1])
                with c1:
                    st.metric("Cena", f"{row['Price']:.2f}")
                    st.write(f"RSI: **{row['RSI']:.1f}**")
                    if "BUY" in row['Signal']:
                        st.write(f"Stop Loss: **{row['SL']:.2f}**")
                        risk_amount = capital * risk_pct
                        dist = row['Price'] - row['SL']
                        if dist > 0:
                            qty = risk_amount / dist
                            st.info(f"Kup: **{int(qty)} szt.**\n(Ryzyko: {risk_amount:.0f})")
                    else:
                        st.caption("Brak sygnału - Kalkulator ukryty.")

                with c2:
                    df_chart = row['DataFrame'].tail(150)
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Cena'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_9'], line=dict(color='blue', width=1), name='EMA 9'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_17'], line=dict(color='orange', width=1), name='EMA 17'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_100'], line=dict(color='purple', width=1.5, dash='dot'), name='EMA 100'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_200'], line=dict(color='black', width=2), name='EMA 200'))
                    
                    if show_crosses:
                        cross_gold = df_chart[(df_chart['EMA_100'] > df_chart['EMA_200']) & (df_chart['EMA_100'].shift(1) < df_chart['EMA_200'].shift(1))]
                        fig.add_trace(go.Scatter(mode='markers', x=cross_gold.index, y=cross_gold['EMA_100'], marker=dict(color='gold', symbol='diamond', size=12, line=dict(width=2, color='black')), name='Golden Cross'))

                    if "BUY" in row['Signal']:
                        fig.add_hline(y=row['SL'], line_dash="dash", line_color="red")
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with c3:
                    st.subheader("Ocena Ryzyka")
                    st.write(f"Sytuacja: {row['Risk Note']}")
                    diff = ((row['Price'] - row['Reg_Last'])/row['Reg_Last'])*100
                    if diff > 0: st.write(f"📈 Odchylenie: **+{diff:.1f}%**")
                    else: st.write(f"📉 Odchylenie: **{diff:.1f}%**")
                    if row['Price'] > row['Reg_Upper']: st.error("🚨 EKSTREMALNIE DROGO (>2SD)")
                    elif diff > 5: st.warning("⚠️ Dość drogo")
                    else: st.success("✅ W normie")
    else:
        st.info("Brak sygnałów kupna w Twoim portfelu.")

# ==========================================
# APLIKACJA 2: KALKULATOR BEZPIECZNEGO INWESTORA (Wersja 2.0 - Arkusz)
# ==========================================
elif app_mode == "🛡️ Kalkulator Bezpiecznego Inwestora":
    st.title("🛡️ Kalkulator Bezpiecznego Inwestora")
    st.write("Strategia: Kupuj, gdy inni się boją (poniżej średniej 200-tygodniowej).")

    # Pobierz tickery z arkusza
    sheet_tickers = load_tickers()
    
    # Wybór trybu pracy
    mode = st.radio("Tryb analizy:", ["🔍 Pojedyncza Spółka", "📋 Skanuj Cały Portfel (Raport)"], horizontal=True)

    @st.cache_data(ttl=600)
    def pobierz_dane_safe(symbol_aktywa):
        # Mapowanie dla złota/ropy jeśli ktoś wpisze dziwnie
        if symbol_aktywa == "GOLD": symbol_aktywa = "GLD"
        
        ticker = yf.Ticker(symbol_aktywa)
        df = ticker.history(period="5y", interval="1wk")
        return df

    # --- FUNKCJA RYSUJĄCA KAFE ---
    def draw_safe_analysis(symbol, data):
        if data.empty:
            st.error(f"Brak danych dla {symbol}")
            return False

        current_price = data['Close'].iloc[-1]
        wma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(wma_200): wma_200 = data['Close'].min()

        risk_floor = wma_200
        ath = data['High'].max()
        reward_ceiling = max(ath, current_price * 1.1)
        
        upside = reward_ceiling - current_price
        downside = current_price - risk_floor
        
        # Logika werdyktu
        if downside <= 0:
            rr_ratio = 5.0 # Max scale
            verdict = "OKAZJA ŻYCIA"
            color = "#21c354"
        else:
            rr_ratio = upside / downside
            if rr_ratio > 3:
                verdict = "OKAZJA (KUPUJ)"
                color = "#21c354"
            elif rr_ratio > 1:
                verdict = "NEUTRALNIE"
                color = "#ffa421"
            else:
                verdict = "NIEOPŁACALNE"
                color = "#ff4b4b"
        
        # Wyświetlanie
        with st.container():
            st.markdown(f"### {symbol}")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Cena", f"{current_price:,.2f}")
                st.markdown(f"Werdykt: **<span style='color:{color}'>{verdict}</span>**", unsafe_allow_html=True)
                st.metric("Bezpieczne Dno", f"{risk_floor:,.2f}", delta=f"-{downside:,.2f}", delta_color="inverse")
            with c2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = rr_ratio,
                    title = {'text': "Zysk/Ryzyko"},
                    gauge = {
                        'axis': {'range': [0, 5]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 1], 'color': "#ff4b4b"},
                            {'range': [1, 3], 'color': "#ffa421"},
                            {'range': [3, 5], 'color': "#21c354"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': rr_ratio}
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
        return True

    # --- TRYB 1: POJEDYNCZY ---
    if mode == "🔍 Pojedyncza Spółka":
        # Selectbox z tickerami z arkusza + opcja wpisania
        if sheet_tickers:
            symbol = st.selectbox("Wybierz walor z listy:", sheet_tickers)
        else:
            symbol = st.text_input("Wpisz symbol (np. BTC-USD):", value="BTC-USD")

        if symbol:
            with st.spinner(f"Analizuję {symbol}..."):
                data = pobierz_dane_safe(symbol)
                draw_safe_analysis(symbol, data)

    # --- TRYB 2: SKANOWANIE CAŁOŚCI ---
    elif mode == "📋 Skanuj Cały Portfel (Raport)":
        if st.button("🚀 Uruchom Pełny Skan (Może chwilę potrwać)"):
            if not sheet_tickers:
                st.error("Brak tickerów w arkuszu.")
            else:
                progress = st.progress(0)
                for i, t in enumerate(sheet_tickers):
                    progress.progress((i+1)/len(sheet_tickers))
                    # Pobieramy dane
                    try:
                        data = pobierz_dane_safe(t)
                        draw_safe_analysis(t, data)
                    except:
                        st.warning(f"Błąd danych dla {t}")
                progress.empty()
                st.success("Skanowanie zakończone!")

# ==========================================
# APLIKACJA 3: IRYDOLOGIA
# ==========================================
elif app_mode == "👁️ Irydologia AI":
    st.title("👁️ Irydologia - Analiza Tęczówki")
    st.write("Wgraj zdjęcie oka, aby dokonać analizy zdrowia.")
    
    uploaded_file = st.file_uploader("Wybierz zdjęcie oka...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Analizowane zdjęcie', use_column_width=True)
        
        if st.button("🔍 Rozpocznij Analizę AI"):
            with st.spinner('Analizuję strukturę tęczówki...'):
                import time
                time.sleep(2) 
                st.success("Analiza zakończona (Wersja Demo)")
                st.info("Tutaj w przyszłości pojawi się wynik z Gemini AI.")
