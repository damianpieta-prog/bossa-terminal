import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# === KONFIGURACJA STRONY (Musi być na samej górze) ===
st.set_page_config(page_title="CENTRUM DOWODZENIA", layout="wide", page_icon="🧠")

# ==========================================
# 🎛️ MENU GŁÓWNE (PASEK BOCZNY)
# ==========================================
st.sidebar.title("🎛️ NAWIGACJA")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Wybierz aplikację:", 
    ["🚀 BOSSA Terminal", "⚖️ Kalkulator Ryzyka (R/R)", "👁️ Irydologia AI"]
)
st.sidebar.markdown("---")

# ==========================================
# APLIKACJA 1: BOSSA TERMINAL
# ==========================================
if app_mode == "🚀 BOSSA Terminal":
    
    # --- Konfiguracja i Funkcje BOSSA ---
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1zAE2mUbcVwBfI78f7v3_4K20Z5ffXymyrIcqcyadF4M/export?format=csv&gid=0"
    RSI_MOMENTUM = 65
    ATR_MULTIPLIER = 2.5
    SL_NORMAL_PCT = 0.015
    SL_TIGHT_PCT = 0.006

    @st.cache_data(ttl=900)
    def load_tickers():
        try:
            df = pd.read_csv(SHEET_URL)
            if df.empty: return []
            tickers = df.iloc[:, 0].dropna().astype(str).tolist()
            return [t.strip() for t in tickers if len(t) > 1]
        except: return []

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
        # Wskaźniki
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema17 = EMAIndicator(close, window=17).ema_indicator()
        ema100 = EMAIndicator(close, window=100).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        rsi = RSIIndicator(close, window=14).rsi()
        atr = AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
        
        # Do wykresu (Dodajemy wszystkie EMA do DataFrame)
        df['EMA_9'] = ema9
        df['EMA_17'] = ema17
        df['EMA_100'] = ema100  # Dodano EMA 100
        df['EMA_200'] = ema200

        # Regresja
        y = close.tail(50).values
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        lin_reg = np.poly1d(coef)(x)
        std_dev = np.std(y - lin_reg)
        
        # Logika
        current_price = close.iloc[-1]
        keltner_upper = EMAIndicator(close, window=20).ema_indicator().iloc[-1] + (atr.iloc[-1] * ATR_MULTIPLIER)
        
        signal = "WAIT"
        risk_note = "Neutral"
        sl_price = 0.0
        
        if current_price > ema200.iloc[-1] and ema100.iloc[-1] > ema200.iloc[-1] and rsi.iloc[-1] >= RSI_MOMENTUM:
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
        show_all = st.checkbox("Pokaż wszystkie", False)
        show_crosses = st.checkbox("Pokaż przecięcia EMA", True) # Opcja włączania krzyżyków

    tickers = load_tickers()
    if not tickers:
        st.error("Brak tickerów w arkuszu!")
        st.stop()

    results = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        progress.progress((i+1)/len(tickers))
        df = get_data(t)
        if df is not None:
            try:
                res = calculate_signals(df)
                res['Ticker'] = t
                results.append(res)
            except: pass
    progress.empty()
    
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
                    st.write(f"SL: **{row['SL']:.2f}**")
                    risk_val = capital * risk_pct
                    dist = row['Price'] - row['SL']
                    if dist > 0:
                        qty = risk_val / dist
                        st.info(f"Kup: **{int(qty)} szt.**\n(Ryzyko: {risk_val:.0f})")
                
                # WYKRES
                with c2:
                    df_chart = row['DataFrame'].tail(150) # Trochę dłuższy horyzont
                    fig = go.Figure()
                    
                    # Świece
                    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Cena'))
                    
                    # Średnie
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_9'], line=dict(color='blue', width=1), name='EMA 9'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_17'], line=dict(color='orange', width=1), name='EMA 17'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_100'], line=dict(color='purple', width=1.5, dash='dot'), name='EMA 100')) # Fioletowa
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_200'], line=dict(color='black', width=2), name='EMA 200'))

                    # Logika Przecięć (Crossovers)
                    if show_crosses:
                        # 1. Przecięcie 9 i 17
                        # Cross UP (9 przebija 17 w górę)
                        cross_9_17_up = df_chart[(df_chart['EMA_9'] > df_chart['EMA_17']) & (df_chart['EMA_9'].shift(1) < df_chart['EMA_17'].shift(1))]
                        fig.add_trace(go.Scatter(mode='markers', x=cross_9_17_up.index, y=cross_9_17_up['EMA_9'], marker=dict(color='green', symbol='triangle-up', size=8), name='Cross 9/17 UP', showlegend=False))
                        
                        # Cross DOWN (9 przebija 17 w dół)
                        cross_9_17_down = df_chart[(df_chart['EMA_9'] < df_chart['EMA_17']) & (df_chart['EMA_9'].shift(1) > df_chart['EMA_17'].shift(1))]
                        fig.add_trace(go.Scatter(mode='markers', x=cross_9_17_down.index, y=cross_9_17_down['EMA_9'], marker=dict(color='red', symbol='triangle-down', size=8), name='Cross 9/17 DOWN', showlegend=False))

                        # 2. Przecięcie 100 i 200 (GOLDEN CROSS / DEATH CROSS)
                        # Golden Cross (100 przebija 200 w górę)
                        cross_gold = df_chart[(df_chart['EMA_100'] > df_chart['EMA_200']) & (df_chart['EMA_100'].shift(1) < df_chart['EMA_200'].shift(1))]
                        fig.add_trace(go.Scatter(mode='markers', x=cross_gold.index, y=cross_gold['EMA_100'], marker=dict(color='gold', symbol='diamond', size=12, line=dict(width=2, color='black')), name='Golden Cross 100/200'))
                        
                        # Death Cross (100 przebija 200 w dół)
                        cross_death = df_chart[(df_chart['EMA_100'] < df_chart['EMA_200']) & (df_chart['EMA_100'].shift(1) > df_chart['EMA_200'].shift(1))]
                        fig.add_trace(go.Scatter(mode='markers', x=cross_death.index, y=cross_death['EMA_100'], marker=dict(color='black', symbol='diamond', size=12, line=dict(width=2, color='white')), name='Death Cross 100/200'))

                    # Stop Loss
                    fig.add_hline(y=row['SL'], line_dash="dash", line_color="red")
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, height=350, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1, x=0, bgcolor="rgba(255,255,255,0.5)"))
                    st.plotly_chart(fig, use_container_width=True)
                
                with c3:
                    st.write(f"**{row['Risk Note']}**")
                    diff = ((row['Price'] - row['Reg_Last'])/row['Reg_Last'])*100
                    st.metric("Odchylenie Trendu", f"{diff:.1f}%")
                    if row['Price'] > row['Reg_Upper']: st.error("Ekstremum (>2SD)")
    else:
        st.info("Brak sygnałów.")

# ==========================================
# APLIKACJA 2: KALKULATOR RYZYKA
# ==========================================
elif app_mode == "⚖️ Kalkulator Ryzyka (R/R)":
    st.title("⚖️ Profesjonalny Kalkulator Ryzyka")
    st.markdown("Szybkie obliczanie wielkości pozycji dla zagrań manualnych.")
    
    col1, col2 = st.columns(2)
    with col1:
        account_balance = st.number_input("Stan Konta", value=10000.0, step=100.0)
        risk_percent = st.number_input("Ryzyko na transakcję (%)", value=1.0, step=0.1)
    with col2:
        entry_price = st.number_input("Cena Wejścia", value=0.0, step=0.1)
        stop_loss = st.number_input("Cena Stop Loss", value=0.0, step=0.1)
    
    st.divider()
    
    if entry_price > 0 and stop_loss > 0:
        risk_amount = account_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        position_size = risk_amount / price_diff
        total_value = position_size * entry_price
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Ryzykowana Kwota", f"{risk_amount:.2f} PLN")
        c2.metric("Wielkość Pozycji", f"{position_size:.4f} szt.")
        c3.metric("Wartość Zlecenia", f"{total_value:.2f} PLN")
        
        st.success(f"💡 Wystaw zlecenie kupna na **{position_size:.4f}** akcji/jednostek.")
        
        # Wizualizacja R:R
        take_profit = entry_price + (price_diff * 3)
        st.write(f"Sugerowany Take Profit (R:R 1:3): **{take_profit:.2f}**")
    else:
        st.info("Wprowadź cenę wejścia i SL, aby obliczyć.")

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
                # TU BĘDZIE KOD GEMINI API W PRZYSZŁOŚCI
                import time
                time.sleep(2) # Symulacja
                st.success("Analiza zakończona pomyślnie!")
                st.markdown("### Wyniki Detekcji:")
                st.info("⚠️ Wykryto potencjalne osłabienie w sektorze: Wątroba (godz. 8:00)")
                st.write("Szczegóły: Widoczne przebarwienie typu 'plamka psora'.")
