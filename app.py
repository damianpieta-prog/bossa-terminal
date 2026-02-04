import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# === KONFIGURACJA STRONY ===
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
    
    # --- Konfiguracja i Funkcje ---
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
        
        # Do wykresu
        df['EMA_9'] = ema9
        df['EMA_17'] = ema17
        df['EMA_100'] = ema100
        df['EMA_200'] = ema200

        # Regresja (Trend Liniowy)
        y = close.tail(50).values
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        lin_reg = np.poly1d(coef)(x)
        std_dev = np.std(y - lin_reg)
        
        # Logika Sygnałów
        current_price = close.iloc[-1]
        keltner_upper = EMAIndicator(close, window=20).ema_indicator().iloc[-1] + (atr.iloc[-1] * ATR_MULTIPLIER)
        
        signal = "WAIT"
        risk_note = "Neutral"
        sl_price = 0.0
        
        # Warunki wejścia
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
        st.header("Ustawienia")
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
    
    # Filtrowanie
    if not show_all:
        final_df = res_df[res_df['Signal'].str.contains("BUY")]
    else:
        final_df = res_df

    # Wyświetlanie
    if not final_df.empty:
        for idx, row in final_df.iterrows():
            with st.expander(f"{row['Ticker']}
