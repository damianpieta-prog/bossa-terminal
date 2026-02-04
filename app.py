import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# === KONFIGURACJA ===
st.set_page_config(page_title="BOSSA TERMINAL", layout="wide", page_icon="🚀")

# Link do Twojego arkusza (CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zAE2mUbcVwBfI78f7v3_4K20Z5ffXymyrIcqcyadF4M/export?format=csv&gid=0"

# Strategia
RSI_MOMENTUM = 65
ATR_MULTIPLIER = 2.5
SL_NORMAL_PCT = 0.015
SL_TIGHT_PCT = 0.006

# === FUNKCJE ===

@st.cache_data(ttl=900) # Cache na 15 min
def load_tickers_from_sheet():
    try:
        df = pd.read_csv(SHEET_URL)
        # Zakładamy, że tickery są w pierwszej kolumnie
        if df.empty: return []
        tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        tickers = [t.strip() for t in tickers if len(t) > 1]
        return tickers
    except Exception as e:
        st.error(f"Błąd arkusza: {e}")
        return []

def get_data(ticker):
    # Mapowanie nazw
    if ticker == "DAX": ticker = "^GDAXI"
    if ticker == "WIG20": ticker = "WIG20.WA"
    
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(df) < 200: return None
        # Naprawa MultiIndex w nowych wersjach yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def calculate_signals(df):
    close = df['Close']
    
    # 1. Obliczamy wskaźniki
    ema9_series = EMAIndicator(close, window=9).ema_indicator()
    ema17_series = EMAIndicator(close, window=17).ema_indicator()
    ema100_series = EMAIndicator(close, window=100).ema_indicator()
    ema200_series = EMAIndicator(close, window=200).ema_indicator()
    rsi_series = RSIIndicator(close, window=14).rsi()
    atr_series = AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
    
    # 2. Dodajemy je do DataFrame (ŻEBY BYŁY NA WYKRESIE)
    df['EMA_9'] = ema9_series
    df['EMA_17'] = ema17_series
    df['EMA_100'] = ema100_series
    df['EMA_200'] = ema200_series
    
    # 3. Regresja Liniowa (Ostatnie 50 dni)
    y = close.tail(50).values
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    lin_reg = np.poly1d(coef)(x) # Linia środkowa
    std_dev = np.std(y - lin_reg)
    
    current_price = close.iloc[-1]
    current_rsi = rsi_series.iloc[-1]
    current_atr = atr_series.iloc[-1]
    
    # Kanał Keltnera (ATR)
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    keltner_upper = ema20.iloc[-1] + (current_atr * ATR_MULTIPLIER)
    
    # Logika Sygnałów
    trend_up = current_price > ema200_series.iloc[-1] and ema100_series.iloc[-1] > ema200_series.iloc[-1]
    momentum = current_rsi >= RSI_MOMENTUM
    
    signal = "WAIT"
    risk_note = "Neutral"
    sl_price = 0.0
    
    if trend_up and momentum:
        if current_price > keltner_upper:
            signal = "⚠️ BUY (HIGH RISK)"
            risk_note = f"Cena wystrzelona > ATR"
            sl_price = current_price * (1 - SL_TIGHT_PCT)
        else:
            signal = "🟢 BUY (MOMENTUM)"
            space_left = ((keltner_upper - current_price) / current_price) * 100
            risk_note = f"Bezpiecznie. Zapas ATR: {space_left:.1f}%"
            sl_price = current_price * (1 - SL_NORMAL_PCT)
            
    return {
        "Price": current_price,
        "RSI": current_rsi,
        "Signal": signal,
        "Risk Note": risk_note,
        "SL": sl_price,
        "Regression_Last": lin_reg[-1], # Punkt na końcu linii trendu
        "Reg_Upper": lin_reg[-1] + (2 * std_dev),
        "DataFrame": df # Cała tabela z EMA do wykresu
    }

# === INTERFEJS ===

with st.sidebar:
    st.header("⚙️ Panel Tradera")
    capital = st.number_input("Kapitał (PLN/USD)", value=10000, step=1000)
    risk_per_trade = st.slider("Ryzyko (%)", 0.5, 5.0, 1.0) / 100
    st.divider()
    show_all = st.checkbox("Pokaż wszystkie (nawet WAIT)", value=False)
    st.caption("v. 1.1 (Fixed Charts & Labels)")

st.title("🚀 BOSSA 3.3 TERMINAL")

tickers = load_tickers_from_sheet()
if not tickers:
    st.warning("Brak tickerów. Sprawdź link do arkusza.")
    st.stop()

# Pasek postępu
results = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, ticker in enumerate(tickers):
    status_text.text(f"Pobieram: {ticker}...")
    progress_bar.progress((i + 1) / len(tickers))
    
    df = get_data(ticker)
    if df is not None:
        try:
            res = calculate_signals(df)
            res['Ticker'] = ticker
            results.append(res)
        except: pass

progress_bar.empty()
status_text.empty()

res_df = pd.DataFrame(results)

# Filtrowanie
if not show_all:
    final_df = res_df[res_df['Signal'].str.contains("BUY")]
else:
    final_df = res_df

# Metryki
if not res_df.empty:
    col1, col2, col3 = st.columns(3)
    buy_signals = len(res_df[res_df['Signal'].str.contains("BUY")])
    high_risk = len(res_df[res_df['Signal'].str.contains("HIGH RISK")])
    col1.metric("Znalezione Okazje", buy_signals)
    col2.metric("Wysokie Ryzyko", high_risk, delta_color="inverse")
    col3.metric("Sprawdzone Spółki", len(res_df))

st.divider()

# === WYŚWIETLANIE KART ===
if not final_df.empty:
    for index, row in final_df.iterrows():
        # Kolor belki
        label_color = "red" if "HIGH" in row['Signal'] else "green"
        emoji = "🔥" if "HIGH" in row['Signal'] else "🚀"
        
        with st.expander(f"{emoji} {row['Ticker']} | {row['Signal']}", expanded=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            
            # KOLUMNA 1: KALKULATOR
            with c1:
                st.metric("Cena", f"{row['Price']:.2f}")
                st.write(f"RSI: **{row['RSI']:.1f}**")
                st.write(f"Stop Loss: **{row['SL']:.2f}**")
                
                risk_amount = capital * risk_per_trade
                dist = row['Price'] - row['SL']
                if dist > 0:
                    qty = risk_amount / dist
                    st.info(f"Kup: **{int(qty)} szt.**\n(Ryzyko: {risk_amount:.0f})")

            # KOLUMNA 2: WYKRES Z ŚREDNIMI
            with c2:
                df_chart = row['DataFrame'].tail(120) # Pokaż ostatnie pół roku
                fig = go.Figure()
                
                # Świece
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'], high=df_chart['High'],
                                low=df_chart['Low'], close=df_chart['Close'], 
                                name='Cena'))
                
                # ŚREDNIE KROCZĄCE (To naprawiłem)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_9'], 
                                         line=dict(color='blue', width=1), name='EMA 9'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_17'], 
                                         line=dict(color='orange', width=1), name='EMA 17'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_200'], 
                                         line=dict(color='black', width=2), name='EMA 200 (Trend)'))

                # Linia SL
                fig.add_hline(y=row['SL'], line_dash="dash", line_color="red", annotation_text="SL")

                fig.update_layout(xaxis_rangeslider_visible=False, height=350, 
                                  margin=dict(l=0, r=0, t=20, b=0), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            # KOLUMNA 3: OPIS RYZYKA (REGRESJA)
            with c3:
                st.subheader("Ocena Ryzyka")
                st.write(f"Sytuacja: {row['Risk Note']}")
                
                # Obliczamy odchylenie od trendu (regresji)
                trend_price = row['Regression_Last']
                diff_pct = ((row['Price'] - trend_price) / trend_price) * 100
                
                # Czytelny komunikat
                if diff_pct > 0:
                    st.write(f"📈 Odchylenie: **+{diff_pct:.1f}%** (powyżej trendu)")
                else:
                    st.write(f"📉 Odchylenie: **{diff_pct:.1f}%** (poniżej trendu)")
                
                # Ocena "na chłopski rozum"
                if row['Price'] > row['Reg_Upper']:
                    st.error("🚨 EKSTREMALNIE DROGO (Ponad 2SD)")
                elif diff_pct > 5:
                    st.warning("⚠️ Dość drogo (Naciągnięte)")
                else:
                    st.success("✅ Cena w normie statystycznej")

else:
    st.info("Brak sygnałów kupna. Odpocznij :)")
