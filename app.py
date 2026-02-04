import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime

# === KONFIGURACJA ===
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zAE2mUbcVwBfI78f7v3_4K20Z5ffXymyrIcqcyadF4M/export?format=csv&gid=0"

# Strategia
RSI_MOMENTUM = 65
ATR_MULTIPLIER = 2.5
SL_NORMAL_PCT = 0.015
SL_TIGHT_PCT = 0.006

# === FUNKCJE ===

@st.cache_data(ttl=3600) # Cache na 1h żeby nie męczyć arkusza
def load_tickers_from_sheet():
    try:
        df = pd.read_csv(SHEET_URL)
        # Zakładamy, że tickery są w pierwszej kolumnie
        tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        # Czyszczenie tickerów (usuwanie spacji, pustych)
        tickers = [t.strip() for t in tickers if len(t) > 1]
        return tickers
    except Exception as e:
        st.error(f"Błąd pobierania z Arkusza: {e}")
        return []

def get_data(ticker):
    # Mapowanie dla yfinance (naprawa tickerów)
    if ticker == "DAX": ticker = "^GDAXI"
    if ticker == "WIG20": ticker = "WIG20.WA"
    # Jeśli to polska spółka bez końcówki, dodaj .WA (chyba że to krypto/USA)
    if ticker.isalpha() and len(ticker) == 3 and ticker not in ["IBM", "MSFT", "CAT", "OXY", "GLD", "SLV", "GDX", "URA"]:
         # Prosta heurystyka - zazwyczaj 3 litery to USA, ale KGH/PKN to PL. 
         # Lepiej w arkuszu trzymać pełne nazwy (KGH.WA). Tutaj zostawiamy jak jest.
         pass
         
    try:
        # Pobieramy dane
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(df) < 200: return None
        
        # Spłaszczenie MultiIndex (problem nowych wersji yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except:
        return None

def calculate_signals(df):
    close = df['Close']
    
    # Wskaźniki
    ema9 = EMAIndicator(close, window=9).ema_indicator() # Tu używamy D1 dla uproszczenia, w wersji PRO pobierzemy H4
    ema100 = EMAIndicator(close, window=100).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    rsi = RSIIndicator(close, window=14).rsi()
    atr = AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
    
    # Regresja Liniowa (Ostatnie 50 dni)
    y = close.tail(50).values
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    lin_reg = poly1d_fn(x)
    
    # Odchylenie od regresji
    std_dev = np.std(y - lin_reg)
    upper_channel = lin_reg + (2 * std_dev)
    lower_channel = lin_reg - (2 * std_dev)
    
    current_price = close.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_atr = atr.iloc[-1]
    
    # Keltner Channel (ATR Band)
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    keltner_upper = ema20.iloc[-1] + (current_atr * ATR_MULTIPLIER)
    
    # Logika Sygnałów
    trend_up = current_price > ema200.iloc[-1] and ema100.iloc[-1] > ema200.iloc[-1]
    momentum = current_rsi >= RSI_MOMENTUM
    
    signal = "WAIT"
    risk_note = "Neutral"
    sl_price = 0.0
    
    if trend_up and momentum:
        if current_price > keltner_upper:
            signal = "⚠️ BUY (HIGH RISK)"
            risk_note = f"Cena > ATR (+{((current_price/keltner_upper)-1)*100:.1f}%)"
            sl_price = current_price * (1 - SL_TIGHT_PCT)
        else:
            signal = "🟢 BUY (MOMENTUM)"
            space_left = ((keltner_upper - current_price) / current_price) * 100
            risk_note = f"Bezpiecznie. Zapas: {space_left:.1f}%"
            sl_price = current_price * (1 - SL_NORMAL_PCT)
            
    # Dane do zwrotu
    return {
        "Price": current_price,
        "RSI": current_rsi,
        "Signal": signal,
        "Risk Note": risk_note,
        "SL": sl_price,
        "ATR": current_atr,
        "Regression": lin_reg[-1],
        "Reg_Upper": upper_channel[-1],
        "Reg_Lower": lower_channel[-1],
        "DataFrame": df # Potrzebne do wykresu
    }

# === INTERFEJS APLIKACJI ===

st.set_page_config(page_title="BOSSA TERMINAL", layout="wide", page_icon="🚀")

# Pasek boczny
with st.sidebar:
    st.header("⚙️ Ustawienia")
    capital = st.number_input("Twój Kapitał (PLN/USD)", value=10000, step=1000)
    risk_per_trade = st.slider("Ryzyko na transakcję (%)", 0.5, 5.0, 1.0) / 100
    st.divider()
    show_all = st.checkbox("Pokaż wszystkie (nawet WAIT)", value=False)
    
    st.info("Dane pobierane są na żywo z Yahoo Finance (opóźnienie 15min dla GPW).")

# Główny ekran
st.title("🚀 BOSSA 3.3 TERMINAL")
st.markdown("### System Analizy Momentum + Zarządzanie Ryzykiem")

tickers = load_tickers_from_sheet()

if not tickers:
    st.warning("Brak tickerów. Sprawdź link do arkusza.")
    st.stop()

# Kontener na wyniki
results = []
progress_bar = st.progress(0)
status_text = st.empty()

# Pętla po tickerach
for i, ticker in enumerate(tickers):
    status_text.text(f"Analizuję: {ticker}...")
    progress_bar.progress((i + 1) / len(tickers))
    
    df = get_data(ticker)
    if df is not None:
        try:
            res = calculate_signals(df)
            res['Ticker'] = ticker
            results.append(res)
        except:
            pass # Ignoruj błędy obliczeń

progress_bar.empty()
status_text.empty()

# Konwersja do DataFrame
res_df = pd.DataFrame(results)

# Filtrowanie
if not show_all:
    final_df = res_df[res_df['Signal'].str.contains("BUY")]
else:
    final_df = res_df

# KPI
col1, col2, col3 = st.columns(3)
buy_signals = len(res_df[res_df['Signal'].str.contains("BUY")])
high_risk = len(res_df[res_df['Signal'].str.contains("HIGH RISK")])

col1.metric("Znalezione Okazje", buy_signals)
col2.metric("Wysokie Ryzyko", high_risk)
col3.metric("Analizowane Spółki", len(res_df))

st.divider()

# Wyświetlanie Kart Okazji (Jeśli są)
if not final_df.empty:
    st.subheader("📡 Radar Okazji")
    
    for index, row in final_df.iterrows():
        # Kolor ramki zależny od sygnału
        border_color = "red" if "HIGH" in row['Signal'] else "green"
        
        with st.expander(f"{row['Ticker']} | {row['Signal']} | RSI: {row['RSI']:.1f}", expanded=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                st.metric("Cena", f"{row['Price']:.2f}")
                st.write(f"**Stop Loss:** {row['SL']:.2f}")
                
                # Kalkulator pozycji
                risk_amount = capital * risk_per_trade
                dist_to_sl = row['Price'] - row['SL']
                if dist_to_sl > 0:
                    qty = risk_amount / dist_to_sl
                    st.info(f"💰 Kup: **{int(qty)} szt.** (Ryzyko: {risk_amount:.0f})")
            
            with c2:
                # Wykres Plotly
                df_chart = row['DataFrame'].tail(100)
                fig = go.Figure()
                
                # Świece
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'], high=df_chart['High'],
                                low=df_chart['Low'], close=df_chart['Close'], name='Cena'))
                
                # Regresja (wizualizacja ryzyka)
                # Musimy przeliczyć regresję dla wycinka wykresu, żeby pasowała wizualnie
                # (Tutaj uproszczenie: rysujemy linię SL)
                fig.add_hline(y=row['SL'], line_dash="dash", line_color="red", annotation_text="STOP LOSS")
                
                # Layout
                fig.update_layout(xaxis_rangeslider_visible=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
            with c3:
                st.write("**Analiza Ryzyka:**")
                st.write(row['Risk Note'])
                
                # Regresja info
                dist_reg = (row['Price'] - row['Regression']) / row['Price'] * 100
                st.write(f"Odchylenie od Regresji: **{dist_reg:.1f}%**")
                
                if row['Price'] > row['Reg_Upper']:
                    st.error("Cena powyżej 2SD Regresji! (Ekstremum)")
                elif row['Price'] < row['Reg_Lower']:
                    st.success("Cena poniżej 2SD (Tanio)")
                else:
                    st.write("Cena w kanale regresji.")

else:
    st.info("Brak sygnałów BUY. Zmień filtry lub idź na spacer :)")

# Tabela zbiorcza na dole
st.divider()
st.subheader("📋 Tabela Wszystkich Walorów")
st.dataframe(res_df[['Ticker', 'Price', 'Signal', 'RSI', 'Risk Note', 'SL']].style.applymap(
    lambda x: 'color: red' if 'HIGH' in x else ('color: green' if 'BUY' in x else ''), subset=['Signal']
))
