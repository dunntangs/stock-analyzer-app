import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as si 
import math

# --- 1. 頁面設定與樣式 ---
st.set_page_config(page_title="TradeGenius AI Options", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

TV_BG_COLOR = "#131722"
TV_UP_COLOR = "#089981"
TV_DOWN_COLOR = "#f23645"
TEXT_COLOR = "#d1d4dc"

st.markdown(f"""
<style>
    .stApp {{ background-color: {TV_BG_COLOR}; font-family: 'Roboto', sans-serif; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    
    .metric-box {{
        background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 8px;
        padding: 15px; height: 100%; text-align: left;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .metric-label {{ color: #787b86; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
    .metric-val {{ color: #d1d4dc; font-size: 22px; font-weight: 700; margin-top: 5px; }}
    .metric-sub {{ font-size: 12px; margin-top: 2px; }}
    
    /* 期權專用樣式 */
    .opt-title {{ color: #fff; font-size: 16px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
    .opt-detail-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px; color: #ccc; }}
    .greek-tag {{ background: #333; padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #aaa; }}
    .recomm-badge {{ background: linear-gradient(45deg, #2962ff, #00d2ff); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}

    /* 預測卡片樣式 */
    .forecast-day {{ font-size: 14px; color: #787b86; }}
    .forecast-price {{ font-size: 18px; font-weight: bold; }}

</style>
""", unsafe_allow_html=True)

# --- 2. 數學模型: Black-Scholes Greeks 計算 ---

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S: 現價, K: 行使價, T: 到期時間(年), r: 無風險利率, sigma: 波動率
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            delta = si.norm.cdf(d1, 0.0, 1.0)
            gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
        else:
            delta = si.norm.cdf(d1, 0.0, 1.0) - 1
            gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
            
        return delta, gamma
    except:
        return 0, 0

# --- 3. 核心運算: AI 選股與期權獵人 ---

def calculate_indicators(df):
    # MA, RSI, MACD
    for ma in [10, 20, 50, 100, 200]: df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12).mean(); exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # KDJ (Stochastics) 計算
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(span=3).mean()
    df['D'] = df['K'].ewm(span=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 歷史波動率 (HV)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252)
    
    return df

def get_ai_sentiment(df):
    score = 50
    row = df.iloc[-1]
    reasons = []
    
    if row['Close'] > row['SMA20']: score += 15; reasons.append("股價站上 MA20")
    else: score -= 15; reasons.append("股價跌破 MA20")
    
    if row['MACD'] > row['Signal']: score += 15; reasons.append("MACD 金叉")
    if row['K'] > row['D'] and df['K'].iloc[-2] <= df['D'].iloc[-2]: 
        score += 10; reasons.append("KDJ 金叉")

    if row['RSI'] < 30: score += 10; reasons.append("RSI 超賣")
    elif row['RSI'] > 70: score -= 10; reasons.append("RSI 超買")
    
    score = max(0, min(100, score))
    direction = "call" if score >= 55 else "put" if score <= 45 else "neutral"
    return score, direction, reasons

def hunt_best_option(ticker_obj, current_price, direction, hv):
    """
    AI 期權獵人：搜尋真實期權鏈，計算 Greeks，找出最佳合約
    """
    best_option = None
    
    try: # <--- try 區塊開始
        exps = ticker_obj.options
        if not exps: return None
        
        target_date = None
        min_days = 25
        max_days = 60
        
        today = datetime.now()
        for date_str in exps:
            # 這是之前出錯行數周圍的邏輯
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            if min_days <= days_to_exp <= max_days:
                target_date = date_str
                break
        
        if not target_date: target_date = exps[0] 
        
        opt_chain = ticker_obj.option_chain(target_date)
        options = opt_chain.calls if direction == "call" else opt_chain.puts
        
        candidates = []
        r = 0.05 
        T = (datetime.strptime(target_date, "%Y-%m-%d") - today).days / 365.0
        
        for index, row in options.iterrows():
            strike = row['strike']
            price = row['lastPrice']
            iv = row['impliedVolatility']
            volume = row['volume']
            openInterest = row['openInterest']
            
            if volume < 5 or openInterest < 10: continue
            if iv <= 0 or price <= 0: continue

            delta, gamma = black_scholes(current_price, strike, T, r, iv, direction)
            
            if 0.3 <= abs(delta) <= 0.7:
                score = (gamma * 100) * (np.log(volume + 1)) / (iv * 10)
                
                candidates.append({
                    "contractSymbol": row['contractSymbol'],
                    "strike": strike,
                    "expiry": target_date,
                    "price": price,
                    "delta": delta,
                    "gamma": gamma,
                    "iv": iv,
                    "volume": volume,
                    "score": score,
                    "type": direction 
                })
        
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_option = candidates[0]
            
        return best_option
        
    except Exception as e: # <--- except 區塊在這裡，確保 try 區塊是完整的
        return None

# --- 4. 介面 Sidebar ---
st.sidebar.markdown("## ⚙️ 參數設定")
ticker = st.sidebar.text_input("代碼", value="TSLA").upper()
period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y"], value="6mo")
st.sidebar.markdown("---")

if not ticker: st.stop()

# --- 新增 AI 預測邏輯函數 ---
def predict_future_trend(df, direction, days=5):
    """
    基於最近動量和 AI 評分進行的簡易線性預測
    """
    current_price = df['Close'].iloc[-1]
    
    if len(df) < 10: 
        return [current_price] * days 
    
    recent_closes = df['Close'].iloc[-10:].values
    
    try:
        X = np.arange(len(recent_closes))
        slope, intercept, _, _, _ = si.linregress(X, recent_closes)
        momentum = slope / 5 
    except ValueError:
        momentum = 0.0 

    if direction == "call":
        daily_change = max(momentum, 0.0005 * current_price) 
    elif direction == "put":
        daily_change = min(momentum, -0.0005 * current_price) 
    else:
        daily_change = momentum

    predicted_prices = [current_price]
    for i in range(days):
        next_price = predicted_prices[-1] + daily_change
        predicted_prices.append(next_price)
        
    return predicted_prices[1:] 

# --- 5. 數據處理 ---
try:
    if ticker.startswith("HK."):
        yf_ticker = ticker.split(".")[1] + ".HK"
    elif ticker.startswith("US."):
        yf_ticker = ticker.split(".")[1]
    else:
        yf_ticker = ticker
        
    stock = yf.Ticker(yf_ticker)
    df = stock.history(period=period)
    
    if df.empty: st.error(f"無效代碼或無數據: {yf_ticker}"); st.stop()
    
    df = calculate_indicators(df)
    score, direction, reasons = get_ai_sentiment(df)
    current_price = df['Close'].iloc[-1]
    hv = df['HV'].iloc[-1]
    
    best_opt = hunt_best_option(stock, current_price, direction, hv)
    
    predicted_prices = predict_future_trend(df, direction, days=5)
    
    # 修正日期處理：確保 future_dates 是 pd.Timestamp 類型
    future_dates = []
    # 使用最後一個有效日期作為起點，確保它是可運算的 Timestamp
    last_valid_date = pd.to_datetime(df.index[-1].date()) 
    current_date = last_valid_date
    while len(future_dates) < 5:
        current_date += timedelta(days=1)
        # 檢查是否為交易日 (0=週一, 4=週五)
        if current_date.weekday() < 5: 
            # 統一日期類型，忽略時區差異
            future_dates.append(pd.to_datetime(current_date))
            
except Exception as e:
    st.error(f"數據處理錯誤: {type(e).__name__}: {e}"); st.stop()

# --- 6. Dashboard ---

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

# A. 股價卡片
with c1:
    last_close = df['Close'].iloc[-1]
    change = last_close - df['Close'].iloc[-2]
    pct = (change / df['Close'].iloc[-2])*100
    color = TV_UP_COLOR if change >= 0 else TV_DOWN_COLOR
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{ticker} 現價</div>
        <div class="metric-val" style="color:{color}">${last_close:.2f}</div>
        <div class="metric-sub" style="color:{color}">{change:+.2f} ({pct:+.2f}%)</div>
        <div class="metric-label" style="margin-top:15px;">HV (歷史波幅)</div>
        <div style="color:#ccc; font-size:16px;">{hv*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# B. AI 評分卡片
with c2:
    score_color = TV_UP_COLOR if score >= 55 else TV_DOWN_COLOR if score <= 45 else "#FF9800"
    sentiment_text = "看漲 (Bullish)" if direction == "call" else "看跌 (Bearish)" if direction == "put" else "中性 (Neutral)"
    
    reason_html = "".join([f"<div>• {r}</div>" for r in reasons])
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 趨勢綜合分析</div>
        <div class="metric-val" style="color
