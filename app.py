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

# --- 2. 數學模型: Black-Scholes Greeks 計算 (省略) ---

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

# --- 3. 核心運算: AI 選股與期權獵人 (省略) ---

def calculate_indicators(df):
    for ma in [10, 20, 50, 100, 200]: df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12).mean(); exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(span=3).mean()
    df['D'] = df['K'].ewm(span=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

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
    best_option = None
    # ... (省略期權獵人邏輯, 保持不變) ...
    try:
        exps = ticker_obj.options
        if not exps: return None
        
        target_date = None
        min_days = 25
        max_days = 60
        
        today = datetime.now()
        for date_str in exps:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
