import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. 頁面設定 (TradingView 風格) ---
st.set_page_config(page_title="TradeGenius Options Pro", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# TradingView 配色
TV_BG_COLOR = "#131722"
TV_GRID_COLOR = "#363a45"
TV_UP_COLOR = "#089981"
TV_DOWN_COLOR = "#f23645"
TEXT_COLOR = "#d1d4dc"

st.markdown(f"""
<style>
    .stApp {{ background-color: {TV_BG_COLOR}; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    
    /* 卡片樣式 */
    .metric-box {{
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 15px;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .metric-label {{ color: #787b86; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }}
    .metric-val {{ color: #d1d4dc; font-size: 24px; font-weight: 600; }}
    .metric-sub {{ font-size: 13px; color: #999; margin-top: 5px; }}
    
    /* AI 原因列表 */
    .reason-item {{ font-size: 13px; margin-bottom: 4px; }}
    .reason-good {{ color: {TV_UP_COLOR}; }}
    .reason-bad {{ color: {TV_DOWN_COLOR}; }}
    
    /* 期權卡片 */
    .option-tag {{ 
        display: inline-block; padding: 4px 8px; border-radius: 4px; 
        font-size: 12px; font-weight: bold; color: #fff; margin-bottom: 8px;
    }}
    .tag-call {{ background-color: {TV_UP_COLOR}; }}
    .tag-put {{ background-color: {TV_DOWN_COLOR}; }}
    .tag-neutral {{ background-color: #FF9800; }}
</style>
""", unsafe_allow_html=True)

# --- 2. 核心運算邏輯 ---

def calculate_indicators(df):
    # MA
    for ma in [10, 20, 50, 100, 200]:
        df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    # KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    # Volatility (20日歷史波動率)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    
    return df

def predict_future(df, days=5):
    recent = df.tail(20).reset_index()
    x = np.array(range(len(recent)))
    y = recent['Close'].values
    slope, intercept = np.polyfit(x, y, 1)
    future_prices = []
    start_x = x[-1]
    for i in range(1, days + 1):
        future_prices.append(slope * (start_x + i) + intercept)
    return future_prices, slope

def get_ai_analysis(df):
    score = 50
    row = df.iloc[-1]
    reasons = []
    
    # 1. 趨勢分析
    if row['Close'] > row['SMA20']: 
        score += 15
        reasons.append(("good", "股價站穩 20日均線 (短期強勢)"))
    else: 
        score -= 15
        reasons.append(("bad", "股價跌破 20日均線 (短期轉弱)"))
        
    if row['SMA50'] > row['SMA200']: 
        score += 10
        reasons.append(("good", "均線呈現多頭排列 (長期看漲)"))
    
    # 2. 動能分析
    if row['MACD'] > row['Signal']: 
        score += 15
        reasons.append(("good", "MACD 出現黃金交叉訊號"))
    elif row['MACD'] < row['Signal']:
        score -= 10
        # reasons.append(("bad", "MACD 處於死亡交叉")) # 避免太多字，只顯示關鍵的
        
    if row['RSI'] < 30: 
        score += 15
        reasons.append(("good", "RSI 進入超賣區 (醞釀反彈)"))
    elif row['RSI'] > 70: 
        score -= 15
        reasons.append(("bad", "RSI 進入超買區 (回調風險)"))

    # 3. 成交量
    vol_ma = df['Volume'].rolling(5).mean().iloc[-1]
    if row['Volume'] > vol_ma * 1.5:
         if row['Close'] > row['Open']:
             reasons.append(("good", "爆量上漲 (資金流入)"))
         else:
             reasons.append(("bad", "爆量下跌 (恐慌拋售)"))

    score = max(0, min(100, score))
    return score, reasons

def generate_option_strategy(df, score, slope, pred_price):
    current_price = df['Close'].iloc[-1]
    volatility = df['Volatility'].iloc[-1]
    
    # 根據分數和波動率推算策略
    if score >= 70:
        # 強力看漲
        target = current_price * 1.05
        strike = round(current_price * 1.02, 1) # 價外一點點
        strategy = "Long Call (買入看漲)"
        desc = f"強勢多頭。建議買入價外 Call，捕捉升勢。\n目標行使價: ${strike}"
        tag_class = "tag-call"
    elif score <= 30:
        # 強力看跌
        target = current_price * 0.95
        strike = round(current_price * 0.98, 1)
        strategy = "Long Put (買入看跌)"
        desc = f"趨勢轉弱。建議買入 Put 避險或做空。\n目標行使價: ${strike}"
        tag_class = "tag-put"
    else:
        # 盤整 / 震盪
        if volatility > 0.4: # 波動大但方向不明
             strategy = "Long Straddle (跨式)"
             desc = "波動劇烈但方向未定，博大行情突破。"
             tag_class = "tag-neutral"
        else: # 波動小
             strategy = "Iron Condor (鐵鷹)"
             desc = "區間震盪，建議賺取時間價值 (Theta)。"
             tag_class = "tag-neutral"
             
    return strategy, desc, tag_class, volatility

# --- 3. 介面 (Sidebar) ---
st.sidebar.markdown("## ⚙️ 參數設定")
ticker = st.sidebar.text_input("代碼", value="TSLA").upper()
period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y", "5y"], value="1y")
st.sidebar.markdown("---")
st.sidebar.info("💡 **操作提示**：\n滾動滑鼠縮放圖表。\nAI 期權僅供策略參考，非投資建議。")

if not ticker: st.stop()

# --- 4. 數據處理 ---
try:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: st.error("無效代碼"); st.stop()
    df = calculate_indicators(df)
    ai_prices, slope = predict_future(df, days=5)
    score, reasons = get_ai_analysis(df)
    opt_strat, opt_desc, opt_tag, vol = generate_option_strategy(df, score, slope, ai_prices[-1])
except Exception as e:
    st.error(f"數據錯誤: {e}"); st.stop()

# --- 5. 頂部儀表板 (Dashboard) ---
# 上半部：價格與預測
c1, c2, c3, c4 = st.columns(4)
last_close = df['Close'].iloc[-1]
change = last_close - df['Close'].iloc[-2]
pct_change = (change / df['Close'].iloc[-2]) * 100
color_hex = TV_UP_COLOR if change >= 0 else TV_DOWN_COLOR

with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{ticker} 現價</div>
        <div class="metric-val" style="color:{color_hex}">${last_close:.2f}</div>
        <div class="metric-sub" style="color:{color_hex}">{change:+.2f} ({pct_change:+.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    pred_p = ai_prices[-1]
    p_chg = ((pred_p - last_close)/last_close)*100
    p_color = TV_UP_COLOR if p_chg > 0 else TV_DOWN_COLOR
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 5日預測</div>
        <div class="metric-val" style="color:{p_color}">${pred_p:.2f}</div>
        <div class="metric-sub">趨勢預估: {p_chg:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# 下半部：AI 分析詳解 + 期權
c_ai_score, c_ai_opt = st.columns([1, 1])

with c_ai_score:
    score_color = TV_UP_COLOR if score >= 60 else "#FFD700" if score >= 40 else TV_DOWN_COLOR
    reasons_html = ""
    for r_type, r_text in reasons:
        cls = "reason-good" if r_type == "good" else "reason-bad"
        icon = "✅" if r_type == "good" else "🔻"
        reasons_html += f"<div class='reason-item {cls}'>{icon} {r_text}</div>"
        
    st.markdown(f"""
    <div class="metric-box" style="text-align: left;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="metric-label">AI 綜合評分</span>
            <span class="metric-val" style="font-size:20px; color:{score_color}">{score}/100</span>
        </div>
        <hr style="border-color: #333; margin: 5px 0;">
        {reasons_html if reasons_html else "<div style='color:#666; font-size:13px;'>暫無明顯訊號</div>"}
    </div>
    """, unsafe_allow_html=True)

with c_ai_opt:
    st.markdown(f"""
    <div class="metric-box" style="text-align: left;">
        <div class="metric-label">🤖 AI 期權推介 (IV: {vol*100:.1f}%)</div>
        <span class="option-tag {opt_tag}">{opt_strat}</span>
        <div style="font-size: 14px; color: #ccc; line-height: 1.4;">{opt_desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. 繪圖核心 (Plotly) ---
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.55, 0.1, 0.15, 0.1, 0.1],
    specs=[[{"secondary_y": False}], [{}], [{}], [{}], [{}]],
)

# K線
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="K線", increasing_line_color=TV_UP_COLOR, increasing_fillcolor=TV_UP_COLOR,
    decreasing_line_color=TV_DOWN_COLOR, decreasing_fillcolor=TV_DOWN_COLOR
), row=1, col=1)

# MA Lines
ma_colors = {10: '#FFEB3B', 20: '#2962FF', 50: '#FF9800', 100: '#FFFFFF', 200: '#F50057'}
for ma, col in ma_colors.items():
    fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA{ma}'], mode='lines', name=f'MA{ma}', line=dict(color=col, width=1)), row=1, col=1)

# AI 預測
last_date = df.index[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
fig.add_trace(go.Scatter(
    x=[last_date] + future_dates, y=[last_close] + ai_prices,
    mode='lines+markers', name='AI 預測', line=dict(color='#00E676', width=2, dash='dot')
), row=1, col=1)

# 指標
colors_vol = [TV_DOWN_COLOR if c < o else TV_UP_COLOR for c, o in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Vol', marker_color=colors_vol), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='MACD', marker_color='#666'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='DIF', line=dict(color='#2962FF')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='DEA', line=dict(color='#FF6D00')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9C27B0')), row=4, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#666", row=4, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#666", row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='#FFD600', width=1)), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='#00E5FF', width=1)), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='#E040FB', width=1)), row=5, col=1)

# Layout
fig.update_layout(
    height=1000, margin=dict(l=10, r=40, t=10, b=10),
    paper_bgcolor=TV_BG_COLOR, plot_bgcolor=TV_BG_COLOR, font=dict(color=TEXT_COLOR),
    showlegend=False, hovermode='x unified', dragmode='pan'
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR, showline=False, rangeslider_visible=False)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR, showline=False)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'scrollZoom']})
