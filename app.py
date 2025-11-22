import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面基礎設定 (年輕化 UI) ---
st.set_page_config(page_title="TradeGenius AI", layout="wide", page_icon="⚡")

# 自定義 CSS：讓介面更有現代感 (Dark Mode Neon Style)
st.markdown("""
<style>
    /* 全局字體優化 */
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* 頂部數據卡片樣式 */
    .metric-container {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: transform 0.2s;
    }
    .metric-container:hover { transform: translateY(-5px); border-color: #00d2ff; }
    
    .metric-label { color: #888; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #fff; margin: 5px 0; }
    .metric-delta.up { color: #00e676; font-weight: bold; }
    .metric-delta.down { color: #ff3d00; font-weight: bold; }
    
    /* AI 標籤 */
    .ai-tag {
        background-color: #2962ff; color: white; padding: 4px 12px; 
        border-radius: 20px; font-size: 0.8rem; font-weight: bold; display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心運算邏輯 ---

def calculate_tech_indicators(df):
    # MA 線 (全部計算)
    for ma in [10, 20, 50, 100, 200]:
        df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()

    # RSI (14)
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
    
    return df

def predict_future(df, days=5):
    # 簡單線性回歸預測
    recent = df.tail(20).reset_index()
    x = np.array(range(len(recent)))
    y = recent['Close'].values
    slope, intercept = np.polyfit(x, y, 1)
    
    future_prices = []
    start_x = x[-1]
    for i in range(1, days + 1):
        future_prices.append(slope * (start_x + i) + intercept)
    
    return future_prices, slope

def get_ai_score(df):
    score = 50
    row = df.iloc[-1]
    reasons = []
    
    # 簡單評分邏輯
    if row['Close'] > row['SMA20']: score += 10; reasons.append("股價高於月線 (強)")
    else: score -= 10
    
    if row['MACD'] > row['Signal']: score += 15; reasons.append("MACD 金叉")
    
    if row['RSI'] < 30: score += 15; reasons.append("RSI 超賣 (博反彈)")
    elif row['RSI'] > 70: score -= 15; reasons.append("RSI 超買 (小心回調)")
    
    if row['SMA50'] > row['SMA200']: score += 10; reasons.append("均線多頭排列")
    
    return max(0, min(100, score)), reasons

# --- 3. 介面佈局 ---

# 側邊欄：簡約設定
st.sidebar.title("⚡ 設定")
ticker = st.sidebar.text_input("股票代碼", value="TSLA").upper() # 預設改為 TSLA
period = st.sidebar.select_slider("時間範圍", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")
st.sidebar.caption("AI 分析模式已啟動")

# 主畫面
if ticker:
    # 獲取數據
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty: st.error("無數據，請檢查代碼"); st.stop()
        info = stock.info
    except: st.error("連線錯誤"); st.stop()

    # 計算
    df = calculate_tech_indicators(df)
    ai_prices, slope = predict_future(df, days=5) # 改為 5 日
    score, reasons = get_ai_score(df)

    # --- 頂部數據 Dashboard ---
    last_close = df['Close'].iloc[-1]
    last_open = df['Open'].iloc[-1]
    change = last_close - df['Close'].iloc[-2]
    pct_change = (change / df['Close'].iloc[-2]) * 100
    color_cls = "up" if change >= 0 else "down"
    sign = "+" if change >= 0 else ""
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{ticker} 收盤價</div>
            <div class="metric-value">${last_close:.2f}</div>
            <div class="metric-delta {color_cls}">{sign}{change:.2f} ({sign}{pct_change:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        trend = "🚀 看漲" if slope > 0 else "🔻 看跌"
        pred_price = ai_prices[-1]
        p_change = ((pred_price - last_close)/last_close)*100
        p_color = "up" if p_change > 0 else "down"
        st.markdown(f"""
        <div class="metric-container">
            <div class="ai-tag">AI 預測 (5日後)</div>
            <div class="metric-value">${pred_price:.2f}</div>
            <div class="metric-delta {p_color}">{trend} {p_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        score_color = "#00e676" if score >= 60 else "#ffea00" if score >= 40 else "#ff3d00"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">AI 綜合評分</div>
            <div class="metric-value" style="color:{score_color}">{score}</div>
            <div class="metric-label">/ 100</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        vol_str = f"{df['Volume'].iloc[-1]/1000000:.2f}M"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">成交量</div>
            <div class="metric-value">{vol_str}</div>
            <div class="metric-label">最新交易日</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 4. 超級圖表 (Plotly Subplots) ---
    # 建立 5 行 Subplots (主圖, Vol, MACD, RSI, KDJ)
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.1, 0.15, 0.15, 0.1], # 分配高度比例
        specs=[[{"secondary_y": False}], [{}], [{}], [{}], [{}]],
        subplot_titles=("股價 & 均線 & AI預測", "成交量", "MACD", "RSI", "KDJ")
    )

    # Row 1: K線 + MA + AI
    # 實色 K 線 (TradingView 風格: 升=綠#089981, 跌=紅#F23645)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="K線",
        increasing_line_color='#089981', increasing_fillcolor='#089981',
        decreasing_line_color='#F23645', decreasing_fillcolor='#F23645'
    ), row=1, col=1)

    # MA 線 (全部預設顯示，用不同顏色)
    ma_colors = {10: '#FFD700', 20: '#00d2ff', 50: '#ff00ff', 100: '#ffffff', 200: '#ff3d00'}
    for ma, color in ma_colors.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f'SMA{ma}'], mode='lines', 
            name=f'MA{ma}', line=dict(color=color, width=1)
        ), row=1, col=1)

    # AI 預測線 (5日)
    last_date = df.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
    pred_x = [last_date] + future_dates
    pred_y = [last_close] + ai_prices
    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y, mode='lines+markers', name='AI 預測路徑',
        line=dict(color='#00e676', width=2, dash='dot')
    ), row=1, col=1)

    # Row 2: Volume (成交量)
    colors_vol = ['#F23645' if c < o else '#089981' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='成交量', marker_color=colors_vol
    ), row=2, col=1)

    # Row 3: MACD
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='MACD Hist', marker_color='gray'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='DIF', line=dict(color='#2962ff')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='DEA', line=dict(color='#ff6d00')), row=3, col=1)

    # Row 4: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#ab47bc')), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # Row 5: KDJ
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='#ffd600', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='#00e5ff', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='#d500f9', width=1)), row=5, col=1)

    # 圖表 Layout 優化
    fig.update_layout(
        height=1200, # 拉長圖表高度
        template="plotly_dark", # 深色主題
        xaxis_rangeslider_visible=False, # 隱藏底部滑條
        hovermode='x unified', # 滑鼠對齊顯示所有數據
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("請輸入代碼開始")
