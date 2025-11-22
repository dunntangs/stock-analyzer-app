import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面設定 (TradingView 沉浸模式) ---
st.set_page_config(page_title="TradeGenius Pro", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# TradingView 經典配色常數
TV_BG_COLOR = "#131722" # 背景深灰
TV_GRID_COLOR = "#363a45" # 網格線
TV_UP_COLOR = "#089981" # 升 (綠)
TV_DOWN_COLOR = "#f23645" # 跌 (紅)
TEXT_COLOR = "#d1d4dc"

# 進階 CSS：移除 Streamlit 多餘邊距，模仿 App 質感
st.markdown(f"""
<style>
    /* 全局背景改為 TradingView 深灰 */
    .stApp {{ background-color: {TV_BG_COLOR}; }}
    
    /* 隱藏 Streamlit 頂部 Hamburger Menu 和 Footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* 調整頂部 Padding，讓內容更靠上 */
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    
    /* 數據卡片樣式 */
    .metric-box {{
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .metric-label {{ color: #787b86; font-size: 12px; text-transform: uppercase; }}
    .metric-val {{ color: #d1d4dc; font-size: 28px; font-weight: 600; font-family: 'Roboto', sans-serif; }}
    .metric-delta {{ font-size: 14px; font-weight: 500; }}
    
    /* 自訂 Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {TV_BG_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: #555; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

# --- 2. 核心邏輯 (保持不變) ---

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
    if row['Close'] > row['SMA20']: score += 10
    else: score -= 10
    if row['SMA50'] > row['SMA200']: score += 10
    if row['MACD'] > row['Signal']: score += 15
    if row['RSI'] < 30: score += 15
    elif row['RSI'] > 70: score -= 15
    
    score = max(0, min(100, score))
    suggestion = "強立買入" if score >= 70 else "強立賣出" if score <= 30 else "觀望/持有"
    color = TV_UP_COLOR if score >= 50 else TV_DOWN_COLOR
    return score, suggestion, color

# --- 3. 介面 (Sidebar) ---
st.sidebar.markdown("## ⚙️ 參數設定")
ticker = st.sidebar.text_input("代碼", value="TSLA").upper()
period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y", "5y"], value="1y")
st.sidebar.markdown("---")
st.sidebar.info("💡 **操作提示**：\n\n圖表已啟用 **滑鼠滾輪縮放**。\n\n滑鼠指住圖表，滾動即可放大縮細 (Zoom)，按住左鍵拖曳 (Pan)。")

if not ticker:
    st.stop()

# --- 4. 數據處理 ---
try:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: st.error("無效代碼"); st.stop()
    df = calculate_indicators(df)
    ai_prices, slope = predict_future(df, days=5)
    score, suggestion, score_color = get_ai_analysis(df)
except Exception as e:
    st.error(f"數據錯誤: {e}"); st.stop()

# --- 5. 頂部儀表板 (Dashboard) ---
last_close = df['Close'].iloc[-1]
change = last_close - df['Close'].iloc[-2]
pct_change = (change / df['Close'].iloc[-2]) * 100
color_arrow = "▲" if change >= 0 else "▼"
color_hex = TV_UP_COLOR if change >= 0 else TV_DOWN_COLOR

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">現價 ({ticker})</div>
        <div class="metric-val" style="color:{color_hex}">${last_close:.2f}</div>
        <div class="metric-delta" style="color:{color_hex}">{color_arrow} {abs(change):.2f} ({pct_change:+.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    pred_p = ai_prices[-1]
    p_chg = ((pred_p - last_close)/last_close)*100
    p_color = TV_UP_COLOR if p_chg > 0 else TV_DOWN_COLOR
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 預測 (5日後)</div>
        <div class="metric-val" style="color:{p_color}">${pred_p:.2f}</div>
        <div class="metric-delta" style="color:{p_color}">{p_chg:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 綜合評分</div>
        <div class="metric-val" style="color:{score_color}">{score}</div>
        <div class="metric-delta" style="color:{score_color}">{suggestion}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    vol_m = df['Volume'].iloc[-1] / 1e6
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">成交量 (Volume)</div>
        <div class="metric-val">{vol_m:.2f}M</div>
        <div class="metric-delta">Latest Session</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. 繪圖核心 (Plotly + TradingView Style) ---
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.55, 0.1, 0.15, 0.1, 0.1],
    specs=[[{"secondary_y": False}], [{}], [{}], [{}], [{}]],
)

# [Main Chart] K線
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="K線",
    increasing_line_color=TV_UP_COLOR, increasing_fillcolor=TV_UP_COLOR,
    decreasing_line_color=TV_DOWN_COLOR, decreasing_fillcolor=TV_DOWN_COLOR
), row=1, col=1)

# [Main Chart] MA Lines
ma_colors = {10: '#FFEB3B', 20: '#2962FF', 50: '#FF9800', 100: '#FFFFFF', 200: '#F50057'}
for ma, col in ma_colors.items():
    fig.add_trace(go.Scatter(
        x=df.index, y=df[f'SMA{ma}'], mode='lines', name=f'MA{ma}',
        line=dict(color=col, width=1)
    ), row=1, col=1)

# [Main Chart] AI Prediction
last_date = df.index[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
fig.add_trace(go.Scatter(
    x=[last_date] + future_dates, y=[last_close] + ai_prices,
    mode='lines+markers', name='AI 預測',
    line=dict(color='#00E676', width=2, dash='dot')
), row=1, col=1)

# [Vol]
colors_vol = [TV_DOWN_COLOR if c < o else TV_UP_COLOR for c, o in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol), row=2, col=1)

# [MACD]
fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Hist', marker_color='#666'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='DIF', line=dict(color='#2962FF')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='DEA', line=dict(color='#FF6D00')), row=3, col=1)

# [RSI]
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9C27B0')), row=4, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#666", row=4, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#666", row=4, col=1)

# [KDJ]
fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='#FFD600', width=1)), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='#00E5FF', width=1)), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='#E040FB', width=1)), row=5, col=1)

# --- 7. TradingView 樣式微調 ---
fig.update_layout(
    height=1000,
    margin=dict(l=10, r=40, t=10, b=10), # 邊距最小化
    paper_bgcolor=TV_BG_COLOR, # 圖表外框顏色
    plot_bgcolor=TV_BG_COLOR,  # 圖表內容顏色
    font=dict(color=TEXT_COLOR),
    showlegend=False, # 隱藏 Legend 增加空間，鼠標 Hover 會顯示
    hovermode='x unified', # 十字準星模式
    
    # 開啟鼠標滾輪縮放 (重點)
    dragmode='pan', # 預設拖曳模式 (平移)
)

# 設定 X 軸樣式 (移除 Range Slider，加入十字線)
fig.update_xaxes(
    showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR,
    showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="#666666",
    rangeslider_visible=False # 隱藏底部滑條
)
# 設定 Y 軸樣式
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR,
    showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="#666666",
)

# 渲染圖表 (啟用 scrollZoom)
st.plotly_chart(
    fig, 
    use_container_width=True, 
    config={
        'scrollZoom': True,        # ✅ 啟用滑鼠滾輪縮放
        'displayModeBar': True,    # 顯示工具列
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'scrollZoom', 'zoomIn', 'zoomOut', 'resetScale2d'],
        'displaylogo': False       # 隱藏 Plotly Logo
    }
)
