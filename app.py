import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 網頁設定 ---
st.set_page_config(page_title="美港股 AI 分析終端機", layout="wide", page_icon="📈")

# --- CSS 優化 ---
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
    .score-high { color: #00e676; font-size: 24px; font-weight: bold; }
    .score-mid { color: #ffea00; font-size: 24px; font-weight: bold; }
    .score-low { color: #ff3d00; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📈 美港股 AI 全能分析儀")
st.caption("包含 MA(10/20/50/100/200), RSI, MACD, KDJ, 成交量及演算法走勢預測")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")
ticker = st.sidebar.text_input("股票代碼", value="0700.HK").upper()
period = st.sidebar.selectbox("數據範圍", ["6mo", "1y", "2y", "5y"], index=1)

st.sidebar.subheader("圖表顯示")
show_ma = st.sidebar.multiselect("移動平均線 (MA)", ["MA10", "MA20", "MA50", "MA100", "MA200"], default=["MA20", "MA50", "MA200"])
show_volume = st.sidebar.checkbox("顯示成交量", value=True)
indicator_select = st.sidebar.selectbox("副圖指標", ["MACD", "RSI", "KDJ", "全部隱藏"], index=0)

# --- 核心運算函數 ---

def calculate_indicators(df):
    # 1. 移動平均線 (SMA)
    for ma in [10, 20, 50, 100, 200]:
        df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()

    # 2. RSI (相對強弱指標) - 14日
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # 4. KDJ (隨機指標)
    low_list = df['Low'].rolling(9, min_periods=9).min()
    high_list = df['High'].rolling(9, min_periods=9).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def ai_analysis_score(df):
    """
    AI 評分邏輯 (0-100分)
    """
    score = 50 # 基礎分
    reasons = []
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # A. 趨勢分析 (30分)
    if current['Close'] > current['SMA20']:
        score += 10
        reasons.append("✅ 股價位於月線 (MA20) 之上 (短期強勢)")
    else:
        score -= 10
        reasons.append("⚠️ 股價跌破月線 (短期轉弱)")
        
    if current['SMA50'] > current['SMA200']:
        score += 10
        reasons.append("✅ 多頭排列 (MA50 > MA200)")
    
    if current['Close'] > current['SMA200']:
        score += 10
    else:
        score -= 10
        reasons.append("⚠️ 股價低於牛熊線 (MA200) (長期弱勢)")

    # B. 動能指標 (40分)
    # RSI
    if current['RSI'] < 30:
        score += 15
        reasons.append("🔥 RSI 進入超賣區 (反彈機會大)")
    elif current['RSI'] > 70:
        score -= 15
        reasons.append("❄️ RSI 進入超買區 (回調風險高)")
    
    # MACD
    if current['MACD'] > current['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
        score += 15
        reasons.append("🚀 MACD 出現黃金交叉 (買入訊號)")
    elif current['MACD'] < current['Signal_Line']:
        score -= 5

    # KDJ
    if current['K'] < 20 and current['K'] > current['D']:
        score += 10
        reasons.append("📈 KDJ 低位金叉")

    # C. 成交量 (10分)
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if current['Volume'] > vol_ma5 * 1.5:
        score += 10
        if current['Close'] > current['Open']:
            reasons.append("📢 爆量上漲 (資金流入)")
        else:
            reasons.append("⚠️ 爆量下跌 (恐慌拋售)")

    # 限制分數範圍 0-100
    score = max(0, min(100, score))
    
    return score, reasons

def predict_trend(df, days=3):
    """
    使用線性回歸預測未來 N 天
    """
    # 取最後 15 天數據做趨勢擬合
    recent_df = df.tail(15).reset_index() 
    x = np.array(range(len(recent_df)))
    y = recent_df['Close'].values
    
    # 計算斜率同截距 (y = mx + c)
    slope, intercept = np.polyfit(x, y, 1)
    
    # 預測未來
    last_x = x[-1]
    future_prices = []
    for i in range(1, days + 1):
        future_prices.append(slope * (last_x + i) + intercept)
        
    return future_prices, slope

# --- 主程式邏輯 ---

if ticker:
    with st.spinner('AI 正在運算數據、繪製圖表及進行預測...'):
        # 1. 獲取數據
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            info = stock.info
            name = info.get('shortName', ticker)
        except:
            st.error("找不到股票，請檢查代碼。")
            st.stop()

        if df.empty:
            st.error("數據庫為空，請嘗試其他股票。")
            st.stop()

        # 2. 計算指標
        df = calculate_indicators(df)
        
        # 3. AI 評分與分析
        ai_score, ai_reasons = ai_analysis_score(df)
        
        # 4. 趨勢預測
        pred_prices, trend_slope = predict_trend(df)
        trend_text = "📈 上升趨勢" if trend_slope > 0 else "📉 下跌趨勢"

        # --- 顯示 AI 儀表板 ---
        st.subheader(f"🤖 AI 智能分析報告: {name} ({ticker})")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <span>AI 綜合評分</span>
                <div class="{ 'score-high' if ai_score >= 70 else 'score-mid' if ai_score >= 40 else 'score-low' }">
                    {ai_score} / 100
                </div>
                <small>{'建議買入' if ai_score >= 70 else '建議觀望' if ai_score >= 40 else '建議賣出'}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            last_close = df['Close'].iloc[-1]
            pred_close = pred_prices[-1]
            change_pct = ((pred_close - last_close) / last_close) * 100
            color = "green" if change_pct > 0 else "red"
            
            st.markdown(f"""
            <div class="metric-card">
                <span>AI 推估未來 3 日走勢</span>
                <div style="color: {color}; font-size: 24px; font-weight: bold;">
                    {pred_close:.2f} ({change_pct:+.2f}%)
                </div>
                <small>{trend_text}</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""<div class="metric-card" style="text-align:left; font-size: 0.9em;"><b>關鍵訊號：</b><br>""", unsafe_allow_html=True)
            for r in ai_reasons[:3]: # 只顯示前3個重要原因
                st.markdown(f"{r}")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- 繪製圖表 ---
        st.markdown("---")
        
        # 設定子圖表 (如果選了副圖指標，就變成 2 行，否則 1 行)
        rows = 2 if indicator_select != "全部隱藏" else 1
        row_heights = [0.7, 0.3] if rows == 2 else [1.0]
        
        fig = make_subplots(
            rows=rows, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=row_heights
        )

        # [主圖] K 線
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="K線"
        ), row=1, col=1)

        # [主圖] MA 線
        colors = {'MA10': 'purple', 'MA20': 'orange', 'MA50': 'blue', 'MA100': 'black', 'MA200': 'red'}
        for ma_name in show_ma:
            col_name = f'SMA{ma_name[2:]}' # MA10 -> SMA10
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], 
                mode='lines', name=ma_name, line=dict(color=colors.get(ma_name, 'gray'), width=1)
            ), row=1, col=1)
            
        # [主圖] AI 預測線 (虛線)
        last_date = df.index[-1]
        # 產生未來日期
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
        # 連接今天和未來
        pred_x = [last_date] + future_dates
        pred_y = [df['Close'].iloc[-1]] + pred_prices
        
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode='lines+markers', name='AI 推估路徑',
            line=dict(color='gold', width=2, dash='dash')
        ), row=1, col=1)

        # [副圖] 根據選擇顯示
        if indicator_select == "MACD":
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color='gray'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='DIF', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='DEA', line=dict(color='orange')), row=2, col=1)
        
        elif indicator_select == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            # 加 30/70 線
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
        elif indicator_select == "KDJ":
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='purple')), row=2, col=1)

        # [成交量] 疊加在主圖底部 (透明度處理) 或 不顯示
        if show_volume:
            # 為了唔好遮住 K 線，將 Volume 縮細並放係主圖底部
            # 呢度我哋用一個簡單技巧，唔開新 Subplot，而係直接畫
            # 但因為比例問題，正規做法係開多個 Row，不過為咗慳位，我哋將佢放係副圖或者用 Text 顯示
            # 更新：如果選了指標，成交量就不顯示圖表，只顯示數值，避免太亂。
            # 或者我們可以強制開第3行。這裡為了美觀，我們只在 Tooltip 顯示，或者如果沒選副圖，就顯示在 Row 2
            if indicator_select == "全部隱藏":
                 fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color='rgba(100, 100, 100, 0.5)'), row=1, col=1)

        # 佈局設定
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            title_text=f"{ticker} 技術走勢圖",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示詳細數據
        with st.expander("查看詳細 OHLCV 及技術指標數據"):
            st.dataframe(df.sort_index(ascending=False).round(2))
