import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- 網頁設定 ---
st.set_page_config(page_title="美港股速查儀", layout="wide")
st.title("📈 美股與港股技術分析儀")
st.markdown("輸入股票代碼，即時查看 K 線圖與移動平均線 (SMA)。數據來源：Yahoo Finance。")

# --- 側邊欄輸入 ---
st.sidebar.header("設定")
# 預設改為 AAPL 測試，因為港股有時會有延遲數據問題
ticker = st.sidebar.text_input("股票代碼 (例如: AAPL, 0700.HK, TSLA)", value="0700.HK").upper()
period = st.sidebar.selectbox("時間範圍", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
st.sidebar.markdown("---")
st.sidebar.write("技術指標：")
show_sma20 = st.sidebar.checkbox("顯示 20日均線 (SMA20)", value=True)
show_sma50 = st.sidebar.checkbox("顯示 50日均線 (SMA50)", value=True)

# --- 數據獲取函數 ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, time_period):
    try:
        stock = yf.Ticker(ticker_symbol)
        # 獲取歷史數據
        df = stock.history(period=time_period)
        
        if df.empty:
            return None, f"搵唔到代碼 {ticker_symbol} 嘅數據，請檢查輸入。"
            
        # 計算移動平均線
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # 嘗試獲取公司名，如果失敗就用代碼代替
        try:
            info = stock.info
            # 檢查 info 是否為 None (有時 yfinance 會返回 None)
            if info is None:
                 company_name = ticker_symbol
            else:
                 company_name = info.get('shortName', ticker_symbol)
        except:
            company_name = ticker_symbol
            
        return df, company_name
        
    except Exception as e:
        return None, str(e)

# --- 主畫面邏輯 ---

if ticker:
    with st.spinner(f'正在分析 {ticker} ...'):
        data, name_or_error = get_stock_data(ticker, period)

        if data is None:
            st.error(f"錯誤: {name_or_error}")
        else:
            # 顯示最新報價資訊
            last_close = data['Close'].iloc[-1]
            last_date = data.index[-1].strftime('%Y-%m-%d')
            
            st.header(f"{name_or_error} ({ticker})")
            
            # --- 修正了這一行 ---
            # 舊版錯誤: using=f"{last_close:.2f}" -> 導致 TypeError
            # 新版正確: value=f"{last_close:.2f}"
            st.metric(label="最新收盤價", value=f"{last_close:.2f}", delta=f"日期: {last_date}")

            # --- 繪製互動圖表 (Plotly) ---
            fig = go.Figure()

            # 1. 加入 K 線圖
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="K線"
            ))

            # 2. 加入均線
            if show_sma20:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['SMA20'], 
                    mode='lines', name='SMA 20 (短期)', line=dict(color='orange', width=1.5)
                ))
            
            if show_sma50:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['SMA50'], 
                    mode='lines', name='SMA 50 (中期)', line=dict(color='royalblue', width=1.5)
                ))

            # 圖表設定
            fig.update_layout(
                title=f'{ticker} 股價走勢與均線分析',
                yaxis_title='股價',
                xaxis_rangeslider_visible=False,
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 顯示原始數據
            with st.expander("查看詳細數據表"):
                st.dataframe(data.sort_index(ascending=False).round(2))

else:
    st.info("請在左側輸入股票代碼開始分析。")
