import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as si
from futu import *
import yfinance as yf 


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
    
    .opt-title {{ color: #fff; font-size: 16px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
    .opt-detail-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px; color: #ccc; }}
    .greek-tag {{ background: #333; padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #aaa; }}
    .recomm-badge {{ background: linear-gradient(45deg, #2962ff, #00d2ff); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)


# --- 2. 數學模型: Black-Scholes Greeks 計算 ---

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """S: 現價, K: 行使價, T: 到期時間(年), r: 無風險利率, sigma: 波動率"""
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


# --- 3. 數據獲取 (使用 yfinance 獲取歷史數據) ---

@st.cache_data(ttl=3600)
def get_stock_data(code, period):
    """使用 yfinance 獲取 K 線數據，並將 Futu 代碼轉為 yfinance 格式"""
    
    # 轉換代碼格式 (例如 US.TSLA -> TSLA, HK.00700 -> 00700.HK)
    if code.startswith("US."):
        yf_code = code.split(".")[1]
    elif code.startswith("HK."):
        yf_code = code.split(".")[1] + ".HK"
    else:
        yf_code = code

    try:
        ticker_obj = yf.Ticker(yf_code)
        
        # 獲取歷史 K 線數據
        df = ticker_obj.history(period=period)
        
        if df.empty:
            return None, f"無法獲取 {code} 數據 (yfinance)"
        
        name = ticker_obj.info.get('longName', yf_code)
    except Exception as e:
         return None, f"yfinance 錯誤: {e}"
        
    return df, name


# --- 4. 技術指標與 AI 邏輯 ---

def calculate_indicators(df):
    for ma in [10, 20, 50, 200]: df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12).mean(); exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252)
    return df


def create_candlestick_chart(df, ticker_name):
    """創建帶有 MA 和 MACD/RSI 的 K 線圖"""
    
    # 創建主 K 線圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='K線',
                                 increasing_line_color=TV_UP_COLOR,
                                 decreasing_line_color=TV_DOWN_COLOR), row=1, col=1)

    # 加入移動平均線
    for ma in [20, 50]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA{ma}'], name=f'MA{ma}', 
                                 line=dict(width=1)), row=1, col=1)

    # MACD 子圖
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'], name='MACD 柱',
                         marker_color='#2962ff'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#ff9900')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='#f6006e')), row=2, col=1)

    # RSI 子圖
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#008080')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f23645", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=TV_UP_COLOR, row=3, col=1)

    # 佈局美化
    fig.update_layout(
        title=f'<span style="color:{TEXT_COLOR}; font-size:24px;">{ticker_name} K線分析 ({df.index[-1].strftime("%Y-%m-%d")})</span>',
        height=900,
        plot_bgcolor=TV_BG_COLOR,
        paper_bgcolor=TV_BG_COLOR,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title='價格'),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False, title='MACD'),
        xaxis3=dict(showgrid=False),
        yaxis3=dict(showgrid=False, title='RSI'),
        font=dict(color=TEXT_COLOR),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 設置每個子圖的背景
    fig.update_xaxes(rangeselector_visible=False, 
                     rangeslider_visible=False, 
                     showgrid=False, 
                     minor_griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor='#2a2e39')
    
    return fig


def get_ai_sentiment(df):
    score = 50
    row = df.iloc[-1]
    reasons = []
    
    if row['Close'] > row['SMA20']: score += 15; reasons.append("股價站上 MA20")
    else: score -= 15; reasons.append("股價跌破 MA20")
    
    if row['MACD'] > row['Signal']: score += 15; reasons.append("MACD 金叉")
    if row['RSI'] < 30: score += 10; reasons.append("RSI 超賣")
    elif row['RSI'] > 70: score -= 10; reasons.append("RSI 超買")
    
    score = max(0, min(100, score))
    direction = "call" if score >= 55 else "put" if score <= 45 else "neutral"
    return score, direction, reasons

# 這是 Futu API 獲取期權鏈的核心函數
def hunt_best_option(code, current_price, direction, hv, _quote_ctx):
    """
    AI 期權獵人：使用 Futu API 獲取真實期權鏈
    _quote_ctx 前面加底線，告訴 Streamlit 不要哈希它
    """
    try:
        # 1. 獲取到期日 (尋找 25-60 天內)
        ret, exps_df = _quote_ctx.get_option_expiry_date(code, OptionMarket.ALL)
        if ret != RET_OK or exps_df.empty: raise ValueError("無期權鏈數據")

        today = datetime.now()
        target_date_str = None
        
        for index, row in exps_df.iterrows():
            exp_date = datetime.strptime(row['strike_time'], "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            if 25 <= days_to_exp <= 60:
                target_date_str = row['strike_time']
                break
        
        if not target_date_str: target_date_str = exps_df.iloc[0]['strike_time']
        
        # 2. 獲取期權鏈
        option_type = OptionCondType.CALL if direction == "call" else OptionCondType.PUT
        ret, df_chain = _quote_ctx.get_option_chain(
            code=code, 
            market=OptionMarket.ALL, 
            index_option_type=OptionType.ALL, 
            datetime=target_date_str, 
            cond_type=option_type
        )
        if ret != RET_OK or df_chain.empty: raise ValueError("獲取期權鏈失敗")
        
        candidates = []
        r = 0.05 
        T = (datetime.strptime(target_date_str, "%Y-%m-%d") - today).days / 365.0
        
        for index, row in df_chain.iterrows():
            if 'implied_volatility' not in row or row['implied_volatility'] <= 0: continue
            if 'price' not in row or row['price'] <= 0: continue
            if 'volume' not in row or row['volume'] < 10: continue

            strike = row['strike']
            iv = row['implied_volatility']
            price = row['price']
            
            delta, gamma = black_scholes(current_price, strike, T, r, iv, direction)
            
            if 0.3 <= abs(delta) <= 0.7:
                score = (gamma * 100) * (np.log(row['volume']+1)) / (iv * 10)
                
                candidates.append({
                    "contractSymbol": row['code'],
                    "strike": strike,
                    "expiry": target_date_str,
                    "price": price,
                    "delta": delta,
                    "gamma": gamma,
                    "iv": iv,
                    "volume": row['volume'],
                    "score": score
                })
        
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[0]
        else:
            return None

    except Exception as e:
        # Fallback 模式：如果 Futu 找不到數據，就顯示模擬建議
        strike_theory = round(current_price * (1.02 if direction == "call" else 0.98), 1)
        days_theory = 30
        
        return {
            "contractSymbol": f"SIM-{direction.upper()}-{strike_theory}",
            "strike": strike_theory,
            "expiry": (datetime.now() + timedelta(days=days_theory)).strftime("%Y-%m-%d"),
            "price": current_price * 0.05,
            "delta": 0.50 if direction == "call" else -0.50,
            "gamma": 0.05,
            "iv": hv,
            "volume": "N/A (模擬)",
            "score": 0,
            "is_simulation": True 
        }

# --- 5. 應用程式主邏輯 ---

def main_app(quote_ctx):
    
    # --- 關鍵變數初始化 (解決 NameError) ---
    name = "數據未載入"
    current_price = 0.0
    hv = 0.0
    best_opt = None 
    # ------------------------------------
    
    # --- 介面 Sidebar ---
    st.sidebar.markdown("## ⚙️ 參數設定")
    ticker_input = st.sidebar.text_input("代碼 (US.TSLA, HK.00700)", value="US.TSLA").upper()
    period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y"], value="6mo")
    st.sidebar.markdown("---")
    st.sidebar.info("K線數據源: yfinance\n期權數據源: Futu OpenD (需本地運行)")

    if not ticker_input: st.stop()

    # --- 數據處理 ---
    try:
        # 呼叫 yfinance 獲取歷史數據
        df, name = get_stock_data(ticker_input, period) 
        if df is None: st.error(f"無法獲取 {ticker_input} 數據: {name}"); st.stop()
        
        df = calculate_indicators(df)
        score, direction, reasons = get_ai_sentiment(df)
        
        # 定義關鍵變數
        current_price = df['Close'].iloc[-1]
        hv = df['HV'].iloc[-1]
        
        # 執行 AI 期權獵人 (使用 Futu Context)
        best_opt = hunt_best_option(ticker_input, current_price, direction, hv, quote_ctx)
        
    except Exception as e:
        st.error(f"應用程式運行錯誤: {e}"); st.stop()

    # --- 6. Dashboard 及 圖表渲染 ---
    
    st.markdown(f"## {name} ({ticker_input}) AI 分析儀表板")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">現價</div><div class="metric-val">${current_price:.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">AI 情緒分數</div><div class="metric-val">{score} / 100</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">暗示波動率 (HV)</div><div class="metric-val">{hv*100:.2f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">AI 傾向</div><div class="metric-val" style="color:{"#089981" if direction=="call" else "#f23645"}">{direction.upper()}</div></div>', unsafe_allow_html=True)


    if best_opt:
        with col5:
            st.markdown(f'<div class="metric-box"><div class="metric-label">AI 獵人推薦</div><div class="metric-val">{best_opt["contractSymbol"]}</div><div class="metric-sub">Delta: {best_opt["delta"]:.2f} / Gamma: {best_opt["gamma"]:.3f}</div></div>', unsafe_allow_html=True)

    # 7. 顯示圖表
    st.plotly_chart(create_candlestick_chart(df, name), use_container_width=True)


# --- 8. 程式進入點 (連線 OpenD) ---
if __name__ == '__main__':
    # 確保 OpenD 已經在你的電腦上運行，並且端口是 11111
    try:
        # 連線 Futu OpenD
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        
        # 運行 Streamlit 主程式
        main_app(quote_ctx)
        
    except Exception as e:
        # 當連線失敗時，顯示具體錯誤
        st.error(f"🚨 Futu OpenD 連接失敗！請檢查:\n1. 確保 OpenD 軟件已啟動且已解鎖。\n2. 確保端口設置為 11111。\n\n錯誤信息: {e}")
        
    finally:
        # 結束連線
        try:
            quote_ctx.close()
        except:
            pass
