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

# 使用 st.markdown 設置全域 CSS 樣式
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

# --- 2. 數學模型: Black-Scholes Greeks 計算 (不變) ---

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

# --- 3. 核心運算: AI 選股與期權獵人 (不變) ---

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
    # 這裡的邏輯與之前版本相同，為了簡潔省略重複貼出
    try:
        exps = ticker_obj.options
        if not exps: return None
        
        target_date = None
        min_days = 25
        max_days = 60
        
        today = datetime.now()
        for date_str in exps:
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
        
    except Exception as e:
        return None

# --- 4. 介面 Sidebar (不變) ---
st.sidebar.markdown("## ⚙️ 參數設定")
ticker = st.sidebar.text_input("代碼", value="TSLA").upper()
period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y"], value="6mo")
st.sidebar.markdown("---")

if not ticker: st.stop()

# --- AI 預測邏輯函數 (不變) ---
def predict_future_trend(df, direction, days=5):
    # 邏輯與之前版本相同，為了簡潔省略重複貼出
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

# --- 5. 數據處理 (不變) ---
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
    
    # 計算預測價格和日期
    predicted_prices = predict_future_trend(df, direction, days=5)
    
    future_dates = []
    last_valid_date = pd.to_datetime(df.index[-1].date()) 
    current_date = last_valid_date
    while len(future_dates) < 5:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5: 
            future_dates.append(pd.to_datetime(current_date))
            
except Exception as e:
    st.error(f"數據處理錯誤: {type(e).__name__}: {e}"); st.stop()

# --- 6. Dashboard (重點修正 C3 卡片) ---

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

# A. 股價卡片 (不變)
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

# B. AI 評分卡片 (不變)
with c2:
    score_color = TV_UP_COLOR if score >= 55 else TV_DOWN_COLOR if score <= 45 else "#FF9800"
    sentiment_text = "看漲 (Bullish)" if direction == "call" else "看跌 (Bearish)" if direction == "put" else "中性 (Neutral)"
    
    reason_html = "".join([f"<div>• {r}</div>" for r in reasons])
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 趨勢綜合分析</div>
        <div class="metric-val" style="color:{score_color}">{score}/100</div>
        <div class="metric-sub" style="color:{score_color}; font-weight:bold; margin-bottom:10px;">{sentiment_text}</div>
        <div style="font-size:12px; color:#999; line-height:1.4;">{reason_html}</div>
    </div>
    """, unsafe_allow_html=True)

# C. AI 期權推介卡片 (已修復 HTML 洩露問題)
with c3:
    if best_opt:
        opt_type_text = "看漲 (CALL)" if best_opt['type'] == 'call' else "看跌 (PUT)"
        opt_type_abbr = "C" if best_opt['type'] == 'call' else "P"
        opt_color = TV_UP_COLOR if best_opt['type'] == "call" else TV_DOWN_COLOR
        
        raw_symbol = best_opt['contractSymbol'] 
        strike_clean = best_opt['strike']       
        type_index = raw_symbol.find('C') if 'C' in raw_symbol else raw_symbol.find('P')
        
        if type_index != -1:
            date_part = raw_symbol[:type_index]
            cleaned_symbol = f"{date_part} {opt_type_abbr}{strike_clean:.2f}"
        else:
            cleaned_symbol = raw_symbol
        
        # 使用 f-string 和三引號確保 HTML 內容被正確封裝和渲染
        html_content = f"""
        <div class="metric-box" style="border-color: {opt_color};">
            <div class="recomm-badge">{opt_type_text} - AI 嚴選</div>
            <div class="opt-title">{cleaned_symbol}</div>
            
            <div class="opt-detail-grid">
                <div>到期日: <span style="color:#fff">{best_opt['expiry']}</span></div>
                <div>行使價: <span style="color:#fff">${best_opt['strike']:.2f}</span></div>
                <div>最新價: <span style="color:#fff; font-size:16px;">${best_opt['price']:.2f}</span></div>
                <div>引伸波幅 (IV): <span style="color:#ffd700">{best_opt['iv']*100:.1f}%</span></div>
            </div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True) 
        
    else:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">AI 期權獵人</div>
            <div style="margin-top:20px; color:#999;">
                ⚠️ 暫無合適期權推介。<br>
                <small>可能原因：數據源無即時期權鏈、流動性不足或市場處於休市。</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# D. AI 推估卡片 (已修復 HTML 洩露問題)
with c4:
    # 確保這裡的 HTML 內容也被正確封裝
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">AI 推估未來 5 日走勢</div>
        <div style="margin-top:10px;">
            <div style="display: flex; justify-content: space-between; gap: 10px;">
                {
                    "".join([
                        f'''
                        <div style="text-align: center;">
                            <div class="forecast-day">{future_dates[i].strftime('%m/%d')}</div>
                            <div class="forecast-price" style="color: {'#089981' if predicted_prices[i] >= current_price else '#f23645'}">
                                ${predicted_prices[i]:.2f}
                            </div>
                        </div>
                        ''' for i in range(5)
                    ])
                }
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 7. 圖表 (不變) ---
st.markdown("<br>", unsafe_allow_html=True)

fig = make_subplots(rows=4, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_heights=[0.5, 0.15, 0.2, 0.2])

# 預測線的數據準備
forecast_index = df.index[-1:].tolist() + future_dates
forecast_prices_plot = [df['Close'].iloc[-1]] + predicted_prices

# Row 1: K線 and MAs
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="K線", increasing_line_color=TV_UP_COLOR, decreasing_line_color=TV_DOWN_COLOR
), row=1, col=1)

# 加入 AI 預測線
fig.add_trace(go.Scatter(
    x=forecast_index, y=forecast_prices_plot, 
    line=dict(color='#ff9900', width=2, dash='dot'), 
    name='AI 預測',
    mode='lines+markers', marker=dict(size=4, symbol='circle') 
), row=1, col=1)

# MA Lines
fig.add_trace(go.Scatter(x=df.index, y=df['SMA10'], line=dict(color='#ffcccc', width=1), name='MA10'), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#2962ff', width=1), name='MA20'), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#ff6d00', width=1), name='MA50'), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], line=dict(color='#9c27b0', width=1), name='MA100'), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='#e91e63', width=1.5), name='MA200'), row=1, col=1) 


# Row 2: Volume
colors_vol = [TV_DOWN_COLOR if c < o else TV_UP_COLOR for c, o in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Volume'), row=2, col=1)

# Row 3: RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ff9900', width=1.5), name='RSI'), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#f23645", row=3, col=1, name='超買')
fig.add_hline(y=30, line_dash="dash", line_color=TV_UP_COLOR, row=3, col=1, name='超賣')
fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

# Row 4: MACD
fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'], name='MACD 柱',
                     marker_color=['#089981' if v >= 0 else '#f23645' for v in (df['MACD'] - df['Signal'])]), 
              row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff'), name='MACD'), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#ff6d00'), name='Signal'), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)


fig.update_layout(
    height=900, margin=dict(t=10, b=10, l=10, r=40), 
    paper_bgcolor=TV_BG_COLOR, plot_bgcolor=TV_BG_COLOR, font=dict(color=TEXT_COLOR),
    showlegend=False, hovermode='x unified', dragmode='pan'
)

fig.update_xaxes(showgrid=True, gridcolor="#333", rangeslider_visible=False)
fig.update_yaxes(showgrid=True, gridcolor="#333")

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
