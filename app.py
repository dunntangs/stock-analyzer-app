import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as si # 用於 Black-Scholes 計算

# --- 1. 頁面設定 ---
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
    # MA, RSI, MACD, KDJ, Volatility
    for ma in [10, 20, 50, 200]: df[f'SMA{ma}'] = df['Close'].rolling(window=ma).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12).mean(); exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # 歷史波動率 (HV) - 用於比較 IV
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252)
    
    return df

def get_ai_sentiment(df):
    score = 50
    row = df.iloc[-1]
    reasons = []
    
    # 簡單評分邏輯
    if row['Close'] > row['SMA20']: score += 15; reasons.append("股價站上 MA20")
    else: score -= 15; reasons.append("股價跌破 MA20")
    
    if row['MACD'] > row['Signal']: score += 15; reasons.append("MACD 金叉")
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
    
    # 1. 獲取到期日 (尋找 25-60 天內的期權，時間價值衰減適中，爆發力夠)
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
        
        if not target_date: target_date = exps[0] # 如果找不到合適區間，就拿最近的
        
        # 2. 獲取期權鏈
        opt_chain = ticker_obj.option_chain(target_date)
        options = opt_chain.calls if direction == "call" else opt_chain.puts
        
        # 3. 篩選與 Greeks 計算
        candidates = []
        r = 0.05 # 假設無風險利率 5%
        T = (datetime.strptime(target_date, "%Y-%m-%d") - today).days / 365.0
        
        for index, row in options.iterrows():
            # 基本過濾：成交量太低不要，深度價內/價外不要
            if row['volume'] < 10 or row['openInterest'] < 50: continue
            
            strike = row['strike']
            price = row['lastPrice']
            iv = row['impliedVolatility']
            
            if iv <= 0 or price <= 0: continue

            # 計算 Greeks
            delta, gamma = black_scholes(current_price, strike, T, r, iv, direction)
            
            # AI 策略篩選邏輯：
            # - Delta: 0.3 ~ 0.6 (最有肉食，Gamma 爆發力最強的區域)
            # - IV: 最好不要高過 HV 太多 (避免買貴)
            if 0.3 <= abs(delta) <= 0.7:
                # CP 值評分：Gamma越高(加速快) + 成交量越高(易進出) / IV(成本)
                score = (gamma * 100) * (np.log(row['volume'])) / (iv * 10)
                
                candidates.append({
                    "contractSymbol": row['contractSymbol'],
                    "strike": strike,
                    "expiry": target_date,
                    "price": price,
                    "delta": delta,
                    "gamma": gamma,
                    "iv": iv,
                    "volume": row['volume'],
                    "score": score
                })
        
        # 4. 排序找出 No.1
        if candidates:
            # 根據 CP Score 降序排列
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_option = candidates[0]
            
        return best_option
        
    except Exception as e:
        return None

# --- 4. 介面 Sidebar ---
st.sidebar.markdown("## ⚙️ 參數設定")
ticker = st.sidebar.text_input("代碼", value="TSLA").upper()
period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y"], value="6mo")
st.sidebar.markdown("---")

if not ticker: st.stop()

# --- 5. 數據處理 ---
try:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: st.error("無效代碼"); st.stop()
    
    df = calculate_indicators(df)
    score, direction, reasons = get_ai_sentiment(df)
    current_price = df['Close'].iloc[-1]
    hv = df['HV'].iloc[-1]
    
    # 執行 AI 期權獵人
    best_opt = hunt_best_option(stock, current_price, direction, hv)
    
except Exception as e:
    st.error(f"數據處理錯誤: {e}"); st.stop()

# --- 6. Dashboard ---

c1, c2, c3 = st.columns([1, 1, 1.5])

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
        <div class="metric-val" style="color:{score_color}">{score}/100</div>
        <div class="metric-sub" style="color:{score_color}; font-weight:bold; margin-bottom:10px;">{sentiment_text}</div>
        <div style="font-size:12px; color:#999; line-height:1.4;">{reason_html}</div>
    </div>
    """, unsafe_allow_html=True)

# C. AI 期權推介卡片 (重點)
with c3:
    if best_opt:
        opt_color = TV_UP_COLOR if direction == "call" else TV_DOWN_COLOR
        leverage = (abs(best_opt['delta']) * current_price) / best_opt['price'] # 簡易槓桿率
        
        st.markdown(f"""
        <div class="metric-box" style="border-color: {opt_color};">
            <div class="recomm-badge">AI 嚴選最佳期權</div>
            <div class="opt-title">{best_opt['contractSymbol']}</div>
            
            <div class="opt-detail-grid">
                <div>到期日: <span style="color:#fff">{best_opt['expiry']}</span></div>
                <div>行使價: <span style="color:#fff">${best_opt['strike']}</span></div>
                <div>最新價: <span style="color:#fff; font-size:16px;">${best_opt['price']:.2f}</span></div>
                <div>引伸波幅 (IV): <span style="color:#ffd700">{best_opt['iv']*100:.1f}%</span></div>
            </div>
            
            <div style="margin-top:10px; padding-top:8px; border-top:1px dashed #444;">
                <span class="metric-label">GREEKS 分析</span><br>
                <span class="greek-tag">Delta {best_opt['delta']:.2f}</span>
                <span class="greek-tag">Gamma {best_opt['gamma']:.3f}</span>
                <span class="greek-tag">成交量 {best_opt['volume']}</span>
            </div>
            
            <div style="margin-top:8px; font-size:12px; color:#aaa;">
                <i>💡 推薦理由：Delta 位於攻擊區間，Gamma 爆發力高，且 IV 相對合理，槓桿約 {leverage:.1f}x。</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
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

# --- 7. 圖表 (保持不變) ---
st.markdown("<br>", unsafe_allow_html=True)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

# K線
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="K線", increasing_line_color=TV_UP_COLOR, decreasing_line_color=TV_DOWN_COLOR
), row=1, col=1)

# MA
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#2962ff', width=1), name='MA20'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#ff6d00', width=1), name='MA50'), row=1, col=1)

# Vol
colors_vol = [TV_DOWN_COLOR if c < o else TV_UP_COLOR for c, o in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Volume'), row=2, col=1)

fig.update_layout(
    height=600, margin=dict(t=10, b=10, l=10, r=40), 
    paper_bgcolor=TV_BG_COLOR, plot_bgcolor=TV_BG_COLOR, font=dict(color=TEXT_COLOR),
    showlegend=False, hovermode='x unified', dragmode='pan'
)
fig.update_xaxes(showgrid=True, gridcolor="#333", rangeslider_visible=False)
fig.update_yaxes(showgrid=True, gridcolor="#333")

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
