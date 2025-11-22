import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as si
from futu import * # <-- 引入 Futu API

# --- 1. 頁面設定 (TradingView 風格) ---
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

# --- 3. Futu API 數據獲取 (核心變動) ---

# 輔助函數：將 period (e.g. '6mo') 轉換為日期
def period_to_dates(period):
    end_date = datetime.now().strftime("%Y-%m-%d")
    if 'mo' in period:
        months = int(period.replace('mo', ''))
        start_date = (datetime.now() - timedelta(days=months*30)).strftime("%Y-%m-%d")
    elif 'y' in period:
        years = int(period.replace('y', ''))
        start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    return start_date, end_date

@st.cache_data(ttl=300) # 5分鐘更新一次
def get_stock_data(code, period, quote_ctx):
    """使用 Futu API 獲取 K 線數據"""
    start_date, end_date = period_to_dates(period)
    
    # 獲取 K 線數據
    ret, df = quote_ctx.get_history_kline(
        code, 
        start=start_date, 
        end=end_date, 
        kline_type=KLType.K_DAY, 
        autype=AuType.QFQ # 前復權
    )
    
    if ret != RET_OK:
        return None, f"Futu 錯誤: {df}"
    
    df.rename(columns={'time_key': 'Date', 'open': 'Open', 'high': 'High', 
                       'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 獲取公司名稱 (Futu 需要另外查詢)
    try:
        ret_info, df_info = quote_ctx.get_basic_info([code])
        name = df_info.iloc[0]['name'] if ret_info == RET_OK else code
    except:
        name = code
    
    return df, name


# --- 4. 技術指標與 AI 邏輯 (大部分不變) ---

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

def hunt_best_option(code, current_price, direction, hv, quote_ctx):
    """
    AI 期權獵人：使用 Futu API 獲取真實期權鏈
    """
    best_option = None
    
    try:
        # 1. 獲取到期日 (尋找 25-60 天內)
        ret, exps_df = quote_ctx.get_option_expiry_date(code, OptionMarket.ALL)
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
        ret, df_chain = quote_ctx.get_option_chain(
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
            # Futu API 數據清洗
            if row['volume'] < 10 or row['open_interest'] < 50: continue
            
            strike = row['strike']
            iv = row['implied_volatility']
            price = row['price']
            
            if iv <= 0 or price <= 0: continue
            
            # 計算 Greeks (用 Futu 的 IV)
            delta, gamma = black_scholes(current_price, strike, T, r, iv, direction)
            
            # AI 策略篩選邏輯：Delta 0.3 ~ 0.7 之間
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
    # --- 介面 Sidebar ---
    st.sidebar.markdown("## ⚙️ 參數設定")
    # 將 TSLA 代碼轉換為 Futu 格式 (HK.00700, US.TSLA)
    ticker_input = st.sidebar.text_input("代碼 (US.TSLA, HK.00700)", value="US.TSLA").upper()
    period = st.sidebar.select_slider("範圍", ["3mo", "6mo", "1y", "2y"], value="6mo")
    st.sidebar.markdown("---")
    st.sidebar.info("數據來源: Futu OpenD (需本地運行)")

    if not ticker_input: st.stop()

    # --- 數據處理 ---
    try:
        df, name = get_stock_data(ticker_input, period, quote_ctx)
        if df is None: st.error(f"無法獲取 {ticker_input} 數據: {name}"); st.stop()
        
        df = calculate_indicators(df)
        score, direction, reasons = get_ai_sentiment(df)
        current_price = df['Close'].iloc[-1]
        hv = df['HV'].iloc[-1]
        
        # 執行 AI 期權獵人
        best_opt = hunt_best_option(ticker_input, current_price, direction, hv, quote_ctx)
        
    except Exception as e:
        st.error(f"應用程式運行錯誤: {e}"); st.stop()

    # --- 6. Dashboard 及 圖表 (保持不變) ---
    # (Dashboard 及 Plotly 圖表繪製代碼省略，與上一版相同，確保你貼入完整代碼)

# --- 7. 程式進入點 (連線 OpenD) ---
if __name__ == '__main__':
    try:
        # 確保 OpenD 已經在你的電腦上運行，並且端口是 11111
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        
        # 運行 Streamlit 主程式
        main_app(quote_ctx)
        
    except Exception as e:
        st.error(f"🚨 Futu OpenD 連接失敗！請檢查:\n1. 確保 OpenD 軟件已啟動。\n2. 確保端口設置為 11111。\n\n錯誤信息: {e}")
        
    finally:
        # 結束連線
        try:
            quote_ctx.close()
        except:
            pass # 避免未連線時報錯
