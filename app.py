import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import re
import time
from datetime import datetime, timedelta
import urllib3

# 關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
st.set_page_config(
    page_title="上櫃挖掘 (Pro版)",
    page_icon="🚀",
    layout="wide"
)

# ==========================================
# PART 1: 資料抓取核心
# ==========================================
@st.cache_data(ttl=3600*4)
def get_tpex_top_buys(input_date):
    """
    Method: GET
    Params: searchType=buy
    """
    roc_year = input_date.year - 1911
    date_str = f"{roc_year}/{input_date.strftime('%m/%d')}"
    
    API = "https://www.tpex.org.tw/www/zh-tw/insti/sitcStat"
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/domestic-inst/day.html"
    }
    
    params = {
        "type": "Daily",
        "date": date_str,
        "searchType": "buy", 
        "id": "",
        "response": "json"
    }

    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=10, verify=False)
        r.raise_for_status()
        data = r.json()
        
        if "tables" in data and len(data["tables"]) > 0:
            raw_data = data["tables"][0]["data"]
            results = []
            for row in raw_data:
                try:
                    code = re.sub(r"\D", "", str(row[1] or ""))
                    name = row[2]
                    net_buy = int(str(row[5]).replace(',', ''))
                    
                    if len(code) == 4 and net_buy > 0:
                        results.append({
                            "code": code,
                            "name": name,
                            "net": net_buy
                        })
                except:
                    continue
            return pd.DataFrame(results), date_str
        else:
            return None, date_str

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, date_str

# ==========================================
# PART 2: K線與趨勢分析 (優化版)
# ==========================================

# 優化 1: 加入快取，調整參數改變時不用重新下載
@st.cache_data(ttl=600) 
def fetch_5m(code, days=5):
    if not code: return pd.DataFrame()
    for suf in (".TWO", ".TW"):  
        try:
            ticker = f"{code}{suf}"
            # 抓取資料
            df = yf.Ticker(ticker).history(
                period=f"{days}d", interval="5m", auto_adjust=False, prepost=False
            )
            if df is None or df.empty: continue
            
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].copy()
            
            # 時區處理
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Taipei")
            else:
                df.index = df.index.tz_convert("Asia/Taipei")
            
            return df
        except Exception:
            continue
    return pd.DataFrame()

def judge_trend_300(df, window=300, r2_thresh=0.10, strength_abs=0.01):
    if df.empty: return "N/A", 0, 0, 0, []

    # 取最後 window 根 K 棒
    d = df.tail(window).dropna(subset=["Close"])
    n = len(d)
    last = float(d["Close"].iloc[-1])
    sma = float(d["Close"].mean()) if n else 0
    
    # 優化 2: 準備給 Sparkline 用的數據 (標準化，避免圖形跑掉)
    # 取最後 50 根 K 棒畫圖就好，不然圖會太密
    sparkline_data = d["Close"].tail(50).tolist()

    if n < max(60, int(window * 0.6)):
        return "資料不足", last, sma, 0, sparkline_data

    # --- 線性迴歸核心 (您的 R2 邏輯) ---
    x = np.arange(n, dtype=float)
    y = d["Close"].astype(float).values
    slope, b = np.polyfit(x, y, 1)
    
    # 計算 R2 (決定係數): 衡量趨勢的穩定度
    yhat = slope * x + b
    ss_res = float(np.sum((y - yhat)**2))         # 殘差平方和
    ss_tot = float(np.sum((y - y.mean())**2))     # 總變異
    r2 = 0.0 if ss_tot == 0 else (1 - ss_res / ss_tot)
    
    # 計算強度 (Strength): 斜率 * 期間 / 均價
    # 意義：這段期間內，股價總共漲/跌了百分之多少
    strength = float(slope * window / y.mean())

    # 判斷邏輯
    up_ok   = (strength >=  strength_abs) and (last > y.mean()) and (r2 >= r2_thresh)
    down_ok = (strength <= -strength_abs) and (last < y.mean()) and (r2 >= r2_thresh)
    
    direction = "➡️ 盤整"
    if up_ok: direction = "🔥 上升"
    elif down_ok: direction = "📉 下降"

    return direction, last, sma, strength, sparkline_data

# ==========================================
# PART 3: Streamlit UI
# ==========================================

st.sidebar.header("🚀 參數設定")
today = datetime.now().date()
selected_date = st.sidebar.date_input("選擇日期", value=today, max_value=today)

st.sidebar.subheader("篩選條件")
top_n = st.sidebar.slider("顯示前幾名買超", 5, 50, 20)
window_size = st.sidebar.number_input("趨勢判斷 K 棒數", value=300, help="300根5分K約等於5-6個交易日")

col1, col2 = st.sidebar.columns(2)
with col1:
    strength_th = st.number_input("強度門檻", value=0.01, step=0.005, format="%.3f", help="數值越大，要求漲幅越陡峭")
with col2:
    r2_th = st.number_input("R2 穩定度", value=0.10, step=0.05, format="%.2f", help="數值越大(Max 1.0)，要求走勢越平滑穩定，雜訊越少")

st.title("🚀 上櫃挖掘機 (Pro)")
st.caption(f"核心邏輯：強度(漲幅) > {strength_th} 且 R2(穩定度) > {r2_th}")

if st.button("開始掃描", type="primary"):
    
    status_text = st.empty()
    status_text.info(f"正在抓取 {selected_date} 資料...")
    
    df_buys, date_str = get_tpex_top_buys(selected_date)
    
    if df_buys is None or df_buys.empty:
        status_text.error(f"❌ 無法取得 {selected_date} 資料 (可能為假日或無資料)。")
    else:
        status_text.success(f"✅ 成功取得 {date_str} 買超排行！")
        
        targets = df_buys.head(top_n).to_dict('records')
        st.info(f"正在分析前 {len(targets)} 檔股票的 5分K 趨勢...")
        
        final_results = []
        progress_bar = st.progress(0)
        
        for i, stock in enumerate(targets):
            progress_bar.progress((i + 1) / len(targets))
            code = stock['code']
            
            df_k = fetch_5m(code, days=10)
            
            # 加入 sparkline_data 回傳
            direction, last, sma, strength, sparkline = judge_trend_300(
                df_k, window=window_size, r2_thresh=r2_th, strength_abs=strength_th
            )
            
            final_results.append({
                "代碼": code,
                "名稱": stock['name'],
                "買超張數": int(stock['net']),
                "現價": round(last, 2) if last else 0,
                "趨勢方向": direction,
                "強度": round(strength, 4) if strength else 0,
                "R2穩定度": 0, # 這裡原本沒回傳R2，如果您需要看R2數值，judge_trend_300 需修改回傳 r2
                "走勢預覽": sparkline # 給 LineChartColumn 用
            })
            
        progress_bar.empty()
        
        res_df = pd.DataFrame(final_results)
        
        # --- 優化顯示設定 ---
        st.write(f"### 📊 買超趨勢 ({date_str})")
        
        st.dataframe(
            res_df,
            column_config={
                "代碼": st.column_config.TextColumn("代碼"),
                "名稱": st.column_config.TextColumn("名稱"),
                "買超張數": st.column_config.NumberColumn(
                    "投信買超 (張)", 
                    format="%d",
                    help="當日投信買賣超張數"
                ),
                "現價": st.column_config.NumberColumn("現價", format="$%.2f"),
                "趨勢方向": st.column_config.TextColumn("趨勢"),
                "強度": st.column_config.ProgressColumn(
                    "趨勢強度",
                    format="%.4f",
                    min_value=-0.1,
                    max_value=0.1,
                    help="紅色代表強勢上漲，藍色代表下跌"
                ),
                # 優化 3: 加入走勢圖 Sparkline
                "走勢預覽": st.column_config.LineChartColumn(
                    "近50根K棒走勢",
                    y_min=None, 
                    y_max=None
                )
            },
            use_container_width=True,
            hide_index=True
        )
