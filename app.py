import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="台股強勢股篩選器",
    page_icon="📈",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid repeated downloads
def get_stock_data(ticker, start_date, end_date):
    try:
        # Suppress yfinance progress bar for cleaner Streamlit output
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            # Optionally, add a warning here if you want to see individual stock failures
            # st.warning(f"無法取得 {ticker} 的資料。請檢查股票代碼或日期範圍。")
            return None
        return data
    except Exception as e:
        # st.error(f"下載 {ticker} 資料時發生錯誤: {e}")
        return None

def calculate_indicators(df):
    if df is None or df.empty:
        return None
    # Calculate Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean() # Add 120-day MA for longer trend
    return df

# --- Sidebar Inputs ---
st.sidebar.header("篩選條件設定")

today = datetime.now()
# Fetch 2 years of data to ensure enough history for 120-day MA and 6-month price change
default_start_date = today - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("資料起始日期", value=default_start_date)
end_date = st.sidebar.date_input("資料結束日期", value=today)

st.sidebar.subheader("股票清單")
# Default list of common Taiwan stocks with .TW suffix
default_stocks = "2330.TW, 2454.TW, 2303.TW, 2317.TW, 2603.TW, 2609.TW, 2881.TW, 2882.TW, 2884.TW, 1101.TW"
stock_symbols_input = st.sidebar.text_area(
    "輸入台股代碼 (以逗號或換行分隔，例如: 2330.TW, 2454.TW)",
    value=default_stocks,
    height=150
)
# Clean and format stock symbols
stock_symbols = [s.strip().upper() for s in stock_symbols_input.replace('\n', ',').split(',') if s.strip()]

st.sidebar.subheader("強勢股條件")
min_price = st.sidebar.number_input("最低股價 (元)", min_value=0.0, value=20.0, step=1.0)
min_volume = st.sidebar.number_input("最低日均成交量 (張)", min_value=0, value=1000, step=100)

price_change_period = st.sidebar.selectbox(
    "股價漲幅計算期間",
    options=["1個月", "3個月", "6個月"],
    index=1 # Default to 3 months
)
price_change_threshold = st.sidebar.slider("最低漲幅 (%)", min_value=-50, max_value=100, value=10, step=1)

volume_change_period = st.sidebar.selectbox(
    "成交量變化計算期間",
    options=["1個月", "3個月"],
    index=0 # Default to 1 month
)
volume_change_threshold = st.sidebar.slider("最低成交量變化 (%)", min_value=-50, max_value=100, value=0, step=1)

st.sidebar.markdown("---")
# Checkboxes for Moving Average conditions
ma20_check = st.sidebar.checkbox("股價高於20日均線", value=True)
ma60_check = st.sidebar.checkbox("股價高於60日均線", value=True)
ma120_check = st.sidebar.checkbox("股價高於120日均線", value=False) # Optional longer MA

# --- Main Application ---
st.title("📈 台股強勢股篩選器")
st.write("輸入股票代碼和篩選條件，找出符合條件的強勢股。")

if not stock_symbols:
    st.warning("請在左側輸入至少一個股票代碼。")
else:
    # Only run the screening process when the button is clicked
    if st.button("開始篩選"): 
        st.info(f"正在篩選 {len(stock_symbols)} 支股票，請稍候...")
        
        strong_stocks = []
        progress_bar = st.progress(0)
        
        # Approximate trading days for periods (assuming ~20 trading days per month)
        period_days_map = {"1個月": 20, "3個月": 60, "6個月": 120}

        for i, symbol in enumerate(stock_symbols):
            # Update progress bar with current stock being processed
            progress_bar.progress((i + 1) / len(stock_symbols), text=f"處理中: {symbol}")
            
            df = get_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                continue # Skip if data cannot be fetched or is empty
            
            df = calculate_indicators(df)
            if df is None or df.empty:
                continue # Skip if indicators cannot be calculated

            # Ensure there's enough data for all calculations (e.g., 120-day MA needs 120 data points)
            max_required_days = max(period_days_map.values()) + 1 # +1 for the current day
            if len(df) < max_required_days:
                # st.warning(f"跳過 {symbol}，因為資料不足以計算所有指標 (至少需要 {max_required_days} 天)。")
                continue

            latest_data = df.iloc[-1] # Get the latest available trading day's data
            
            # 1. 最低股價條件
            if latest_data['Close'] < min_price:
                continue
            
            # 2. 最低日均成交量 (過去20個交易日的平均成交量)
            avg_volume_period_days = 20 
            if len(df) < avg_volume_period_days:
                continue
            
            # Convert volume from shares to '張' (1張 = 1000股)
            avg_volume = df['Volume'].iloc[-avg_volume_period_days:].mean() / 1000 
            if avg_volume < min_volume:
                continue

            # 3. 股價漲幅條件
            price_period_days = period_days_map[price_change_period]
            
            # Ensure enough data points for the selected period
            if len(df) < price_period_days + 1: 
                continue
            
            # Get the price from 'price_period_days' ago (approximate trading days)
            # iloc[-1] is current day, iloc[-2] is 1 day ago, so iloc[-price_period_days - 1] is 'price_period_days' ago
            start_price_idx = -price_period_days - 1
            if abs(start_price_idx) > len(df):
                continue # Not enough data for the period
            start_price = df['Close'].iloc[start_price_idx]
            current_price = latest_data['Close']
            price_change_pct = ((current_price - start_price) / start_price) * 100 if start_price != 0 else 0
            
            if price_change_pct < price_change_threshold:
                continue

            # 4. 成交量變化條件
            volume_period_days = period_days_map[volume_change_period]
            
            # Need data for two consecutive periods to compare volume change
            if len(df) < volume_period_days * 2 + 1: 
                continue
            
            # Average volume for the current period
            current_period_avg_volume = df['Volume'].iloc[-volume_period_days:].mean()
            # Average volume for the previous period
            previous_period_avg_volume = df['Volume'].iloc[-volume_period_days*2:-volume_period_days].mean()
            
            volume_change_pct = ((current_period_avg_volume - previous_period_avg_volume) / previous_period_avg_volume) * 100 if previous_period_avg_volume != 0 else 0
            
            if volume_change_pct < volume_change_threshold:
                continue

            # 5. 股價高於均線條件
            ma_conditions_met = True
            # Check if MA values are available (not NaN) and if price is above MA
            if ma20_check and (pd.isna(latest_data['MA20']) or latest_data['Close'] < latest_data['MA20']):
                ma_conditions_met = False
            if ma60_check and (pd.isna(latest_data['MA60']) or latest_data['Close'] < latest_data['MA60']):
                ma_conditions_met = False
            if ma120_check and (pd.isna(latest_data['MA120']) or latest_data['Close'] < latest_data['MA120']):
                ma_conditions_met = False
            
            if not ma_conditions_met:
                continue
            
            # If all conditions are met, add the stock to the results list
            strong_stocks.append({
                "股票代碼": symbol,
                "最新股價": f"{latest_data['Close']:.2f}",
                f"{price_change_period}漲幅": f"{price_change_pct:.2f}%",
                f"{volume_change_period}成交量變化": f"{volume_change_pct:.2f}%",
                "20日均線": f"{latest_data['MA20']:.2f}" if pd.notna(latest_data['MA20']) else "N/A",
                "60日均線": f"{latest_data['MA60']:.2f}" if pd.notna(latest_data['MA60']) else "N/A",
                "120日均線": f"{latest_data['MA120']:.2f}" if pd.notna(latest_data['MA120']) else "N/A",
                "日均成交量(張)": f"{avg_volume:.0f}"
            })
        
        progress_bar.empty() # Clear the progress bar once screening is complete
        
        if strong_stocks:
            st.success(f"找到 {len(strong_stocks)} 支符合條件的強勢股！")
            results_df = pd.DataFrame(strong_stocks)
            # Display results in a sortable and interactive DataFrame
            st.dataframe(results_df.set_index("股票代碼"), use_container_width=True)
            
            # Provide a download button for the results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下載篩選結果 (CSV)",
                data=csv,
                file_name="strong_taiwan_stocks.csv",
                mime="text/csv",
            )
        else:
            st.warning("沒有找到符合條件的強勢股。請嘗試調整篩選條件或檢查股票代碼。")
