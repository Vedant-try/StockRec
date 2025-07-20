import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import talib
import time
import os
import schedule
import threading
import json
import logging

# Configure logging
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined list of F&O-eligible stocks
ASSETS = [
    {"name": "Reliance Industries Ltd.", "symbol": "RELIANCE.NS"},
    {"name": "HDFC Bank Ltd.", "symbol": "HDFCBANK.NS"},
    {"name": "Tata Consultancy Services Ltd.", "symbol": "TCS.NS"},
    {"name": "Bharti Airtel Ltd.", "symbol": "BHARTIARTL.NS"},
    {"name": "ICICI Bank Ltd.", "symbol": "ICICIBANK.NS"},
    {"name": "State Bank of India", "symbol": "SBIN.NS"},
    {"name": "Infosys Ltd.", "symbol": "INFY.NS"},
    {"name": "Bajaj Finance Ltd.", "symbol": "BAJFINANCE.NS"},
    {"name": "Hindustan Unilever Ltd.", "symbol": "HINDUNILVR.NS"},
    {"name": "LIC of India", "symbol": "LICI.NS"},
    {"name": "ITC Ltd.", "symbol": "ITC.NS"},
    {"name": "Larsen & Toubro Ltd.", "symbol": "LT.NS"},
    {"name": "Kotak Mahindra Bank Ltd.", "symbol": "KOTAKBANK.NS"},
    {"name": "HCL Technologies Ltd.", "symbol": "HCLTECH.NS"},
    {"name": "Sun Pharmaceutical Industries Ltd.", "symbol": "SUNPHARMA.NS"},
    {"name": "Mahindra & Mahindra Ltd.", "symbol": "M&M.NS"},
    {"name": "Maruti Suzuki India Ltd.", "symbol": "MARUTI.NS"},
    {"name": "UltraTech Cement Ltd.", "symbol": "ULTRACEMCO.NS"},
    {"name": "Axis Bank Ltd.", "symbol": "AXISBANK.NS"},
    {"name": "NTPC Ltd.", "symbol": "NTPC.NS"},
    {"name": "Bajaj Finserv Ltd.", "symbol": "BAJAJFINSV.NS"},
    {"name": "Adani Ports and SEZ Ltd.", "symbol": "ADANIPORTS.NS"},
    {"name": "Hindustan Aeronautics Ltd.", "symbol": "HAL.NS"},
    {"name": "Oil & Natural Gas Corporation Ltd.", "symbol": "ONGC.NS"},
    {"name": "Titan Company Ltd.", "symbol": "TITAN.NS"},
    {"name": "Adani Enterprises Ltd.", "symbol": "ADANIENT.NS"},
    {"name": "Bharat Electronics Ltd.", "symbol": "BEL.NS"},
    {"name": "Wipro Ltd.", "symbol": "WIPRO.NS"},
    {"name": "Power Grid Corporation of India Ltd.", "symbol": "POWERGRID.NS"},
    {"name": "Avenue Supermarts Ltd.", "symbol": "DMART.NS"},
    {"name": "JSW Steel Ltd.", "symbol": "JSWSTEEL.NS"},
    {"name": "Tata Motors Ltd.", "symbol": "TATAMOTORS.NS"},
    {"name": "Coal India Ltd.", "symbol": "COALINDIA.NS"},
    {"name": "Nestle India Ltd.", "symbol": "NESTLEIND.NS"},
    {"name": "Bajaj Auto Ltd.", "symbol": "BAJAJ-AUTO.NS"},
    {"name": "Asian Paints Ltd.", "symbol": "ASIANPAINT.NS"},
    {"name": "InterGlobe Aviation Ltd.", "symbol": "INDIGO.NS"},
    {"name": "Indian Oil Corporation Ltd.", "symbol": "IOC.NS"},
    {"name": "DLF Ltd.", "symbol": "DLF.NS"},
    {"name": "Tata Steel Ltd.", "symbol": "TATASTEEL.NS"},
    {"name": "Jio Financial Services Ltd.", "symbol": "JIOFIN.NS"},
    {"name": "Trent Ltd.", "symbol": "TRENT.NS"},
    {"name": "Grasim Industries Ltd.", "symbol": "GRASIM.NS"},
    {"name": "Hindustan Zinc Ltd.", "symbol": "HINDZINC.NS"},
    {"name": "SBI Life Insurance Company Ltd.", "symbol": "SBILIFE.NS"},
    {"name": "Divi's Laboratories Ltd.", "symbol": "DIVISLAB.NS"},
    {"name": "Indian Railway Finance Corporation Ltd.", "symbol": "IRFC.NS"},
    {"name": "Vedanta Ltd.", "symbol": "VEDL.NS"},
    {"name": "Varun Beverages Ltd.", "symbol": "VBL.NS"},
    {"name": "Eternal Limited", "symbol": "ETERNAL.NS"},
]

FNO_ASSETS = [asset for asset in ASSETS]

# Function to fetch stock and futures data with delay
@st.cache_data
def fetch_data(tickers, start_date, end_date, max_retries=3, delay=1):
    data = {}
    failed_tickers = []
    for asset in tickers:
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(asset["symbol"])
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty and len(hist) >= 26:  # Minimum for MACD
                    # Technical indicators using talib
                    sma50 = talib.SMA(hist["Close"], timeperiod=50).iloc[-1] if len(hist) >= 50 else None
                    sma200 = talib.SMA(hist["Close"], timeperiod=200).iloc[-1] if len(hist) >= 200 else None
                    rsi = talib.RSI(hist["Close"], timeperiod=14).iloc[-1] if len(hist) >= 14 else None
                    macd, _, _ = talib.MACD(hist["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
                    macd = macd.iloc[-1] if len(hist) >= 26 and not np.isnan(macd.iloc[-1]) else None
                    bb_high, _, bb_low = talib.BBANDS(hist["Close"], timeperiod=20)
                    bb_high = bb_high.iloc[-1] if len(hist) >= 20 and not np.isnan(bb_high.iloc[-1]) else None
                    bb_low = bb_low.iloc[-1] if len(hist) >= 20 and not np.isnan(bb_low.iloc[-1]) else None
                    adx = talib.ADX(hist["High"], hist["Low"], hist["Close"], timeperiod=14).iloc[-1] if len(hist) >= 14 and not np.isnan(talib.ADX(hist["High"], hist["Low"], hist["Close"], timeperiod=14).iloc[-1]) else None
                    momentum = talib.MOM(hist["Close"], timeperiod=10).iloc[-1] if len(hist) >= 10 else None
                    slowk, _ = talib.STOCH(hist["High"], hist["Low"], hist["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
                    slowk = slowk.iloc[-1] if len(hist) >= 14 and not np.isnan(slowk.iloc[-1]) else None
                    vol_5d = hist["Volume"].rolling(window=5).mean().iloc[-1] if len(hist) >= 5 else None
                    vol_50d = hist["Volume"].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
                    # Futures data (simplified)
                    futures_price = hist["Close"].iloc[-1] * 1.02
                    cost_of_carry = ((futures_price / hist["Close"].iloc[-1]) - 1) * 100 if hist["Close"].iloc[-1] else None
                    oi = hist["Volume"].iloc[-1] * 1.5
                    oi_change = ((hist["Volume"].iloc[-1] / hist["Volume"].iloc[-5]) - 1) * 100 if len(hist) >= 5 and hist["Volume"].iloc[-5] else None
                    data[asset["name"]] = {
                        "symbol": asset["symbol"],
                        "history": hist,
                        "current_price": hist["Close"].iloc[-1],
                        "pe_ratio": ticker.info.get("trailingPE", None),
                        "peg_ratio": ticker.info.get("pegRatio", None),
                        "revenue_growth": ticker.info.get("revenueGrowth", None),
                        "roe": ticker.info.get("returnOnEquity", None),
                        "debt_to_equity": ticker.info.get("debtToEquity", None),
                        "eps_growth": ticker.info.get("earningsGrowth", None),
                        "profit_margin": ticker.info.get("grossMargins", None),
                        "dividend_yield": ticker.info.get("dividendYield", None),
                        "1m_return": ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100 if len(hist) > 1 else None,
                        "volatility": hist["Close"].pct_change().std() * (252 ** 0.5) * 100 if len(hist) > 1 else None,
                        "sma50": sma50,
                        "sma200": sma200,
                        "rsi": rsi,
                        "macd": macd,
                        "bb_high": bb_high,
                        "bb_low": bb_low,
                        "adx": adx,
                        "momentum": momentum,
                        "slowk": slowk,
                        "vol_5d": vol_5d,
                        "vol_50d": vol_50d,
                        "oi": oi,
                        "oi_change": oi_change,
                        "cost_of_carry": cost_of_carry
                    }
                    logging.info(f"Successfully fetched data for {asset['name']}")
                else:
                    failed_tickers.append(asset["name"])
                    logging.warning(f"Insufficient data for {asset['name']}: {len(hist)} rows")
                time.sleep(delay)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                failed_tickers.append(asset["name"])
                logging.error(f"Failed to fetch data for {asset['name']}: {e}")
                st.warning(f"Failed to fetch data for {asset['name']}: {e}")
    if failed_tickers:
        st.warning(f"Data fetch failed for {len(failed_tickers)} stocks: {', '.join(failed_tickers)}")
        logging.warning(f"Data fetch failed for {len(failed_tickers)} stocks: {', '.join(failed_tickers)}")
    return data

# Score stocks for buy/sell recommendations
def score_stock(data, action="buy"):
    scores = {}
    for name, info in data.items():
        score = 0
        if action == "buy":
            if info["rsi"] and info["rsi"] < 30: score += 2
            if info["sma50"] and info["sma200"] and info["sma50"] > info["sma200"]: score += 2
            if info["macd"] and info["macd"] > 0: score += 2
            if info["current_price"] and info["bb_low"] and info["current_price"] < info["bb_low"]: score += 2
            if info["adx"] and info["adx"] > 25 and info["1m_return"] > 0: score += 2
            if info["momentum"] and info["momentum"] > 0: score += 1
            if info["slowk"] and info["slowk"] < 20: score += 1
            if info["vol_5d"] and info["vol_50d"] and info["vol_5d"] > 1.5 * info["vol_50d"]: score += 1
            if info["pe_ratio"] and info["pe_ratio"] < 20: score += 1
            if info["peg_ratio"] and info["peg_ratio"] < 1: score += 1
            if info["revenue_growth"] and info["revenue_growth"] > 0.1: score += 1
            if info["roe"] and info["roe"] > 0.15: score += 1
            if info["debt_to_equity"] and info["debt_to_equity"] < 50: score += 1
            if info["eps_growth"] and info["eps_growth"] > 0.1: score += 1
            if info["profit_margin"] and info["profit_margin"] > 0.2: score += 1
            if info["dividend_yield"] and info["dividend_yield"] > 0.01: score += 1
            if info["oi_change"] and info["oi_change"] > 10: score += 1
        else:  # Sell
            if info["rsi"] and info["rsi"] > 70: score += 2
            if info["sma50"] and info["sma200"] and info["sma50"] < info["sma200"]: score += 2
            if info["macd"] and info["macd"] < 0: score += 2
            if info["current_price"] and info["bb_high"] and info["current_price"] > info["bb_high"]: score += 2
            if info["adx"] and info["adx"] > 25 and info["1m_return"] < 0: score += 2
            if info["momentum"] and info["momentum"] < 0: score += 1
            if info["slowk"] and info["slowk"] > 80: score += 1
            if info["vol_5d"] and info["vol_50d"] and info["vol_5d"] < 0.5 * info["vol_50d"]: score += 1
            if info["pe_ratio"] and info["pe_ratio"] > 30: score += 1
            if info["peg_ratio"] and info["peg_ratio"] > 2: score += 1
            if info["revenue_growth"] and info["revenue_growth"] < 0: score += 1
            if info["roe"] and info["roe"] < 0.05: score += 1
            if info["eps_growth"] and info["eps_growth"] < 0: score += 1
            if info["profit_margin"] and info["profit_margin"] < 0.05: score += 1
            if info["dividend_yield"] and info["dividend_yield"] < 0.005: score += 1
            if info["oi_change"] and info["oi_change"] < -10: score += 1
        scores[name] = score
    return scores

# Update portfolio for buys and sells
def update_portfolio(portfolio, asset_name, amount, price, date, action="buy"):
    if asset_name not in portfolio:
        portfolio[asset_name] = {"shares": 0, "total_invested": 0, "dates": [], "prices": [], "realized_pnl": 0}
    if action == "buy":
        portfolio[asset_name]["shares"] += amount / price
        portfolio[asset_name]["total_invested"] += amount
        portfolio[asset_name]["dates"].append(date)
        portfolio[asset_name]["prices"].append(price)
    else:  # Sell
        if portfolio[asset_name]["shares"] >= amount / price:
            portfolio[asset_name]["shares"] -= amount / price
            avg_buy_price = sum([p for p in portfolio[asset_name]["prices"] if p > 0]) / len([p for p in portfolio[asset_name]["prices"] if p > 0]) if any(p > 0 for p in portfolio[asset_name]["prices"]) else price
            portfolio[asset_name]["realized_pnl"] += (price - avg_buy_price) * (amount / price)
            portfolio[asset_name]["dates"].append(date)
            portfolio[asset_name]["prices"].append(-price)
    return portfolio

# Nightly analysis function
def nightly_analysis():
    logging.info("Starting nightly analysis")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    daily_data = fetch_data(FNO_ASSETS, start_date, end_date)
    if not daily_data:
        st.error("No data fetched for any stocks. Check internet connection or ticker validity (e.g., ETERNAL.NS).")
        logging.error("No data fetched for any stocks")
        return
    df = pd.DataFrame({
        "Asset": [name for name in daily_data],
        "Symbol": [daily_data[name]["symbol"] for name in daily_data],
        "Current Price (₹)": [round(daily_data[name]["current_price"], 2) if daily_data[name]["current_price"] else "N/A" for name in daily_data],
        "1-Month Return (%)": [round(daily_data[name]["1m_return"], 2) if daily_data[name]["1m_return"] else "N/A" for name in daily_data],
        "RSI": [round(daily_data[name]["rsi"], 2) if daily_data[name]["rsi"] else "N/A" for name in daily_data],
        "MACD": [round(daily_data[name]["macd"], 2) if daily_data[name]["macd"] else "N/A" for name in daily_data],
        "ADX": [round(daily_data[name]["adx"], 2) if daily_data[name]["adx"] else "N/A" for name in daily_data],
        "Momentum": [round(daily_data[name]["momentum"], 2) if daily_data[name]["momentum"] else "N/A" for name in daily_data],
        "Stochastic %K": [round(daily_data[name]["slowk"], 2) if daily_data[name]["slowk"] else "N/A" for name in daily_data],
        "P/E Ratio": [round(daily_data[name]["pe_ratio"], 2) if daily_data[name]["pe_ratio"] else "N/A" for name in daily_data],
        "PEG Ratio": [round(daily_data[name]["peg_ratio"], 2) if daily_data[name]["peg_ratio"] else "N/A" for name in daily_data],
        "ROE (%)": [round(daily_data[name]["roe"] * 100, 2) if daily_data[name]["roe"] else "N/A" for name in daily_data],
        "EPS Growth (%)": [round(daily_data[name]["eps_growth"] * 100, 2) if daily_data[name]["eps_growth"] else "N/A" for name in daily_data],
        "Profit Margin (%)": [round(daily_data[name]["profit_margin"] * 100, 2) if daily_data[name]["profit_margin"] else "N/A" for name in daily_data],
        "Dividend Yield (%)": [round(daily_data[name]["dividend_yield"] * 100, 2) if daily_data[name]["dividend_yield"] else "N/A" for name in daily_data],
        "OI Change (%)": [round(daily_data[name]["oi_change"], 2) if daily_data[name]["oi_change"] else "N/A" for name in daily_data],
        "Analysis Date": datetime.today().strftime("%Y-%m-%d")
    })
    df.to_csv("daily_analysis.csv", index=False)
    logging.info("Saved daily_analysis.csv with {} stocks".format(len(daily_data)))
    
    # Weekly recommendations (generated on Friday)
    if datetime.today().weekday() == 4:  # Friday
        buy_scores = score_stock(daily_data, action="buy")
        sell_scores = score_stock(daily_data, action="sell")
        buy_stock = max(buy_scores, key=buy_scores.get, default=None) if buy_scores else None
        sell_stock = max(sell_scores, key=sell_scores.get, default=None) if sell_scores else None
        recommendation = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "buy": {
                "stock": buy_stock,
                "score": buy_scores.get(buy_stock, 0) if buy_stock else 0,
                "details": {
                    "price": daily_data[buy_stock]["current_price"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "rsi": daily_data[buy_stock]["rsi"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "macd": daily_data[buy_stock]["macd"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "adx": daily_data[buy_stock]["adx"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "momentum": daily_data[buy_stock]["momentum"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "slowk": daily_data[buy_stock]["slowk"] if buy_stock and buy_scores[buy_stock] > 4 else None,
                    "pe_ratio": daily_data[buy_stock]["pe_ratio"] if buy_stock and buy_scores[buy_stock] > 4 else None
                }
            },
            "sell": {
                "stock": sell_stock,
                "score": sell_scores.get(sell_stock, 0) if sell_stock else 0,
                "details": {
                    "price": daily_data[sell_stock]["current_price"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "rsi": daily_data[sell_stock]["rsi"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "macd": daily_data[sell_stock]["macd"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "adx": daily_data[sell_stock]["adx"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "momentum": daily_data[sell_stock]["momentum"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "slowk": daily_data[sell_stock]["slowk"] if sell_stock and sell_scores[sell_stock] > 4 else None,
                    "pe_ratio": daily_data[sell_stock]["pe_ratio"] if sell_stock and sell_scores[sell_stock] > 4 else None
                }
            }
        }
        with open("weekly_recommendations.json", "w") as f:
            json.dump(recommendation, f)
        logging.info("Generated weekly_recommendations.json")

# Schedule nightly analysis
try:
    schedule.every().day.at("02:00").do(nightly_analysis)
    logging.info("Scheduled nightly analysis at 2 AM IST")
except Exception as e:
    st.warning(f"Scheduling failed: {e}. Use the 'Run Manual Analysis' button below.")
    logging.error(f"Scheduling failed: {e}")

# Run scheduler in background
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Streamlit App
st.title("F&O Stock Tracker with Buy/Sell Recommendations")
st.write("Daily analysis of 50 F&O stocks at 2 AM IST. Weekly buy/sell recommendations (generated Fridays) displayed below with the latest analysis date. Check daily for updates.")

# Manual analysis button
if st.button("Run Manual Analysis"):
    with st.spinner("Running manual analysis for 50 stocks..."):
        nightly_analysis()
        st.success("Manual analysis completed. Check the results below.")

# Initialize session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}
if "weekly_data" not in st.session_state:
    st.session_state.weekly_data = {}

# Sidebar for user inputs
st.sidebar.header("Investment Settings")
investment_amount = st.sidebar.number_input("Weekly Investment (₹)", min_value=0, max_value=5000, value=1250)

# Sold stock input
st.sidebar.subheader("Log Sold Stocks")
sold_ticker = st.sidebar.selectbox("Select Sold Stock", [asset["name"] for asset in FNO_ASSETS])
sold_amount = st.sidebar.number_input("Sold Amount (₹)", min_value=0.0, value=0.0)
sold_price = st.sidebar.number_input("Sold Price per Share (₹)", min_value=0.0, value=0.0)
if st.sidebar.button("Log Sold Stock"):
    if sold_ticker and sold_amount > 0 and sold_price > 0:
        st.session_state.portfolio = update_portfolio(
            st.session_state.portfolio, sold_ticker, sold_amount, sold_price, datetime.today(), action="sell"
        )
        st.success(f"Logged sale of {sold_ticker} for ₹{sold_amount} at ₹{sold_price}/share.")

# Load daily analysis from CSV if available
daily_data = {}
if os.path.exists("daily_analysis.csv"):
    try:
        df = pd.read_csv("daily_analysis.csv")
        if df.empty or df.shape[0] == 0 or df["Asset"].isna().all():
            st.warning("The daily_analysis.csv file is empty or invalid. Running fresh analysis...")
            logging.warning("daily_analysis.csv is empty or invalid, running fresh analysis")
            with st.spinner("Fetching market data for 50 stocks..."):
                end_date = datetime.today()
                start_date = end_date - timedelta(days=30)
                daily_data = fetch_data(FNO_ASSETS, start_date, end_date)
                if not daily_data:
                    st.error("No data fetched. Check internet connection or ticker validity (e.g., ETERNAL.NS).")
                    logging.error("No data fetched during fresh analysis")
                else:
                    df = pd.DataFrame({
                        "Asset": [name for name in daily_data],
                        "Symbol": [daily_data[name]["symbol"] for name in daily_data],
                        "Current Price (₹)": [round(daily_data[name]["current_price"], 2) if daily_data[name]["current_price"] else "N/A" for name in daily_data],
                        "1-Month Return (%)": [round(daily_data[name]["1m_return"], 2) if daily_data[name]["1m_return"] else "N/A" for name in daily_data],
                        "RSI": [round(daily_data[name]["rsi"], 2) if daily_data[name]["rsi"] else "N/A" for name in daily_data],
                        "MACD": [round(daily_data[name]["macd"], 2) if daily_data[name]["macd"] else "N/A" for name in daily_data],
                        "ADX": [round(daily_data[name]["adx"], 2) if daily_data[name]["adx"] else "N/A" for name in daily_data],
                        "Momentum": [round(daily_data[name]["momentum"], 2) if daily_data[name]["momentum"] else "N/A" for name in daily_data],
                        "Stochastic %K": [round(daily_data[name]["slowk"], 2) if daily_data[name]["slowk"] else "N/A" for name in daily_data],
                        "P/E Ratio": [round(daily_data[name]["pe_ratio"], 2) if daily_data[name]["pe_ratio"] else "N/A" for name in daily_data],
                        "PEG Ratio": [round(daily_data[name]["peg_ratio"], 2) if daily_data[name]["peg_ratio"] else "N/A" for name in daily_data],
                        "ROE (%)": [round(daily_data[name]["roe"] * 100, 2) if daily_data[name]["roe"] else "N/A" for name in daily_data],
                        "EPS Growth (%)": [round(daily_data[name]["eps_growth"] * 100, 2) if daily_data[name]["eps_growth"] else "N/A" for name in daily_data],
                        "Profit Margin (%)": [round(daily_data[name]["profit_margin"] * 100, 2) if daily_data[name]["profit_margin"] else "N/A" for name in daily_data],
                        "Dividend Yield (%)": [round(daily_data[name]["dividend_yield"] * 100, 2) if daily_data[name]["dividend_yield"] else "N/A" for name in daily_data],
                        "OI Change (%)": [round(daily_data[name]["oi_change"], 2) if daily_data[name]["oi_change"] else "N/A" for name in daily_data],
                        "Analysis Date": datetime.today().strftime("%Y-%m-%d")
                    })
                    df.to_csv("daily_analysis.csv", index=False)
                    logging.info("Generated new daily_analysis.csv with {} stocks".format(len(daily_data)))
                    st.subheader(f"Daily Analysis (All 50 F&O Stocks, Processed on {datetime.today().strftime('%Y-%m-%d')})")
                    st.dataframe(df)
                    st.session_state.weekly_data.update(daily_data)
        else:
            analysis_date = df["Analysis Date"].iloc[0] if "Analysis Date" in df.columns and not df["Analysis Date"].isna().all() and df.shape[0] > 0 else "Unknown"
            st.subheader(f"Daily Analysis (All 50 F&O Stocks, Processed on {analysis_date})")
            st.dataframe(df)
            # Refresh daily_data for consistency
            end_date = datetime.today()
            start_date = end_date - timedelta(days=30)
            daily_data = fetch_data(FNO_ASSETS, start_date, end_date)
            st.session_state.weekly_data.update(daily_data)
            logging.info("Loaded daily_analysis.csv with {} stocks, analysis date: {}".format(len(df), analysis_date))
    except Exception as e:
        st.error(f"Failed to load daily_analysis.csv: {e}. Running fresh analysis...")
        logging.error(f"Failed to load daily_analysis.csv: {e}")
        with st.spinner("Fetching market data for 50 stocks..."):
            end_date = datetime.today()
            start_date = end_date - timedelta(days=30)
            daily_data = fetch_data(FNO_ASSETS, start_date, end_date)
            if not daily_data:
                st.error("No data fetched. Check internet connection or ticker validity (e.g., ETERNAL.NS).")
                logging.error("No data fetched during fresh analysis")
            else:
                df = pd.DataFrame({
                    "Asset": [name for name in daily_data],
                    "Symbol": [daily_data[name]["symbol"] for name in daily_data],
                    "Current Price (₹)": [round(daily_data[name]["current_price"], 2) if daily_data[name]["current_price"] else "N/A" for name in daily_data],
                    "1-Month Return (%)": [round(daily_data[name]["1m_return"], 2) if daily_data[name]["1m_return"] else "N/A" for name in daily_data],
                    "RSI": [round(daily_data[name]["rsi"], 2) if daily_data[name]["rsi"] else "N/A" for name in daily_data],
                    "MACD": [round(daily_data[name]["macd"], 2) if daily_data[name]["macd"] else "N/A" for name in daily_data],
                    "ADX": [round(daily_data[name]["adx"], 2) if daily_data[name]["adx"] else "N/A" for name in daily_data],
                    "Momentum": [round(daily_data[name]["momentum"], 2) if daily_data[name]["momentum"] else "N/A" for name in daily_data],
                    "Stochastic %K": [round(daily_data[name]["slowk"], 2) if daily_data[name]["slowk"] else "N/A" for name in daily_data],
                    "P/E Ratio": [round(daily_data[name]["pe_ratio"], 2) if daily_data[name]["pe_ratio"] else "N/A" for name in daily_data],
                    "PEG Ratio": [round(daily_data[name]["peg_ratio"], 2) if daily_data[name]["peg_ratio"] else "N/A" for name in daily_data],
                    "ROE (%)": [round(daily_data[name]["roe"] * 100, 2) if daily_data[name]["roe"] else "N/A" for name in daily_data],
                    "EPS Growth (%)": [round(daily_data[name]["eps_growth"] * 100, 2) if daily_data[name]["eps_growth"] else "N/A" for name in daily_data],
                    "Profit Margin (%)": [round(daily_data[name]["profit_margin"] * 100, 2) if daily_data[name]["profit_margin"] else "N/A" for name in daily_data],
                    "Dividend Yield (%)": [round(daily_data[name]["dividend_yield"] * 100, 2) if daily_data[name]["dividend_yield"] else "N/A" for name in daily_data],
                    "OI Change (%)": [round(daily_data[name]["oi_change"], 2) if daily_data[name]["oi_change"] else "N/A" for name in daily_data],
                    "Analysis Date": datetime.today().strftime("%Y-%m-%d")
                })
                df.to_csv("daily_analysis.csv", index=False)
                logging.info("Generated new daily_analysis.csv with {} stocks".format(len(daily_data)))
                st.subheader(f"Daily Analysis (All 50 F&O Stocks, Processed on {datetime.today().strftime('%Y-%m-%d')})")
                st.dataframe(df)
                st.session_state.weekly_data.update(daily_data)
else:
    st.subheader("Daily Analysis (All 50 F&O Stocks)")
    with st.spinner("Fetching market data for 50 stocks..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=30)
        daily_data = fetch_data(FNO_ASSETS, start_date, end_date)
        if not daily_data:
            st.error("No data fetched. Check internet connection or ticker validity (e.g., ETERNAL.NS).")
            logging.error("No data fetched during initial analysis")
        else:
            df = pd.DataFrame({
                "Asset": [name for name in daily_data],
                "Symbol": [daily_data[name]["symbol"] for name in daily_data],
                "Current Price (₹)": [round(daily_data[name]["current_price"], 2) if daily_data[name]["current_price"] else "N/A" for name in daily_data],
                "1-Month Return (%)": [round(daily_data[name]["1m_return"], 2) if daily_data[name]["1m_return"] else "N/A" for name in daily_data],
                "RSI": [round(daily_data[name]["rsi"], 2) if daily_data[name]["rsi"] else "N/A" for name in daily_data],
                "MACD": [round(daily_data[name]["macd"], 2) if daily_data[name]["macd"] else "N/A" for name in daily_data],
                "ADX": [round(daily_data[name]["adx"], 2) if daily_data[name]["adx"] else "N/A" for name in daily_data],
                "Momentum": [round(daily_data[name]["momentum"], 2) if daily_data[name]["momentum"] else "N/A" for name in daily_data],
                "Stochastic %K": [round(daily_data[name]["slowk"], 2) if daily_data[name]["slowk"] else "N/A" for name in daily_data],
                "P/E Ratio": [round(daily_data[name]["pe_ratio"], 2) if daily_data[name]["pe_ratio"] else "N/A" for name in daily_data],
                "PEG Ratio": [round(daily_data[name]["peg_ratio"], 2) if daily_data[name]["peg_ratio"] else "N/A" for name in daily_data],
                "ROE (%)": [round(daily_data[name]["roe"] * 100, 2) if daily_data[name]["roe"] else "N/A" for name in daily_data],
                "EPS Growth (%)": [round(daily_data[name]["eps_growth"] * 100, 2) if daily_data[name]["eps_growth"] else "N/A" for name in daily_data],
                "Profit Margin (%)": [round(daily_data[name]["profit_margin"] * 100, 2) if daily_data[name]["profit_margin"] else "N/A" for name in daily_data],
                "Dividend Yield (%)": [round(daily_data[name]["dividend_yield"] * 100, 2) if daily_data[name]["dividend_yield"] else "N/A" for name in daily_data],
                "OI Change (%)": [round(daily_data[name]["oi_change"], 2) if daily_data[name]["oi_change"] else "N/A" for name in daily_data],
                "Analysis Date": datetime.today().strftime("%Y-%m-%d")
            })
            df.to_csv("daily_analysis.csv", index=False)
            logging.info("Generated initial daily_analysis.csv with {} stocks".format(len(daily_data)))
            st.subheader(f"Daily Analysis (All 50 F&O Stocks, Processed on {datetime.today().strftime('%Y-%m-%d')})")
            st.dataframe(df)
            st.session_state.weekly_data.update(daily_data)

# Display weekly recommendations (generated on Fridays)
st.subheader("Weekly Buy/Sell Recommendations (Generated Fridays)")
if os.path.exists("weekly_recommendations.json"):
    try:
        with open("weekly_recommendations.json", "r") as f:
            recommendation = json.load(f)
            st.write(f"**Recommendations for {recommendation['date']}**")
            if recommendation["buy"]["stock"] and recommendation["buy"]["score"] > 4:
                st.write(f"**Buy Recommendation**: {recommendation['buy']['stock']} (Score: {recommendation['buy']['score']}/17)")
                st.write(f"Price: ₹{recommendation['buy']['details']['price']:.2f}, RSI: {recommendation['buy']['details']['rsi']:.2f}, MACD: {recommendation['buy']['details']['macd']:.2f}, ADX: {recommendation['buy']['details']['adx']:.2f}, Momentum: {recommendation['buy']['details']['momentum']:.2f}, Stochastic %K: {recommendation['buy']['details']['slowk']:.2f}, P/E: {recommendation['buy']['details']['pe_ratio']:.2f}")
            else:
                st.write("No strong buy recommendation this week.")
            if recommendation["sell"]["stock"] and recommendation["sell"]["score"] > 4:
                st.write(f"**Sell Recommendation**: {recommendation['sell']['stock']} (Score: {recommendation['sell']['score']}/16)")
                st.write(f"Price: ₹{recommendation['sell']['details']['price']:.2f}, RSI: {recommendation['sell']['details']['rsi']:.2f}, MACD: {recommendation['sell']['details']['macd']:.2f}, ADX: {recommendation['sell']['details']['adx']:.2f}, Momentum: {recommendation['sell']['details']['momentum']:.2f}, Stochastic %K: {recommendation['sell']['details']['slowk']:.2f}, P/E: {recommendation['sell']['details']['pe_ratio']:.2f}")
            else:
                st.write("No strong sell recommendation this week.")
            st.write("Invest ₹1250 in buy recommendations via SIP.")
            logging.info("Displayed weekly recommendations for {}".format(recommendation['date']))
    except Exception as e:
        st.warning(f"Failed to load weekly_recommendations.json: {e}. Run manual analysis to generate new recommendations.")
        logging.error(f"Failed to load weekly_recommendations.json: {e}")
else:
    st.write("No weekly recommendations available. Check back on Friday or run manual analysis.")
    logging.info("No weekly_recommendations.json found")

# Plot price trends (sample of 10 stocks)
st.subheader("Price Trends (Sample of 10 Stocks, Last 30 Days)")
sample_tickers = random.sample(FNO_ASSETS, min(10, len(FNO_ASSETS)))
sample_data = fetch_data(sample_tickers, start_date, end_date)
fig, ax = plt.subplots()
for name in sample_data:
    if not sample_data[name]["history"].empty:
        ax.plot(sample_data[name]["history"].index, sample_data[name]["history"]["Close"], label=name)
ax.set_xlabel("Date")
ax.set_ylabel("Price (₹)")
ax.legend()
st.pyplot(fig)
logging.info("Displayed price trends for {} stocks".format(len(sample_data)))

# Real-time buy/sell recommendations
st.subheader("Real-Time Buy/Sell Recommendations")
selected_assets = st.multiselect("Select Stocks for Real-Time Analysis", [asset["name"] for asset in FNO_ASSETS], default=[asset["name"] for asset in FNO_ASSETS[:3]])
if selected_assets:
    with st.spinner("Fetching real-time data..."):
        real_time_data = fetch_data([asset for asset in FNO_ASSETS if asset["name"] in selected_assets], start_date, end_date)
        buy_scores = score_stock(real_time_data, action="buy")
        sell_scores = score_stock(real_time_data, action="sell")
        buy_recommendation = max(buy_scores, key=buy_scores.get, default=None) if buy_scores else None
        sell_recommendation = max(sell_scores, key=sell_scores.get, default=None) if sell_scores else None
        if buy_recommendation and buy_scores[buy_recommendation] > 4:
            st.write(f"**Buy Recommendation**: {buy_recommendation} (Score: {buy_scores[buy_recommendation]}/17)")
            st.write(f"Price: ₹{real_time_data[buy_recommendation]['current_price']:.2f}, RSI: {real_time_data[buy_recommendation]['rsi']:.2f}, MACD: {real_time_data[buy_recommendation]['macd']:.2f}, ADX: {real_time_data[buy_recommendation]['adx']:.2f}, Momentum: {real_time_data[buy_recommendation]['momentum']:.2f}, Stochastic %K: {real_time_data[buy_recommendation]['slowk']:.2f}, P/E: {real_time_data[buy_recommendation]['pe_ratio']:.2f}")
        else:
            st.write("No strong buy recommendation at this time.")
        if sell_recommendation and sell_scores[sell_recommendation] > 4:
            st.write(f"**Sell Recommendation**: {sell_recommendation} (Score: {sell_scores[sell_recommendation]}/16)")
            st.write(f"Price: ₹{real_time_data[sell_recommendation]['current_price']:.2f}, RSI: {real_time_data[sell_recommendation]['rsi']:.2f}, MACD: {real_time_data[sell_recommendation]['macd']:.2f}, ADX: {real_time_data[sell_recommendation]['adx']:.2f}, Momentum: {real_time_data[sell_recommendation]['momentum']:.2f}, Stochastic %K: {real_time_data[sell_recommendation]['slowk']:.2f}, P/E: {real_time_data[sell_recommendation]['pe_ratio']:.2f}")
        else:
            st.write("No strong sell recommendation at this time.")
        logging.info("Generated real-time recommendations for {} stocks".format(len(selected_assets)))

# Portfolio management
st.subheader("Your Portfolio")
if st.button("Add Investment"):
    portfolio_data = fetch_data([asset for asset in FNO_ASSETS if asset["name"] in selected_assets], start_date, end_date)
    for name in selected_assets:
        if name in portfolio_data and portfolio_data[name]["current_price"]:
            st.session_state.portfolio = update_portfolio(
                st.session_state.portfolio, name, investment_amount, portfolio_data[name]["current_price"], end_date, action="buy"
            )
            st.success(f"Added ₹{investment_amount} investment in {name}.")
            logging.info(f"Added ₹{investment_amount} investment in {name}")

if st.session_state.portfolio:
    portfolio_df = pd.DataFrame([
        {
            "Asset": name,
            "Shares": round(info["shares"], 4),
            "Total Invested (₹)": round(info["total_invested"], 2),
            "Realized P&L (₹)": round(info["realized_pnl"], 2),
            "Current Value (₹)": round(info["shares"] * fetch_data([{"name": name, "symbol": next(a["symbol"] for a in FNO_ASSETS if a["name"] == name)}], start_date, end_date)[name]["current_price"], 2) if name in fetch_data([{"name": name, "symbol": next(a["symbol"] for a in FNO_ASSETS if a["name"] == name)}], start_date, end_date) else "N/A"
        }
        for name, info in st.session_state.portfolio.items()
    ])
    st.dataframe(portfolio_df)
    logging.info("Displayed portfolio with {} assets".format(len(st.session_state.portfolio)))

# Download data
if 'df' in locals() and not df.empty:
    csv = df.to_csv(index=False)
    st.download_button("Download Daily Analysis", csv, "daily_analysis.csv")
    logging.info("Provided download option for daily_analysis.csv")
else:
    st.warning("No daily analysis data available to download. Run manual analysis to generate data.")
    logging.warning("No daily analysis data available for download")

st.write("**Note**: Analysis runs nightly at 2 AM IST for all 50 F&O stocks. Weekly recommendations are generated Fridays and displayed here with the analysis date. Use the 'Run Manual Analysis' button if data is missing or scheduling fails.")
logging.info("Streamlit app rendered successfully")