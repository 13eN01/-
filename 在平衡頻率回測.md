# -
Using different balance frequency to observe how it affects the risk 
import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------
# 設定股票與回測時間
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]  # 科技五大
start_date = "2019-01-01"
end_date = "2024-12-31"

# 下載股價
data = yf.download(tickers, start=start_date, end=end_date)["Close"]

# 計算日對數報酬率
returns = np.log(data / data.shift(1)).dropna()


# -----------------------
# 回測函數
def backtest_with_cost(returns, rebalance_freq="ME", cost=0.01, multiplier=3, noise_pct=0.005):
    # 初始化投資組合
    port_values = pd.Series(index=returns.index, dtype=float)
    n_assets = returns.shape[1]
    current_weights = np.array([1 / n_assets] * n_assets)

    # 取得再平衡日期
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for i, date in enumerate(returns.index):
        if i == 0:
            port_values.iloc[0] = 1.0  # 初始資產
            continue

        # 判斷是否再平衡
        if date in rebalance_dates:
            # 加入隨機噪音 ±noise_pct
            noise = np.random.uniform(-noise_pct, noise_pct, n_assets)
            new_weights = current_weights + noise
            new_weights = np.maximum(new_weights, 1e-6)  # 避免全為0
            new_weights /= new_weights.sum()

            # 單次扣除交易成本
            turnover = np.abs(new_weights - current_weights).sum()
            port_values.iloc[i - 1] *= (1 - cost * turnover)

            current_weights = new_weights

        # 計算當日投組價值變化
        port_values.iloc[i] = port_values.iloc[i - 1] * np.exp(
            (returns.loc[date].values * current_weights * multiplier).sum())

    # 計算績效指標
    total_days = len(port_values)
    ann_return = (np.log(port_values.iloc[-1] / port_values.iloc[0]) / total_days) * 252
    ann_vol = returns.dot(current_weights * multiplier).std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    max_drawdown = (port_values / port_values.cummax() - 1).min()

    metrics = {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

    return port_values, metrics, rebalance_dates


# -----------------------
# 測試不同再平衡頻率
for freq, name in [("ME", "M"), ("QE-DEC", "Q"), ("YE-DEC", "Y")]:
    port_series, metrics, rebalance_dates = backtest_with_cost(returns, rebalance_freq=freq)
    print(f"[{name}] 再平衡共 {len(rebalance_dates)} 天")
    print(rebalance_dates)
    print(f"策略：{name} 再平衡")
    print(f"  年化報酬率: {metrics['Annual Return'] * 100:.2f}%")
    print(f"  年化波動率: {metrics['Annual Volatility'] * 100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"  最大回撤: {metrics['Max Drawdown'] * 100:.2f}%")
    print("-" * 40)
