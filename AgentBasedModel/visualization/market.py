from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_price(info: SimulatorInfo, spread=False, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Stock Price') if rolling == 1 else plt.title(f'Stock Price (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), color='black')
    if spread:
        v1 = [el['bid'] for el in info.spreads]
        v2 = [el['ask'] for el in info.spreads]
        plt.plot(range(rolling - 1, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
        plt.plot(range(rolling - 1, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    


def plot_price_fundamental(info: SimulatorInfo, spread=False, access: int = 1, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if rolling == 1:
        plt.title('Stock Fundamental and Market value')
    else:
        plt.title(f'Stock Fundamental and Market value (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Present value')
    if spread:
        v1 = [el['bid'] for el in info.spreads]
        v2 = [el['ask'] for el in info.spreads]
        plt.plot(range(rolling - 1, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
        plt.plot(range(rolling - 1, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), label='market value', color='black')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.fundamental_value(access), rolling),
             label='fundamental value')
    plt.legend()
    


def plot_arbitrage(info: SimulatorInfo, access: int = 1, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if rolling == 1:
        plt.title('Stock Fundamental and Market value difference %')
    else:
        plt.title(f'Stock Fundamental and Market value difference % (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Present value')
    market = info.prices
    fundamental = info.fundamental_value(access)
    arbitrage = [(fundamental[i] - market[i]) / fundamental[i] for i in range(len(market))]
    plt.plot(range(rolling - 1, len(arbitrage)), math.rolling(arbitrage, rolling), color='black')
    


def plot_dividend(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Stock Dividend') if rolling == 1 else plt.title(f'Stock Dividend (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Dividend')
    plt.plot(range(rolling - 1, len(info.dividends)), math.rolling(info.dividends, rolling), color='black')
    


def plot_orders(info: SimulatorInfo, stat: str = 'quantity', rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Book Orders') if rolling == 1 else plt.title(f'Book Orders (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel(stat)
    v1 = [v[stat]['bid'] for v in info.orders]
    v2 = [v[stat]['ask'] for v in info.orders]
    plt.plot(range(rolling - 1, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
    plt.plot(range(rolling - 1, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    plt.legend()
    


def plot_volatility_price(info: SimulatorInfo, window: int = 5, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock Price Volatility (window {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Price Volatility')
    volatility = info.price_volatility(window)
    plt.plot(range(window, len(volatility) + window), volatility, color='black')

def plot_volatility_return(info: SimulatorInfo, window: int = 5, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock Return Volatility (window {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Return Volatility')
    volatility = info.return_volatility(window)
    plt.plot(range(window, len(volatility) + window), volatility, color='black')
    


def plot_liquidity(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Liquidity') if rolling == 1 else plt.title(f'Liquidity (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Spread / avg. Price')
    plt.plot(info.liquidity(rolling), color='black')

#my
def plot_fixed_spread_sweep(df: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Market Liquidity: Fixed Volume Spread vs $J$')
    plt.xlabel('Interaction Parameter $J$')
    plt.ylabel('Mean Spread (Qty 1000)')
    
    if not df.empty and 'fixed_spread_1000' in df.columns:
        data = df.groupby('J')['fixed_spread_1000'].mean().reset_index()
        plt.plot(data['J'], data['fixed_spread_1000'], 'D-', color='green', label='Spread (1000)')
        plt.grid(True, alpha=0.3)
        plt.legend()

def plot_returns_distribution(info: SimulatorInfo, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Returns Distribution (Log-scale)')
    plt.xlabel('Return Value')
    plt.ylabel('Log Density')
    
    prices = info.prices
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    plt.hist(returns, bins=50, density=True, color='black', alpha=0.7, log=True)


def plot_abs_return_acf(info, max_lag=50, title_suffix=""):
    returns = np.diff(info.prices) / info.prices[:-1]
    if len(returns) < max_lag + 5:
        return
    r_eff = returns
    r = np.abs(r_eff - r_eff.mean())
    r = r - r.mean()
    var = np.dot(r, r)
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c = np.dot(r[:-lag], r[lag:])
            acf.append(c / var if var > 0 else 0.0)
    acf = np.array(acf)

    plt.figure(figsize=(6, 4))
    plt.stem(range(max_lag + 1), acf, use_line_collection=True)
    plt.xlabel('Lag')
    plt.ylabel('ACF(|r_t|)')
    plt.title(f'Volatility clustering: ACF(|r_t|) {title_suffix}')


def plot_vol_cluster_metrics(df_results: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    agg = df_results.groupby('J')['acf_abs_mean'].mean().reset_index()
    plt.plot(agg['J'], agg['acf_abs_mean'], marker='o')
    plt.xlabel('J')
    plt.ylabel('Mean ACF(|r_t|) (lags 1..L)')
    plt.title('Volatility clustering vs J')
    plt.grid(True)

    plt.figure(figsize=(6, 4))
    agg2 = df_results.groupby('J')['cluster_len_avg'].mean().reset_index()
    plt.plot(agg2['J'], agg2['cluster_len_avg'], marker='o', color='orange')
    plt.xlabel('J')
    plt.ylabel('Average high-vol cluster length')
    plt.title('High-volatility cluster length vs J')
    plt.grid(True)


def plot_fixed_spread_vs_J(df_results: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    agg = df_results.groupby('J')['fixed_spread_1000'].mean().reset_index()
    plt.plot(agg['J'], agg['fixed_spread_1000'], marker='o')
    plt.xlabel('J')
    plt.ylabel('Fixed-volume spread (V=1000)')
    plt.title('Fixed-volume spread vs J')
    plt.grid(True)


def plot_cluster_vs_spread(df_results: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    plt.scatter(df_results['acf_abs_mean'], df_results['fixed_spread_1000'],
                alpha=0.6)
    plt.xlabel('Mean ACF(|r_t|)')
    plt.ylabel('Fixed-volume spread (V=1000)')
    plt.title('Volatility clustering vs liquidity cost')
    plt.grid(True)
