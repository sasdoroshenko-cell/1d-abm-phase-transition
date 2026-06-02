from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math
import matplotlib.pyplot as plt
import pandas as pd


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

def plot_sentiment_imbalance(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Sentiment Imbalance') if rolling == 1 else plt.title(f'Sentiment Imbalance (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Order Parameter $m(t)$')
    
    m_t = []
    for sent_dict in info.sentiments:
        n_opt = sum(s == 'Optimistic' for s in sent_dict.values())
        n_pes = sum(s == 'Pessimistic' for s in sent_dict.values())
        n_tot = n_opt + n_pes
        m_t.append(abs(n_opt - n_pes) / n_tot if n_tot > 0 else 0)

    plt.plot(range(rolling - 1, len(m_t)), math.rolling(m_t, rolling), color='black')



def plot_phase_transition(df: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Phase Transition: Sentiment Order vs $J$')
    plt.xlabel('Interaction Parameter $J$')
    plt.ylabel('Order Parameter $m_J$')
    
    data = df.groupby('J')['m_J'].mean().reset_index()
    
    plt.plot(data['J'], data['m_J'], 'o-', color='black', label='$m(J)$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()



def plot_volatility_sweep(df: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Volatility Jump vs $J$')
    plt.xlabel('Interaction Parameter $J$')
    plt.ylabel('Mean Return Volatility $vol_J$')
    
    data = df.groupby('J')['vol_J'].mean().reset_index()

    plt.plot(data['J'], data['vol_J'], 's--', color='red', label='Volatility')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_returns_distribution(info: SimulatorInfo, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Returns Distribution (Log-scale)')
    plt.xlabel('Return Value')
    plt.ylabel('Log Density')
    
    prices = info.prices
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    plt.hist(returns, bins=50, density=True, color='black', alpha=0.7, log=True)


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

#FOR DIFF AGENTS


def plot_phase_by_chartist(df_results: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Order parameter m(J) for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('m(J)')

    agg = df_results.groupby(['J', 'Chartist'])['m_J'].mean().reset_index()
    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        sub = agg[agg['Chartist'] == ch]
        plt.plot(sub['J'], sub['m_J'], marker='o', label=f'Chartist = {ch}')

    plt.legend()

def plot_volatility_by_chartist(df_results: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Return volatility vol(J) for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('vol(J)')

    agg = df_results.groupby(['J', 'Chartist'])['vol_J'].mean().reset_index()
    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        sub = agg[agg['Chartist'] == ch]
        plt.plot(sub['J'], sub['vol_J'], marker='o', label=f'Chartist = {ch}')

    plt.legend()


def plot_spread_by_chartist(df_results: pd.DataFrame, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Fixed-volume spread for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('Spread for volume=1000')

    agg = df_results.groupby(['J', 'Chartist'])['fixed_spread_1000'].mean().reset_index()
    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        sub = agg[agg['Chartist'] == ch]
        plt.plot(sub['J'], sub['fixed_spread_1000'], marker='o', label=f'Chartist = {ch}')

    plt.legend()