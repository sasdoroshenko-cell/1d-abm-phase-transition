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

def plot_depth_by_mode(df_results: pd.DataFrame, figsize=(6, 6)):
    """Средняя глубина стакана (orders_count) по J для baseline vs panic."""
    plt.figure(figsize=figsize)
    plt.title('Order book depth vs J')
    plt.xlabel('J')
    plt.ylabel('Average number of orders')

    agg = df_results.groupby(['J', 'panic_mode'])['orders_count'].mean().reset_index()

    for mode, style, label in [(0, 'o-', 'Baseline chartists'),
                               (1, 's--', 'Multi-state / panic chartists')]:
        tmp = agg[agg['panic_mode'] == mode]
        if len(tmp) == 0:
            continue
        plt.plot(tmp['J'], tmp['orders_count'], style, label=label)

    plt.legend()


def plot_returns_distribution(info: SimulatorInfo, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Returns Distribution (Log-scale)')
    plt.xlabel('Return Value')
    plt.ylabel('Log Density')
    
    prices = info.prices
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    plt.hist(returns, bins=50, density=True, color='black', alpha=0.7, log=True)


def plot_spread_by_mode(df_results: pd.DataFrame, figsize=(6, 6)):
    """Fixed-volume spread(J) для baseline vs panic."""
    plt.figure(figsize=figsize)
    plt.title('Fixed-volume spread vs J (baseline vs panic)')
    plt.xlabel('J')
    plt.ylabel('Spread for volume=1000')

    agg = df_results.groupby(['J', 'panic_mode'])['fixed_spread_1000'].mean().reset_index()

    for mode, style, label in [(0, 'o-', 'Baseline chartists'),
                               (1, 's--', 'Multi-state / panic chartists')]:
        tmp = agg[agg['panic_mode'] == mode]
        if len(tmp) == 0:
            continue
        plt.plot(tmp['J'], tmp['fixed_spread_1000'], style, label=label)

    plt.legend()

def plot_skewness_by_mode(df_results: pd.DataFrame, figsize=(6, 6)):
    """Skewness(J) для baseline vs panic."""
    plt.figure(figsize=figsize)
    plt.title('Return skewness vs J')
    plt.xlabel('J')
    plt.ylabel('Skewness')

    agg = df_results.groupby(['J', 'panic_mode'])['skew'].mean().reset_index()

    for mode, style, label in [(0, 'o-', 'Baseline chartists'),
                               (1, 's--', 'Multi-state / panic chartists')]:
        tmp = agg[agg['panic_mode'] == mode]
        if len(tmp) == 0:
            continue
        plt.plot(tmp['J'], tmp['skew'], style, label=label)

    plt.axhline(0.0, color='gray', linestyle=':')
    plt.legend()


def plot_gain_loss_asymmetry(df_results: pd.DataFrame, J_focus=10.0, figsize=(6, 6)):
    """Scatter: max run-up vs |max drawdown| для baseline vs panic при фиксированном J."""
    plt.figure(figsize=figsize)
    plt.title(f'Gain–Loss Asymmetry at J={J_focus}')
    plt.xlabel('Max run-up')
    plt.ylabel('Max drawdown (absolute value)')

    sub = df_results[df_results['J'] == J_focus]

    for mode, marker, label in [(0, 'o', 'Baseline chartists'),
                                (1, 'x', 'Multi-state / panic chartists')]:
        tmp = sub[sub['panic_mode'] == mode]
        if len(tmp) == 0:
            continue
        plt.scatter(tmp['max_runup'], -tmp['max_drawdown'],
                    marker=marker, label=label, alpha=0.7)

    plt.legend()

#DIFF AGENTS
def plot_skew_by_chartist_and_mode(df_results: pd.DataFrame, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.title('Skewness vs J for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('Skewness')

    agg = (
        df_results
        .groupby(['J', 'Chartist', 'panic_mode'])['skew']
        .mean()
        .reset_index()
    )

    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        for mode, style, label_suffix in [(0, '-', 'baseline'),
                                          (1, '--', 'panic')]:
            sub = agg[(agg['Chartist'] == ch) & (agg['panic_mode'] == mode)]
            if len(sub) == 0:
                continue
            label = f'Chartist={ch}, {label_suffix}'
            plt.plot(sub['J'], sub['skew'], style, marker='o', label=label)

    plt.axhline(0.0, color='gray', linestyle=':')
    plt.legend()

def plot_gain_loss_by_chartist(df_results: pd.DataFrame, J_focus=10.0, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.title(f'Gain–Loss Asymmetry at J={J_focus} for different Chartist counts')
    plt.xlabel('Max run-up')
    plt.ylabel('Max drawdown (absolute value)')

    sub = df_results[df_results['J'] == J_focus]

    chart_levels = sorted(sub['Chartist'].unique())

    for ch in chart_levels:
        for mode, marker, label_suffix in [(0, 'o', 'baseline'),
                                           (1, 'x', 'panic')]:
            tmp = sub[(sub['Chartist'] == ch) & (sub['panic_mode'] == mode)]
            if len(tmp) == 0:
                continue
            plt.scatter(
                tmp['max_runup'],
                -tmp['max_drawdown'],
                marker=marker,
                alpha=0.6,
                label=f'Chartist={ch}, {label_suffix}'
            )

    plt.legend()


def plot_spread_by_chartist_and_mode(df_results: pd.DataFrame, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.title('Fixed-volume spread vs J for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('Spread for volume=1000')

    agg = (
        df_results
        .groupby(['J', 'Chartist', 'panic_mode'])['fixed_spread_1000']
        .mean()
        .reset_index()
    )

    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        for mode, style, label_suffix in [(0, '-', 'baseline'),
                                          (1, '--', 'panic')]:
            sub = agg[(agg['Chartist'] == ch) & (agg['panic_mode'] == mode)]
            if len(sub) == 0:
                continue
            label = f'Chartist={ch}, {label_suffix}'
            plt.plot(sub['J'], sub['fixed_spread_1000'],
                     style, marker='o', label=label)

    plt.legend()

def plot_depth_by_chartist_and_mode(df_results: pd.DataFrame, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.title('Order book depth vs J for different Chartist counts')
    plt.xlabel('J')
    plt.ylabel('Average number of orders')

    agg = (
        df_results
        .groupby(['J', 'Chartist', 'panic_mode'])['orders_count']
        .mean()
        .reset_index()
    )

    chart_levels = sorted(agg['Chartist'].unique())

    for ch in chart_levels:
        for mode, style, label_suffix in [(0, '-', 'baseline'),
                                          (1, '--', 'panic')]:
            sub = agg[(agg['Chartist'] == ch) & (agg['panic_mode'] == mode)]
            if len(sub) == 0:
                continue
            label = f'Chartist={ch}, {label_suffix}'
            plt.plot(sub['J'], sub['orders_count'],
                     style, marker='o', label=label)

    plt.legend()
