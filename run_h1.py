from AgentBasedModel import *
from AgentBasedModel.utils.math import *
from AgentBasedModel.visualization.market import (
    plot_sentiment_imbalance,
    plot_phase_transition,
    plot_volatility_sweep,
    plot_returns_distribution,
    plot_fixed_spread_sweep,
    plot_phase_by_chartist,
    plot_volatility_by_chartist,
    plot_spread_by_chartist


)

import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp, fligner




J_RANGE = np.linspace(0, 50, 6)
WINDOW = 20
T_STEPS = 500
T0 = 100
TARGET_QTY = 1000
N_RUNS = 20 

COMBOS = [
    (5, 1, 1, 1),
    (1, 5, 1, 1),
    (1, 1, 5, 1),
    (2, 2, 2, 2),
    (1, 2, 5, 1),
    (1, 1, 5, 3), 
]

results = []
info_last = None

returns_J0 = []
returns_J10 = []


def get_fixed_volume_spread(exchange, target_qty=1000):
    def get_weighted_price(order_list, target):
        total_qty, total_cost = 0, 0
        for order in order_list:
            take = min(order.qty, target - total_qty)
            total_cost += take * order.price
            total_qty += take
            if total_qty >= target:
                break
        return total_cost / total_qty if total_qty >= target else None

    avg_ask = get_weighted_price(exchange.order_book['ask'], target_qty)
    avg_bid = get_weighted_price(exchange.order_book['bid'], target_qty)
    return avg_ask - avg_bid if (avg_ask is not None and avg_bid is not None) else None


def get_returns_from_info(info):
    prices = info.prices
    returns = np.diff(prices) / prices[:-1]
    return np.array(returns)


for current_J in tqdm(J_RANGE, desc="Sweep J"):
    for (n_rand, n_fund, n_chart, n_univ) in tqdm(COMBOS, leave=False, desc="Agent mixes"):
        for run in range(N_RUNS):
            is_mm = 1

            exchange = ExchangeAgent(volume=1000)
            trader_list = [
                *[Random(exchange, 10 ** 3) for _ in range(n_rand)],
                *[Fundamentalist(exchange, 10 ** 3) for _ in range(n_fund)],
                *[Chartist(exchange, 10 ** 3, J=float(current_J)) for _ in range(n_chart)],
                *[Universalist(exchange, 10 ** 3) for _ in range(n_univ)],
                *[MarketMaker(exchange, 10 ** 3) for _ in range(is_mm)]
            ]


            radius = 2
            N = len(trader_list)
            for i, tr in enumerate(trader_list):
                neigh = [trader_list[(i - d) % N] for d in range(1, radius + 1)] + \
                        [trader_list[(i + d) % N] for d in range(1, radius + 1)]
                tr.neighbors = neigh

            simulator = Simulator(**{
                'exchange': exchange,
                'traders': trader_list,
                'events': [MarketPriceShock(200, -10)]
            })

            try:
                simulator.simulate(T_STEPS, silent=True)
            except Exception:
                continue

            info = simulator.info
            info_last = info

            if current_J in [0.0, 10.0]:
                r = get_returns_from_info(info)
                if len(r) > T0:
                    r = r[T0:]
                if current_J == 0.0:
                    returns_J0.append(r)
                elif current_J == 10.0:
                    returns_J10.append(r)


            m_vals = []
            for sent_dict in info.sentiments[T0:]:
                n_opt = sum(s == 'Optimistic' for s in sent_dict.values())
                n_pes = sum(s == 'Pessimistic' for s in sent_dict.values())
                n_tot = n_opt + n_pes
                if n_tot > 0:
                    m_vals.append(abs(n_opt - n_pes) / n_tot)
            m_J = np.mean(m_vals) if m_vals else 0.0


            vol_series = info.return_volatility(WINDOW)
            vol_J = np.mean(vol_series[T0:]) if len(vol_series) > T0 else 0.0

            step_spreads = []
            step_orders_count = []

            for _ in range(T_STEPS):
                try:
                    simulator.simulate(1, silent=True)
                    s = get_fixed_volume_spread(exchange, TARGET_QTY)
                    if s is not None:
                        step_spreads.append(s)

                    n_bids = len(exchange.order_book['bid'])
                    n_asks = len(exchange.order_book['ask'])
                    step_orders_count.append(n_bids + n_asks)
                except Exception:
                    break

            avg_fixed_spread = np.mean(step_spreads[T0:]) if len(step_spreads) > T0 else 0.0
            avg_orders_num = np.mean(step_orders_count[T0:]) if len(step_orders_count) > T0 else 0.0


            results.append({
                'J': current_J,
                'Random': n_rand,
                'Fundamentalist': n_fund,
                'Chartist': n_chart,
                'Universalist': n_univ,
                'MarketMaker': is_mm,
                'run_id': run,
                'm_J': m_J,
                'vol_J': vol_J,
                'fixed_spread_1000': avg_fixed_spread,
                'orders_count': avg_orders_num
            })



df_results = pd.DataFrame(results)
file_exists = os.path.isfile("results_all.csv")
df_results.to_csv(
    "results_all.csv",
    mode="a",
    header=not file_exists,
    index=False
)


returns_J0_all = np.concatenate(returns_J0, axis=0) if returns_J0 else np.array([])
returns_J10_all = np.concatenate(returns_J10, axis=0) if returns_J10 else np.array([])

if len(returns_J0_all) > 0 and len(returns_J10_all) > 0:
    print("Nonparametric tests between J=0 and J=10 regimes")
    print("Sample sizes:", len(returns_J0_all), len(returns_J10_all))

    ks_stat, ks_p = ks_2samp(returns_J0_all, returns_J10_all)
    print(f"KS test: statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")

    fl_stat, fl_p = fligner(returns_J0_all, returns_J10_all)
    print(f"Fligner–Killeen test: statistic = {fl_stat:.4f}, p-value = {fl_p:.4e}")
else:
    print("Not enough return data for J=0 or J=10 to run nonparametric tests.")

if info_last is not None:
     plot_sentiment_imbalance(info_last, rolling=10, figsize=(6, 6))
     plot_returns_distribution(info_last, figsize=(6, 6))

plot_volatility_sweep(df_results)
plot_phase_transition(df_results)
plot_fixed_spread_sweep(df_results)

plot_phase_by_chartist(df_results)
plot_volatility_by_chartist(df_results)
plot_spread_by_chartist(df_results)


plt.show()


