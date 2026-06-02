from AgentBasedModel import *
from AgentBasedModel.utils.math import *
from AgentBasedModel.visualization.market import (
    plot_returns_distribution,
    plot_fixed_spread_sweep,
    plot_abs_return_acf,
    plot_vol_cluster_metrics,
    plot_fixed_spread_vs_J,
    plot_cluster_vs_spread
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

all_returns = []


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



def compute_abs_return_acf(returns, max_lag=50):
    """ACF(|r_t|) для диагностики кластеринга."""
    r = np.abs(returns - returns.mean())
    r = r - r.mean()
    acf = []
    var = np.dot(r, r)
    if var == 0:
        return np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c = np.dot(r[:-lag], r[lag:])
            acf.append(c / var)
    return np.array(acf)


def compute_volatility_clusters(returns, threshold_mult=2.0, window=20):
    """
    Считаем длительности высоковолатильных 'кластеров':
    1) rolling std по окну window;
    2) кластер, если std > threshold_mult * median(std).
    """
    if len(returns) < window + 5:
        return []

    vol = pd.Series(returns).rolling(window=window).std().to_numpy()
    vol = vol[~np.isnan(vol)]
    if len(vol) == 0:
        return []

    med = np.median(vol)
    thr = threshold_mult * med

    clusters = []
    current = 0
    for v in vol:
        if v > thr:
            current += 1
        else:
            if current > 0:
                clusters.append(current)
                current = 0
    if current > 0:
        clusters.append(current)
    return clusters


for current_J in tqdm(J_RANGE, desc="Sweep J"):
    for (n_rand, n_fund, n_chart, n_univ) in tqdm(COMBOS, leave=False, desc="Agent mixes"):
        for run in range(N_RUNS):
            is_mm = 1  # фиксируем одного маркет-мейкера

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

            simulator = Simulator(
                exchange=exchange,
                traders=trader_list,
                events=[MarketPriceShock(200, -10)]
            )

            try:
                simulator.simulate(T_STEPS, silent=True)
            except Exception:
                continue

            info = simulator.info
            info_last = info

            r = get_returns_from_info(info)
            if len(r) > T0:
                r_eff = r[T0:]
            else:
                r_eff = r

            if len(r_eff) > 0:
                all_returns.append({
                    'J': current_J,
                    'Random': n_rand,
                    'Fundamentalist': n_fund,
                    'Chartist': n_chart,
                    'Universalist': n_univ,
                    'run_id': run,
                    'returns': r_eff
                })


            vol_series = info.return_volatility(WINDOW)
            vol_J = np.mean(vol_series[T0:]) if len(vol_series) > T0 else 0.0


            max_lag = 20
            acf_abs = compute_abs_return_acf(r_eff, max_lag=max_lag)
            if len(acf_abs) > 1:
                acf_tail_mean = np.mean(acf_abs[1:])
            else:
                acf_tail_mean = 0.0


            clusters = compute_volatility_clusters(r_eff, threshold_mult=2.0, window=WINDOW)
            avg_cluster_len = np.mean(clusters) if clusters else 0.0
            num_clusters = len(clusters)


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
                'vol_J': vol_J,
                'acf_abs_mean': acf_tail_mean,
                'cluster_len_avg': avg_cluster_len,
                'cluster_count': num_clusters,
                'fixed_spread_1000': avg_fixed_spread,
                'orders_count': avg_orders_num
            })


df_results = pd.DataFrame(results)
file_exists = os.path.isfile("results_h3.csv")
df_results.to_csv(
    "results_h3.csv",
    mode="w",
    header=True,
    index=False
)


def permutation_test_diff_means(x, y, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    observed = x.mean() - y.mean()
    combined = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        x_perm = combined[:n_x]
        y_perm = combined[n_x:]
        stat_perm = x_perm.mean() - y_perm.mean()
        if abs(stat_perm) >= abs(observed):
            count += 1
    p_value = count / n_perm
    return observed, p_value


J0 = 0.0
Jc_len = 20.0

len_J0 = df_results[df_results['J'] == J0]['cluster_len_avg'].values
len_Jc = df_results[df_results['J'] == Jc_len]['cluster_len_avg'].values

if len(len_J0) > 1 and len(len_Jc) > 1:
    diff_len, p_len = permutation_test_diff_means(len_Jc, len_J0, n_perm=5000, seed=123)
    print(f"Permutation test for cluster length: J={Jc_len} vs J={J0}")
    print(f"  mean_len(J={Jc_len}) - mean_len(J={J0}) = {diff_len:.4f}, p-value = {p_len:.4e}")
else:
    print("Not enough data for cluster-length permutation test.")

Jc_acf = 30.0

acf_J0 = df_results[df_results['J'] == J0]['acf_abs_mean'].values
acf_Jc = df_results[df_results['J'] == Jc_acf]['acf_abs_mean'].values

if len(acf_J0) > 1 and len(acf_Jc) > 1:
    diff_acf, p_acf = permutation_test_diff_means(acf_Jc, acf_J0, n_perm=5000, seed=456)
    print(f"Permutation test for mean ACF(|r_t|): J={Jc_acf} vs J={J0}")
    print(f"  mean_acf(J={Jc_acf}) - mean_acf(J={J0}) = {diff_acf:.4f}, p-value = {p_acf:.4e}")
else:
    print("Not enough data for ACF permutation test.")


plot_vol_cluster_metrics(df_results)
plot_returns_distribution(info)
plot_abs_return_acf(info)
plot_fixed_spread_vs_J(df_results)
plot_cluster_vs_spread(df_results)
plot_fixed_spread_sweep(df_results)


plt.show()



