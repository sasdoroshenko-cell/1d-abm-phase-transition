from AgentBasedModel import *
from AgentBasedModel.utils.math import *
from AgentBasedModel.visualization.market import (
    plot_returns_distribution, plot_depth_by_mode,
    plot_spread_by_mode, plot_skewness_by_mode,
    plot_gain_loss_asymmetry, plot_skew_by_chartist_and_mode,
    plot_gain_loss_by_chartist,
    plot_spread_by_chartist_and_mode,
    plot_depth_by_chartist_and_mode,
)

import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, ks_2samp
import random


J_RANGE = np.linspace(0, 50, 6)
WINDOW = 20
T_STEPS = 500
T0 = 100
T0_MICRO = 50
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


def get_fixed_volume_spread(exchange, target_qty=1000, allow_partial=True):
    def get_weighted_price(order_list, target):
        if not order_list:
            return None
        total_qty, total_cost = 0, 0
        for order in order_list:
            take = min(order.qty, target - total_qty)
            total_cost += take * order.price
            total_qty += take
            if total_qty >= target:
                break
        if total_qty == 0:
            return None
        if total_qty < target and not allow_partial:
            return None
        return total_cost / total_qty

    avg_ask = get_weighted_price(exchange.order_book['ask'], target_qty)
    avg_bid = get_weighted_price(exchange.order_book['bid'], target_qty)
    return avg_ask - avg_bid if (avg_ask is not None and avg_bid is not None) else None


def get_returns_from_info(info):
    prices = np.array(info.prices)
    returns = np.diff(prices) / prices[:-1]
    return returns


returns_baseline = [] 
returns_panic = []

for current_J in tqdm(J_RANGE, desc="Sweep J"):
    for (n_rand, n_fund, n_chart, n_univ) in tqdm(COMBOS, leave=False, desc="Agent mixes"):
        for panic_mode in [0, 1]:
            for run in range(N_RUNS):
                is_mm = 1

                exchange = ExchangeAgent(volume=1000)

                if panic_mode == 0:
                    chartist_class = Chartist
                else:
                    chartist_class = PanicChartist

                trader_list = [
                    *[Random(exchange, 10 ** 3) for _ in range(n_rand)],
                    *[Fundamentalist(exchange, 10 ** 3) for _ in range(n_fund)],
                    *[chartist_class(exchange, 10 ** 3, J=float(current_J)) for _ in range(n_chart)],
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


                returns = get_returns_from_info(info)
                if len(returns) > T0:
                    returns_eff = returns[T0:]
                else:
                    returns_eff = returns

                skewness = skew(returns_eff) if len(returns_eff) > 0 else 0.0

                if len(returns_eff) > 0:
                    cum = np.cumprod(1 + returns_eff)
                    peak = np.maximum.accumulate(cum)
                    drawdown = (cum - peak) / peak
                    max_drawdown = drawdown.min()

                    trough = np.minimum.accumulate(cum)
                    runup = (cum - trough) / trough
                    max_runup = runup.max()
                else:
                    max_drawdown = 0.0
                    max_runup = 0.0

                if len(returns_eff) > 0:
                    if panic_mode == 0:
                        returns_baseline.append((current_J, n_chart, returns_eff))
                    else:
                        returns_panic.append((current_J, n_chart, returns_eff))

                step_spreads = []
                step_orders_count = []
                step_bids = []
                step_asks = []

                for _ in range(T_STEPS):
                    try:
                        simulator.simulate(1, silent=True)
                        s = get_fixed_volume_spread(exchange, TARGET_QTY, allow_partial=True)
                        if s is not None:
                            step_spreads.append(s)

                        bids = exchange.order_book['bid']
                        asks = exchange.order_book['ask']
                        n_bids = len(bids)
                        n_asks = len(asks)
                        step_orders_count.append(n_bids + n_asks)

                        best_bid = bids[0].price if n_bids > 0 else 0.0
                        best_ask = asks[0].price if n_asks > 0 else 0.0
                        step_bids.append(best_bid)
                        step_asks.append(best_ask)
                    except Exception:
                        break

                cut = T0_MICRO if len(step_spreads) > T0_MICRO else 0
                avg_fixed_spread = np.mean(step_spreads[cut:]) if step_spreads[cut:] else 0.0
                avg_orders_num = np.mean(step_orders_count[cut:]) if step_orders_count[cut:] else 0.0
                avg_best_bid = np.mean(step_bids[T0:]) if len(step_bids) > T0 else 0.0
                avg_best_ask = np.mean(step_asks[T0:]) if len(step_asks) > T0 else 0.0

                results.append({
                    'J': current_J,
                    'Random': n_rand,
                    'Fundamentalist': n_fund,
                    'Chartist': n_chart,
                    'Universalist': n_univ,
                    'MarketMaker': is_mm,
                    'panic_mode': panic_mode,
                    'run_id': run,
                    'fixed_spread_1000': avg_fixed_spread,
                    'orders_count': avg_orders_num,
                    'best_bid': avg_best_bid,
                    'best_ask': avg_best_ask,
                    'skew': skewness,
                    'max_drawdown': max_drawdown,
                    'max_runup': max_runup
                })


df_results = pd.DataFrame(results)
file_exists = os.path.isfile("results_h2.csv")
df_results.to_csv(
    "results_h2.csv",
    mode="w",
    header=True,
    index=False
)

J_TEST = 30.0
CHARTIST_TEST = 5

r0 = []
r1 = []

for J_val, ch_n, r_eff in returns_baseline:
    if J_val == J_TEST and ch_n == CHARTIST_TEST:
        r0.append(r_eff)

for J_val, ch_n, r_eff in returns_panic:
    if J_val == J_TEST and ch_n == CHARTIST_TEST:
        r1.append(r_eff)

if r0 and r1:
    r0_all = np.concatenate(r0)
    r1_all = np.concatenate(r1)
    ks_stat, ks_p = ks_2samp(r0_all, r1_all)
    print(f"KS test for returns, J={J_TEST}, Chartist={CHARTIST_TEST}")
    print(f"  statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")
else:
    print(f"Not enough return data for KS test at J={J_TEST}, Chartist={CHARTIST_TEST}.")



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

sub_base = df_results[
    (df_results['J'] == J_TEST) &
    (df_results['Chartist'] == CHARTIST_TEST) &
    (df_results['panic_mode'] == 0)
]

sub_panic = df_results[
    (df_results['J'] == J_TEST) &
    (df_results['Chartist'] == CHARTIST_TEST) &
    (df_results['panic_mode'] == 1)
]

if len(sub_base) > 1 and len(sub_panic) > 1:
    diff_skew, p_skew = permutation_test_diff_means(
        sub_panic['skew'].values,
        sub_base['skew'].values,
        n_perm=5000,
        seed=123
    )
    print(f"Permutation test for skew, J={J_TEST}, Chartist={CHARTIST_TEST}")
    print(f"  mean(skew_panic) - mean(skew_base) = {diff_skew:.4f}, p-value = {p_skew:.4e}")

    diff_dd, p_dd = permutation_test_diff_means(
        -sub_panic['max_drawdown'].values,
        -sub_base['max_drawdown'].values,
        n_perm=5000,
        seed=456
    )
    print(f"Permutation test for max drawdown, J={J_TEST}, Chartist={CHARTIST_TEST}")
    print(f"  mean(|DD_panic|) - mean(|DD_base|) = {diff_dd:.4f}, p-value = {p_dd:.4e}")
else:
    print(f"Not enough runs for permutation tests at J={J_TEST}, Chartist={CHARTIST_TEST}.")

plot_returns_distribution(info)
plot_skewness_by_mode(df_results)
plot_gain_loss_asymmetry(df_results)
plot_spread_by_chartist_and_mode(df_results)
plot_depth_by_chartist_and_mode(df_results)
plt.show()



