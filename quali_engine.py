"""
quali_engine.py — Qualifying Monte Carlo Micro-Simulation
==========================================================
Draws random sector times from each driver's theoretical-best normal
distribution (S1 + S2 + S3) across 100 000 iterations.

Track evolution is modelled by assigning each driver a random run-order
per iteration; later runners receive a multiplicative grip improvement
of up to 1.5 % (configurable).

Outputs: expected grid positions and pole-position probability density.
"""

import numpy as np


def run_quali_sim(quali_stats, num_iterations=100_000):
    """
    Vectorised qualifying simulation.

    Args:
        quali_stats: dict[driver] -> {S1_mean, S1_std, S2_mean, S2_std,
                                       S3_mean, S3_std}
        num_iterations: number of Monte Carlo draws

    Returns:
        expected_grid:      dict[driver -> float]   (average grid slot)
        pole_probabilities: dict[driver -> float]   (P(pole))
        full_results:       ndarray (iterations × drivers) of lap times
        drivers:            list[str]
    """
    drivers = list(quali_stats.keys())
    num_drivers = len(drivers)

    if num_drivers == 0:
        return {}, {}, None, []

    # ── Vectorise stats into arrays (shape: num_drivers,) ──────────────
    s1_means = np.array([quali_stats[d]['S1_mean'] for d in drivers])
    s1_stds  = np.array([quali_stats[d]['S1_std']  for d in drivers])
    s2_means = np.array([quali_stats[d]['S2_mean'] for d in drivers])
    s2_stds  = np.array([quali_stats[d]['S2_std']  for d in drivers])
    s3_means = np.array([quali_stats[d]['S3_mean'] for d in drivers])
    s3_stds  = np.array([quali_stats[d]['S3_std']  for d in drivers])

    # ── Draw random sector times  (iterations × drivers) ──────────────
    s1 = np.random.normal(loc=s1_means, scale=s1_stds,
                          size=(num_iterations, num_drivers))
    s2 = np.random.normal(loc=s2_means, scale=s2_stds,
                          size=(num_iterations, num_drivers))
    s3 = np.random.normal(loc=s3_means, scale=s3_stds,
                          size=(num_iterations, num_drivers))

    total_laps = s1 + s2 + s3                       # (iters, drivers)

    # ── Track evolution by run order ───────────────────────────────────
    #   Each iteration shuffles a random run order (0 … N-1).
    #   The last driver on track (order = N-1) receives the maximum
    #   improvement multiplier (e.g. ×0.985 → 1.5 % faster).
    #   The first driver on track (order = 0) gets no benefit (×1.0).
    run_order = np.argsort(
        np.random.random(size=(num_iterations, num_drivers)), axis=1
    )
    max_improvement = 0.015  # 1.5 % of lap time
    track_evo = 1.0 - (run_order / max(num_drivers - 1, 1)) * max_improvement
    total_laps *= track_evo

    # ── Rank drivers (lowest time = P1) ────────────────────────────────
    ranks = np.argsort(np.argsort(total_laps, axis=1), axis=1) + 1

    pole_counts    = (ranks == 1).sum(axis=0)
    pole_probs     = pole_counts / num_iterations
    expected_ranks = ranks.mean(axis=0)

    pole_probabilities = {drivers[i]: float(pole_probs[i])
                          for i in range(num_drivers)}
    expected_grid      = {drivers[i]: float(expected_ranks[i])
                          for i in range(num_drivers)}

    return expected_grid, pole_probabilities, total_laps, drivers
