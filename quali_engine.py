"""
quali_engine.py — Qualifying Monte Carlo Micro-Simulation
==========================================================
Draws random sector times from each driver's theoretical-best normal
distribution (S1 + S2 + S3) across 100 000 iterations.

Track evolution is modelled by assigning each driver a random run-order
per iteration; later runners receive a multiplicative grip improvement
of up to 1.5 % (configurable).

Slipstream / tow effects are modelled per-track: at low-downforce circuits
(Monza, Spa), randomly selected drivers receive a tow bonus each iteration.

Historical Qualifying Power Rank (sandbagging correction) gently pulls
drivers toward their true qualifying pace when FP3 telemetry is misleading.

Outputs: expected grid positions and pole-position probability density.
"""

import numpy as np


# ── Slipstream ranges (imported from data_pipeline at call site) ──────────
SLIPSTREAM_DEFAULTS = {
    'LOW_DF':  (0.15, 0.30),
    'MEDIUM':  (0.05, 0.12),
    'HIGH_DF': (0.00, 0.03),
}


def run_quali_sim(
    quali_stats,
    num_iterations=100_000,
    track_type='MEDIUM',
    quali_power_rank=None,
    season_trends=None,
):
    """
    Vectorised qualifying simulation.

    Args:
        quali_stats:       dict[driver] -> {S1_mean, S1_std, S2_mean, S2_std,
                                            S3_mean, S3_std}
        num_iterations:    number of Monte Carlo draws
        track_type:        'HIGH_DF', 'MEDIUM', or 'LOW_DF' (for slipstream)
        quali_power_rank:  dict[driver -> float] historical avg delta-to-pole
        season_trends:     dict[driver -> {sunday_conversion, power_rank_delta}]

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

    # ── Historical Qualifying Power Rank (Sandbagging Correction) ──────
    #    If a driver's FP3 pace suggests P6 but their historical delta says
    #    they're a P2 qualifier, gently correct their sector means.
    #    Blend factor: 40% toward historical pace (conservative).
    if quali_power_rank and len(quali_power_rank) > 0:
        # Calculate each driver's FP3-derived total theoretical best
        fp3_totals = s1_means + s2_means + s3_means
        fp3_pole_time = np.min(fp3_totals)

        for i, d in enumerate(drivers):
            if d in quali_power_rank:
                fp3_delta = fp3_totals[i] - fp3_pole_time
                hist_delta = quali_power_rank[d]

                # Only correct if FP3 makes the driver look significantly slower
                # than their historical qualifying pace (potential sandbagging)
                if fp3_delta > hist_delta + 0.05:
                    correction = (fp3_delta - hist_delta) * 0.4
                    # Distribute correction proportionally across all 3 sectors
                    total = s1_means[i] + s2_means[i] + s3_means[i]
                    s1_means[i] -= correction * (s1_means[i] / total)
                    s2_means[i] -= correction * (s2_means[i] / total)
                    s3_means[i] -= correction * (s3_means[i] / total)

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

    # ── Slipstream / Tow Effect ────────────────────────────────────────
    #   At low-downforce tracks, randomly 2-4 drivers "get a tow" per iteration.
    #   They receive a time bonus drawn from the track-specific range.
    tow_range = SLIPSTREAM_DEFAULTS.get(track_type, (0.05, 0.12))
    tow_min, tow_max = tow_range

    if tow_max > 0.01:  # Only apply if there's a meaningful tow effect
        # Number of drivers getting a tow: 2-4
        num_towed = min(4, max(2, num_drivers // 5))

        # Create a tow mask: for each iteration, randomly select drivers
        tow_mask = np.zeros((num_iterations, num_drivers), dtype=bool)
        for it in range(num_iterations):
            tow_indices = np.random.choice(num_drivers, size=num_towed, replace=False)
            tow_mask[it, tow_indices] = True

        # Draw tow bonus values
        tow_bonus = np.random.uniform(tow_min, tow_max,
                                       size=(num_iterations, num_drivers))
        total_laps -= np.where(tow_mask, tow_bonus, 0.0)

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
