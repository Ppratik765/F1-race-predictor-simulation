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


def run_quali_sim(quali_stats, track_info=None, season_trends=None, num_iterations=100_000, power_index=None):
    """
    Vectorised qualifying simulation.

    Args:
        quali_stats: dict[driver] -> {S1_mean, S1_std, S2_mean, S2_std,
                                       S3_mean, S3_std}
        track_info: dict containing track characteristics like tow_factor
        season_trends: dict containing quali_power_rank for sandbagging correction
        num_iterations: number of Monte Carlo draws
        power_index: dict[driver] -> float straight-line speed z-score from
                     extract_speed_metrics (data_pipeline.py). On power-sensitive
                     circuits (high tow_factor), a genuine straight-line/energy
                     deployment advantage shaves real time off the lap even in
                     qualifying trim — this rewards that directly rather than
                     leaving it to be inferred purely from theoretical-best sectors.

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

    # ── Sandbagging Correction (Qualifying Power Rank) ─────────────────
    # If a driver's simulated pace is drastically slower than their historical quali power rank,
    # apply a "Sandbagging Correction" to boost their simulated sectors, reflecting engine mode turn-ups.
    sandbag_bonus = np.zeros(num_drivers)
    if season_trends:
        # Determine the fastest base pace in the field
        field_base_paces = s1_means + s2_means + s3_means
        pole_base_pace = np.min(field_base_paces) if len(field_base_paces) > 0 else 0
        
        for i, d in enumerate(drivers):
            if d in season_trends and 'quali_power_rank' in season_trends[d]:
                expected_delta = season_trends[d]['quali_power_rank']
                actual_delta = field_base_paces[i] - pole_base_pace
                
                # If they are historically faster than their FP3 time suggests, give them a boost
                if expected_delta < actual_delta - 0.1:
                    # Boost by half the difference to be conservative
                    boost = (actual_delta - expected_delta) * 0.5
                    # Distribute boost across 3 sectors
                    sandbag_bonus[i] = boost / 3.0

    s1_means -= sandbag_bonus
    s2_means -= sandbag_bonus
    s3_means -= sandbag_bonus

    # ── Straight-line Speed / Power-Unit Correction ────────────────────
    # Scaled down vs. the race-pace version (quali laps are short — one
    # or two straights — so the absolute time on offer is smaller than
    # over a full race distance).
    POWER_INDEX_SCALE_QUALI = 0.12  # seconds per 1 std-dev of speed-trap advantage, at tow_factor=1.0
    if power_index and track_info:
        tow_factor = track_info.get('tow_factor', 0.15)
        power_bonus = np.array([power_index.get(d, 0.0) for d in drivers]) * tow_factor * POWER_INDEX_SCALE_QUALI
        per_sector = power_bonus / 3.0
        s1_means -= per_sector
        s2_means -= per_sector
        s3_means -= per_sector

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

    # ── Slipstream / Tow Factor ────────────────────────────────────────
    # For each iteration, grant a random tow bonus to 3 drivers.
    # This simulates the chaos of Q3 out-laps and slipstreaming.
    if track_info and 'tow_factor' in track_info and track_info['tow_factor'] > 0:
        tow_factor = track_info['tow_factor']
        # Select 3 random drivers per iteration to get the tow
        tow_recipients = np.argsort(np.random.random(size=(num_iterations, num_drivers)), axis=1)[:, :3]
        
        # Create a mask for those who get the tow
        tow_mask = np.zeros((num_iterations, num_drivers), dtype=bool)
        np.put_along_axis(tow_mask, tow_recipients, True, axis=1)
        
        # Apply the tow reduction
        total_laps = np.where(tow_mask, total_laps - tow_factor, total_laps)

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