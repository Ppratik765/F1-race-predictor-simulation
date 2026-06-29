"""
race_engine.py — Race Monte Carlo Macro-Simulation
====================================================
State-machine loop that simulates a full race distance lap-by-lap across
100 000 iterations *simultaneously* using pure NumPy vectorisation.

Key mechanics:
  • Lap time = base_pace + tire_age × deg_slope + noise
  • Dynamic dirty-air penalty scaled by gap (capped to prevent compounding)
  • N-stop pit-stop strategy with randomised windows
  • Stochastic DNF trigger per lap → produces bimodal finishing distributions
"""

import numpy as np


def run_race_sim(
    race_stats,
    reliability_stats,
    grid_positions,
    num_iterations=100_000,
    num_laps=50,
    num_pitstops=1,
    pitstop_time_loss=22.0,
):
    """
    Fully-vectorised race simulation.

    Args:
        race_stats:       dict[driver -> {compound -> {base_pace, deg_slope}}]
        grid_positions:   dict[driver -> float]  (expected grid slot from quali)
        num_iterations:   Monte Carlo iterations
        num_laps:         total race laps
        num_pitstops:     planned stops (1 / 2 / 3)
        pitstop_time_loss: average pit-lane delta (seconds)

    Returns:
        finishing_probabilities: dict[driver -> {Win, Podium, Top10, Finish, DNF}]
        final_ranks:             ndarray (iterations × drivers)
        drivers:                 list[str]
        active_mask:             ndarray (iterations × drivers) bool
    """
    drivers = list(race_stats.keys())
    num_drivers = len(drivers)

    if num_drivers == 0:
        return {}, None, [], np.array([])

    # ── Compounds used in strategy sequence ────────────────────────────
    #    For a 1-stop: SOFT → HARD
    #    For a 2-stop: SOFT → HARD → MEDIUM  (indices 0 → 1 → 2)
    #    For a 3-stop: SOFT → HARD → MEDIUM → HARD
    COMPOUND_KEYS = ['SOFT', 'MEDIUM', 'HARD']

    # ── Build pace arrays  (num_drivers × 3) ──────────────────────────
    #    Column 0 = SOFT, 1 = MEDIUM, 2 = HARD
    #    Fallback: if a compound is missing for a driver, use the field
    #    median for that compound (already filled by data_pipeline, but
    #    we guard here too).
    field_medians = {}
    for cidx, ckey in enumerate(COMPOUND_KEYS):
        paces = [race_stats[d][ckey]['base_pace']
                 for d in drivers if ckey in race_stats[d]]
        field_medians[ckey] = float(np.median(paces)) if paces else 90.0

    base_pace = np.zeros((num_drivers, 3))
    deg_slope = np.zeros((num_drivers, 3))

    for i, d in enumerate(drivers):
        stats = race_stats[d]
        for cidx, ckey in enumerate(COMPOUND_KEYS):
            cstats = stats.get(ckey, {
                'base_pace': field_medians[ckey],
                'deg_slope': 0.06,
            })
            base_pace[i, cidx] = cstats['base_pace']
            deg_slope[i, cidx] = cstats['deg_slope']

    # ── Strategy: compound sequence per stint ─────────────────────────
    #    stint_compounds[s] = compound index used during stint s
    if num_pitstops == 1:
        stint_compounds = [0, 2]        # SOFT → HARD
    elif num_pitstops == 2:
        stint_compounds = [0, 2, 1]     # SOFT → HARD → MEDIUM
    elif num_pitstops >= 3:
        stint_compounds = [0, 2, 1, 2]  # SOFT → HARD → MEDIUM → HARD
    else:
        stint_compounds = [0]           # No stop

    # ── Generate pit-stop lap arrays ──────────────────────────────────
    pitstop_lap_arrays = []
    if num_pitstops > 0:
        interval = num_laps / (num_pitstops + 1)
        for p in range(1, num_pitstops + 1):
            centre = interval * p
            lo = max(2, int(centre - interval * 0.15))
            hi = min(num_laps - 2, int(centre + interval * 0.15))
            if hi <= lo:
                hi = lo + 1
            p_laps = np.random.randint(lo, hi + 1,
                                       size=(num_iterations, num_drivers))
            pitstop_lap_arrays.append(p_laps)

    # ── State tensors ─────────────────────────────────────────────────
    grid_offsets = np.array([grid_positions.get(d, 20) for d in drivers])
    total_race_time = np.tile(grid_offsets * 0.2,
                              (num_iterations, 1))  # (iters, drivers)

    active_mask = np.ones((num_iterations, num_drivers), dtype=bool)
    tire_ages   = np.ones((num_iterations, num_drivers))  # start at lap 1

    # ── DNF thresholds array ──────────────────────────────────────────
    fallback_dnf = reliability_stats.get('__ROOKIE_FALLBACK__', 0.003)
    dnf_thresholds = np.array([
        reliability_stats.get(d, fallback_dnf) for d in drivers
    ])  # shape (num_drivers,)

    # Track which stint each driver is on: 0-indexed
    current_stint = np.zeros((num_iterations, num_drivers), dtype=np.intp)

    # Pre-compute driver index row for advanced indexing
    drv_idx = np.arange(num_drivers)                      # (D,)

    # DNF probability per lap (≈ 14 % retirement rate over 57 laps)
    DNF_PROB_PER_LAP = 0.003

    # ── Main lap loop ─────────────────────────────────────────────────
    for lap in range(1, num_laps + 1):

        # ── Determine current compound index for each (iter, driver) ──
        #    current_stint is (iters, D) with values 0…num_pitstops
        #    stint_compounds maps stint → compound index
        #    We clip to the last stint if somehow out of range.
        stint_clipped = np.clip(current_stint, 0, len(stint_compounds) - 1)
        compound_idx = np.array(stint_compounds)[stint_clipped]  # (iters, D)

        # ── Look up pace & deg via explicit advanced indexing ─────────
        #    base_pace is (D, 3).  We need (iters, D).
        #    Index: base_pace[driver_index, compound_index_per_iter]
        #    → broadcast driver index across iterations
        current_bp = base_pace[drv_idx, compound_idx]     # (iters, D)
        current_ds = deg_slope[drv_idx, compound_idx]     # (iters, D)

        # ── Lap time ──────────────────────────────────────────────────
        lap_time = current_bp + (tire_ages * current_ds)

        # Stochastic variance (± 0.3 s)
        lap_time += np.random.normal(0, 0.3,
                                     size=(num_iterations, num_drivers))

        # ── Pit stop on this lap? ─────────────────────────────────────
        pitting = np.zeros((num_iterations, num_drivers), dtype=bool)
        for p_laps in pitstop_lap_arrays:
            pitting |= (lap == p_laps)

        lap_time += np.where(pitting, pitstop_time_loss, 0.0)

        # Advance stint counter and reset tire age on pit
        current_stint = np.where(pitting, current_stint + 1, current_stint)
        tire_ages     = np.where(pitting, 1.0, tire_ages + 1.0)

        # ── Accumulate race time (only active cars) ───────────────────
        total_race_time += np.where(active_mask, lap_time, 0.0)

        # ── Dynamic dirty-air penalty ─────────────────────────────────
        #    Sort cars by current race time; compute gap to car ahead.
        #    Penalty = (2.0 − gap) × 0.3, clamped to [0, 0.6].
        #    Cap total penalty per lap at 0.6 s to prevent compounding.
        sort_idx    = np.argsort(total_race_time, axis=1)
        sorted_time = np.take_along_axis(total_race_time, sort_idx, axis=1)

# --- Traffic Penalty Math ---
        # Sort current race times to find cars ahead
        sort_idx    = np.argsort(total_race_time, axis=1)
        sorted_time = np.take_along_axis(total_race_time, sort_idx, axis=1)

        # Calculate gaps to the car ahead
        gaps = sorted_time[:, 1:] - sorted_time[:, :-1]     

        # SAFEGUARD: Replace any NaNs generated by inf - inf calculations with 2.0s
        # This treats retired cars as being safely out of the dirty-air window
        gaps = np.where(np.isnan(gaps), 2.0, gaps)

        # Dynamic traffic penalty: scaled up to 0.6s penalty if right on the gearbox
        dirty_air_mask = gaps < 2.0
        pen_sorted = np.zeros_like(total_race_time)
        pen_sorted[:, 1:] = np.where(dirty_air_mask, (2.0 - gaps) * 0.3, 0.0)

        # Scatter penalties back to original driver arrays
        penalties = np.zeros_like(total_race_time)
        np.put_along_axis(penalties, sort_idx, pen_sorted, axis=1)
        
        # Apply penalties only to drivers who are still actively racing
        total_race_time += np.where(active_mask, penalties, 0.0)

        # ── Stochastic DNF ────────────────────────────────────────────
        dnf_rolls = np.random.random(size=(num_iterations, num_drivers))
        new_dnfs  = (dnf_rolls < dnf_thresholds) & active_mask
        active_mask &= ~new_dnfs
        total_race_time = np.where(active_mask, total_race_time, np.inf)

    # ── Final classification ──────────────────────────────────────────
    final_ranks = np.argsort(np.argsort(total_race_time, axis=1), axis=1) + 1

    finishing_probabilities = {}
    for i, d in enumerate(drivers):
        r = final_ranks[:, i]
        a = active_mask[:, i]

        wins    = int(np.sum((r == 1) & a))
        podiums = int(np.sum((r <= 3) & a))
        top10s  = int(np.sum((r <= 10) & a))
        finishes = int(np.sum(a))
        dnfs    = num_iterations - finishes

        finishing_probabilities[d] = {
            'Win':    wins    / num_iterations,
            'Podium': podiums / num_iterations,
            'Top10':  top10s  / num_iterations,
            'Finish': finishes / num_iterations,
            'DNF':    dnfs    / num_iterations,
        }

    return finishing_probabilities, final_ranks, drivers, active_mask
