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
    season_trends=None,
    weather_context=None,
    track_info=None,
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

    # ── Qualifying Pace Anchor ─────────────────────────────────────────
    # Free Practice 2 lap times are notoriously distorted by unknown fuel loads.
    # To prevent backmarkers (who ran low-fuel FP2 sims) from beating Pole sitters, 
    # we anchor the extracted base pace using their true qualifying hierarchy.
    # 1 grid drop ≈ +0.12s lap time penalty in expected race pace.
    if grid_positions:
        # Instead of using the pole sitter's FP2 time (who might have been sandbagging or had a bad FP2),
        # we ALWAYS use the absolute fastest FP2 time as the baseline for grid anchoring.
        pole_pace = np.min(base_pace, axis=0)
            
        for i, d in enumerate(drivers):
            grid_pos = grid_positions.get(d, 20)
            # Calculate what this driver's pace *should* be based on qualifying
            expected_pace = pole_pace + ((grid_pos - 1) * 0.12)
            
            # Apply upper clamping:
            # - If a driver ran heavy fuel or skipped long runs in FP2, their raw pace will be terrible.
            #   We clamp them to their expected grid pace so they aren't unfairly penalized for bad practice data.
            base_pace[i] = np.minimum(base_pace[i], expected_pace)
            
            # Apply Season Trend Modifier directly to final base pace (Max +/- 0.1s to respect current weekend form)
            if season_trends and d in season_trends:
                sunday_conv = season_trends[d].get('sunday_conversion', 0.0)
                sunday_bonus = np.clip(sunday_conv, -0.1, 0.1)
                base_pace[i] -= sunday_bonus

    # ── Generic Degradation Anchor ─────────────────────────────────────────────
    # Short 3-lap practice stints often produce flat or negative slopes, 
    # resulting in zero tire wear. To prevent drivers from dominating the 
    # race by never losing grip, we floor their tire degradation to at 
    # least 50% of the field median. We use a relaxed absolute floor (-0.02) 
    # to still permit fuel-burn offsets on smooth tracks like Monza.
    for cidx in range(len(COMPOUND_KEYS)):
        median_deg = np.median(deg_slope[:, cidx])
        deg_floor = max(-0.02, median_deg * 0.5)
        deg_slope[:, cidx] = np.maximum(deg_slope[:, cidx], deg_floor)

    # ── Qualifying Degradation Anchor (Setup Correction) ────────────────
    # F1 teams often have terrible tire wear in Friday practice, but fix 
    # their suspension/aero setups overnight. A car that qualifies P2 
    # inherently has excellent downforce and tire management. We calculate 
    # an expected tire degradation based on True Grid Position and blend it 
    # with the FP2 data to simulate overnight setup improvements.
    if grid_positions:
        for cidx in range(len(COMPOUND_KEYS)):
            median_field_deg = np.median(deg_slope[:, cidx])
            
            for i, d in enumerate(drivers):
                grid_pos = grid_positions.get(d, 20)
                # Front runners expect better deg, backmarkers expect worse (+0.003s/lap per grid drop)
                expected_deg = median_field_deg + ((grid_pos - 10) * 0.003)
                
                # Apply Season Trend Modifier to Degradation (Max +/- 0.005 to prevent massive blowouts over long stints)
                if season_trends and d in season_trends:
                    sunday_conv = season_trends[d].get('sunday_conversion', 0.0)
                    deg_bonus = np.clip(sunday_conv * 0.05, -0.005, 0.005)
                    expected_deg -= deg_bonus

                # F1 race pace is heavily dictated by Qualifying speed. We aggressively 
                # blend (30% FP2, 70% Expected) to simulate overnight setup fixes.
                blended_deg = (deg_slope[i, cidx] * 0.3) + (expected_deg * 0.7)
                
                # Strict clamps to preserve the true racing hierarchy
                deg_slope[i, cidx] = np.clip(blended_deg, expected_deg - 0.015, expected_deg + 0.015)

    # ── Weather-Adjusted Tire Degradation ──────────────────────────────
    # Track temperature has a massive impact on tire wear. Hot surfaces
    # (Bahrain ~50°C) dramatically increase graining and blistering.
    # Baseline = 30°C.  Every +10°C → +15% more deg.  Every -10°C → -10% less deg.
    if weather_context:
        track_temp = weather_context.get('track_temp', 30.0)
        temp_delta = track_temp - 30.0  # degrees above baseline
        if temp_delta > 0:
            temp_scale = 1.0 + (temp_delta / 10.0) * 0.15
        else:
            temp_scale = 1.0 + (temp_delta / 10.0) * 0.10
        # Clamp between 0.7x and 1.6x to prevent extreme distortion
        temp_scale = np.clip(temp_scale, 0.7, 1.6)
        deg_slope *= temp_scale

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
    # Initial grid spacing: ~0.4s per grid slot to prevent instant mega-DRS trains
    total_race_time = np.tile(grid_offsets * 0.4,
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

        # ── Dynamic dirty-air penalty (Safe Infinity Arithmetic) ──────
        #    Before sorting, replace np.inf (DNF'd cars) with safe
        #    trailing values spaced 10 s apart from the slowest active
        #    car. This guarantees sorted_time never contains inf, so
        #    the gap subtraction can never produce inf − inf = NaN.
        safe_time = total_race_time.copy()
        inf_mask = np.isinf(safe_time)

        if np.any(inf_mask):
            # Per-row max of active cars (replace inf with -inf to ignore)
            max_active = np.where(inf_mask, -np.inf, safe_time).max(axis=1,
                                                                     keepdims=True)
            # Cumulative count of inf positions per row for spacing
            inf_cumcount = np.cumsum(inf_mask, axis=1)
            safe_time = np.where(inf_mask,
                                 max_active + inf_cumcount * 10.0,
                                 safe_time)

        sort_idx    = np.argsort(safe_time, axis=1)
        sorted_time = np.take_along_axis(safe_time, sort_idx, axis=1)

        # Gap to the car directly ahead — fully NaN-free
        gaps = sorted_time[:, 1:] - sorted_time[:, :-1]

        # Dynamic traffic penalty: up to 0.6 s if right on the gearbox
        dirty_air_mask = gaps < 2.0
        
        # Apply Track Overtaking Difficulty (default 0.5 if not found)
        overtaking_diff = track_info.get('overtaking_diff', 0.5) if track_info else 0.5
        
        # Base penalty scaled by overtaking difficulty (0.3 is the old default)
        # Higher difficulty = harsher penalty for being stuck in dirty air (simulating battery drain & aero loss)
        penalty_multiplier = 0.3 * (overtaking_diff / 0.5)
        dirty_pen = np.where(dirty_air_mask, (2.0 - gaps) * penalty_multiplier, 0.0)
        
        # DRS Bonus: Only applied if within 1.0s, scaled inversely by overtaking difficulty
        # Easy overtaking (low diff) = Powerful DRS (-0.7s)
        # Hard overtaking (high diff e.g. Monaco) = Weak DRS (0.0s)
        drs_effectiveness = np.clip(1.0 - overtaking_diff, 0.0, 1.0)
        # DRS is disabled on Lap 1.
        drs_bonus = np.where((gaps < 1.0) & (lap > 1), -0.7 * drs_effectiveness, 0.0)
        
        pen_sorted = np.zeros_like(total_race_time)
        
        # Combine dirty air and DRS. Add a tiny stochastic element so cars don't indefinitely swap
        stochastic_pass = np.random.normal(1.0, 0.1, size=gaps.shape)
        pen_sorted[:, 1:] = (dirty_pen + drs_bonus) * stochastic_pass

        # Scatter penalties back to original driver order
        penalties = np.zeros_like(total_race_time)
        np.put_along_axis(penalties, sort_idx, pen_sorted, axis=1)

        # Apply penalties only to drivers still actively racing
        total_race_time += np.where(active_mask, penalties, 0.0)

        # ── Stochastic DNF ────────────────────────────────────────────
        dnf_rolls = np.random.random(size=(num_iterations, num_drivers))
        
        # Turn 1 Chaos Modifier
        current_dnf_thresholds = dnf_thresholds.copy()
        if lap == 1 and track_info and 'turn_1_chaos' in track_info:
            current_dnf_thresholds += track_info['turn_1_chaos']
            
        new_dnfs  = (dnf_rolls < current_dnf_thresholds) & active_mask
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
