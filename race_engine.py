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
    year=None,
    team_mapping=None,
    pitstop_stats=None,
    power_index=None,
    tow_flags=None,
    pitlane_starters=None,
):
    """
    Fully-vectorised race simulation.

    Args:
        race_stats:       dict[driver -> {compound -> {base_pace, deg_slope}}]
        grid_positions:   dict[driver -> float]  (expected grid slot from quali)
        num_iterations:   Monte Carlo iterations
        num_laps:         total race laps
        num_pitstops:     planned stops (1 / 2 / 3)
        pitstop_time_loss: fallback average pit-lane delta (seconds), used only
                           when pitstop_stats/team_mapping aren't supplied.
        team_mapping:     dict[driver] -> {'team': str, ...} from extract_team_mapping
        pitstop_stats:    dict[team] -> {'pit_loss': float, 'botch_prob': float}
                          from extract_pitstop_stats. Falls back to pitstop_time_loss
                          for any team not present (via '__FIELD__' key).
        power_index:      dict[driver] -> float straight-line speed z-score from
                          extract_speed_metrics. Rewards genuine power-unit /
                          energy-deployment advantage on power-sensitive tracks.
        tow_flags:        dict[driver] -> float estimated seconds of tow-inflated
                          grid pace from detect_tow_assisted_laps. A driver's grid
                          slot is trusted less as a signal of their true race pace
                          when it was likely boosted by an unusually large draft
                          (e.g. "Hadjar towed Verstappen to P2 at Spa" should not
                          make the model assume Verstappen has P2-equivalent race pace).
        track_info:       also consulted here for 'sc_probability' / 'vsc_probability'
                          to drive the Safety Car / Virtual Safety Car model below.

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

    # ── Power-Unit Deployment / Straight-Line Speed Correction ─────────
    # 2026 cars are far more sensitive to energy deployment and drag trim than
    # prior regs. A driver/car that is genuinely faster in a straight line
    # (speed-trap z-score from extract_speed_metrics) should be rewarded for
    # it, scaled by how power-sensitive this circuit is (track_info tow_factor
    # is already our proxy for "how much of the lap is flat-out"). This is
    # what lets a car with a straight-line/energy advantage (e.g. better ERS
    # deployment) beat a car that merely looked good in low-fuel single-lap
    # pace but has nothing extra on the straights.
    POWER_INDEX_SCALE = 0.35  # seconds of lap time per 1 std-dev of speed-trap advantage, at tow_factor=1.0
    if power_index:
        tow_factor = track_info.get('tow_factor', 0.15) if track_info else 0.15
        for i, d in enumerate(drivers):
            pidx = power_index.get(d, 0.0)
            base_pace[i] -= pidx * tow_factor * POWER_INDEX_SCALE

    # ── Direct Race-Pace Tow Penalty ──────────────────────────────────
    if tow_flags:
        for i, d in enumerate(drivers):
            tow_penalty = tow_flags.get(d, 0.0)
            if tow_penalty > 0.0:
                # A qualifying slipstream does not exist on clean-air race laps
                base_pace[i] += tow_penalty

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
            grid_pos = grid_positions.get(d, len(drivers))
            # Calculate what this driver's pace *should* be based on qualifying
            expected_pace = pole_pace + ((grid_pos - 1) * 0.12)

            # Apply upper clamping:
            # - If a driver ran heavy fuel or skipped long runs in FP2, their raw pace will be terrible.
            #   We clamp them to their expected grid pace so they aren't unfairly penalized for bad practice data.
            #   We add a tiny +0.05s penalty to ensure pole sitters aren't given *perfect* grace if they lacked data.
            # - If this driver's grid slot was flagged as likely tow-inflated (detect_tow_assisted_laps),
            #   we give them much more room below their "expected" pace — their grid position is a
            #   weaker signal of true race pace than usual, so the clamp shouldn't force them toward it.
            tow_slack = tow_flags.get(d, 0.0) if tow_flags else 0.0
            base_pace[i] = np.minimum(base_pace[i], expected_pace + 0.05 + tow_slack)
            
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
                grid_pos = grid_positions.get(d, len(drivers))
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

    # ── Team-specific pit-lane loss & botch probability ────────────────
    # Real pit crews are not identical: some teams are consistently a few
    # tenths quicker, and every team has some chance of a slow/botched stop.
    # Falls back to the flat pitstop_time_loss scalar if no team data supplied.
    field_fallback = {'pit_loss': pitstop_time_loss, 'botch_prob': 0.06}
    if pitstop_stats:
        field_fallback = pitstop_stats.get('__FIELD__', field_fallback)

    pit_loss_per_driver = np.zeros(num_drivers)
    botch_prob_per_driver = np.zeros(num_drivers)
    for i, d in enumerate(drivers):
        team = team_mapping.get(d, {}).get('team') if team_mapping else None
        stats = (pitstop_stats.get(team, field_fallback) if (pitstop_stats and team) else field_fallback)
        pit_loss_per_driver[i] = stats.get('pit_loss', pitstop_time_loss)
        botch_prob_per_driver[i] = stats.get('botch_prob', 0.06)

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

    if pitlane_starters is None:
        pitlane_starters = []
        
    # ── State tensors ─────────────────────────────────────────────────
    grid_offsets = np.array([grid_positions.get(d, len(drivers)) for d in drivers])
    # Initial grid spacing: ~0.4s per grid slot to prevent instant mega-DRS trains
    total_race_time = np.tile(grid_offsets * 0.4,
                              (num_iterations, 1))  # (iters, drivers)
                              
    # Pit lane start penalty (simulating holding at the end of pit lane)
    for i, d in enumerate(drivers):
        if d in pitlane_starters:
            total_race_time[:, i] += 12.0

    active_mask = np.ones((num_iterations, num_drivers), dtype=bool)
    tire_ages   = np.ones((num_iterations, num_drivers))  # start at lap 1

    # ── Pre-Compute Bootstrapped DNFs (Empirical PMF) ─────────────────
    # 1. Sample total DNFs per iteration based on empirical PMF
    dnf_pmf = [0.10, 0.25, 0.35, 0.20, 0.10]
    dnf_counts = np.array([0, 1, 2, 3, 4])
    num_dnfs_per_iter = np.random.choice(dnf_counts, size=num_iterations, p=dnf_pmf)

    # 2. Extract driver base reliability as weights
    fallback_dnf = reliability_stats.get('__ROOKIE_FALLBACK__', 0.003)
    driver_weights = np.array([reliability_stats.get(d, fallback_dnf) for d in drivers])
    driver_weights /= np.sum(driver_weights)
    
    # 3. Gumbel-Max Trick for vectorized weighted sampling without replacement
    weights_mat = np.tile(driver_weights, (num_iterations, 1))
    u = np.random.uniform(0, 1, size=(num_iterations, num_drivers))
    gumbel_scores = np.log(weights_mat) - np.log(-np.log(u))
    
    # Sort scores descending to find the "top" drivers who will DNF
    rankings = np.argsort(-gumbel_scores, axis=1)
    ranks = np.empty_like(rankings)
    np.put_along_axis(ranks, rankings, np.arange(num_drivers)[np.newaxis, :], axis=1)
    
    # Boolean mask of which drivers DNF in each iteration
    is_dnf = ranks < num_dnfs_per_iter[:, np.newaxis]

    # 4. Assign stochastic DNF laps (Lap 1 has higher weight for chaos)
    lap_probs = np.ones(num_laps)
    # Apply track-specific Turn 1 chaos if provided, else use baseline 5x weight
    lap_probs[0] += (track_info.get('turn_1_chaos', 0.05) * 100) if track_info else 5.0
    lap_probs /= lap_probs.sum()
    
    random_laps = np.random.choice(np.arange(1, num_laps + 1), size=(num_iterations, num_drivers), p=lap_probs)
    dnf_laps = np.where(is_dnf, random_laps, np.inf)

    # Track which stint each driver is on: 0-indexed
    current_stint = np.zeros((num_iterations, num_drivers), dtype=np.intp)

    # Pre-compute driver index row for advanced indexing
    drv_idx = np.arange(num_drivers)                      # (D,)

    # ── Race Day Setup Variance ───────────────────────────────────────
    # Simulates missing the setup window or hitting the sweet spot. 
    # Generates a permanent pace offset for each driver in each universe.
    setup_variance = np.random.normal(0, 0.15, size=(num_iterations, num_drivers))

    # ── Pre-compute Unscheduled Pitstops ──────────────────────────────
    # Probability per lap is 0.0005. We determine if a driver gets an unscheduled 
    # stop over the whole race, and at what lap, to remove RNG from the inner loop.
    unscheduled_probs = 1 - (1 - 0.0005) ** num_laps
    has_unscheduled = np.random.random(size=(num_iterations, num_drivers)) < unscheduled_probs
    unscheduled_lap = np.random.randint(1, num_laps + 1, size=(num_iterations, num_drivers))
    unscheduled_lap = np.where(has_unscheduled, unscheduled_lap, -1)

    # ── Safety Car / Virtual Safety Car Model ──────────────────────────
    # This is the single biggest miss in the previous model: races were
    # simulated as if under green-flag conditions from start to finish. A
    # real SC bunches the field (a car in P8 can be within a second of the
    # leader within a couple of laps), effectively erases the time cost of a
    # pit stop taken during the window, and briefly disables DRS/dirty air.
    # A VSC does the "everyone slows down" and "cheap pit stop" parts but
    # does NOT bunch the field (drivers hold station at delta time).
    sc_prob = track_info.get('sc_probability', 0.35) if track_info else 0.35
    vsc_prob = track_info.get('vsc_probability', 0.25) if track_info else 0.25

    sc_happens = np.random.random(num_iterations) < sc_prob
    vsc_happens = np.random.random(num_iterations) < vsc_prob

    # Don't let cautions start in the first 2 laps (no time to deploy) or the
    # last 3 (no point neutralising a near-finished race).
    safe_lo, safe_hi = 3, max(4, num_laps - 3)
    sc_start = np.random.randint(safe_lo, safe_hi, size=num_iterations)
    sc_duration = np.random.randint(3, 7, size=num_iterations)  # full SC: 3-6 laps
    sc_start = np.where(sc_happens, sc_start, -1)

    vsc_start = np.random.randint(safe_lo, safe_hi, size=num_iterations)
    vsc_duration = np.random.randint(2, 5, size=num_iterations)  # VSC: 2-4 laps
    # Keep VSC from starting inside an already-running SC window (avoid double counting)
    vsc_start = np.where(vsc_happens & ~((vsc_start >= sc_start) & (vsc_start < sc_start + sc_duration)),
                          vsc_start, -1)

    sc_end = sc_start + sc_duration        # (iters,) — exclusive
    vsc_end = vsc_start + vsc_duration

    # Field-median green-flag pace, used to anchor caution-lap pace regardless
    # of which compound a car happens to be on.
    field_flat_pace = float(np.median(base_pace[:, 0]))
    SC_PACE_MULT = 1.38   # full SC laps run ~38% off green-flag pace
    VSC_PACE_MULT = 1.25  # VSC laps run ~25% off green-flag pace
    SC_PIT_DISCOUNT = 0.30   # a stop taken under full SC costs ~30% of its normal loss
    VSC_PIT_DISCOUNT = 0.60  # a stop taken under VSC costs ~60% of its normal loss

    bunching_done = np.zeros(num_iterations, dtype=bool)  # has the one-time SC bunch-up fired yet?

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
        lap_time = current_bp + (tire_ages * current_ds) + setup_variance

        # ── Dynamic Leader Pacing Penalty ─────────────────────────────
        # Find the index of the leader in each iteration
        leader_idx = np.argmin(np.where(active_mask, total_race_time, np.inf), axis=1)
        
        # Create boolean mask for the leader
        is_leader = np.zeros((num_iterations, num_drivers), dtype=bool)
        np.put_along_axis(is_leader, leader_idx[:, np.newaxis], True, axis=1)
        
        # Mean offset: 0.0 for normal cars, +0.15 for the leader (pacing/ERS management)
        mean_offset = np.where(is_leader, 0.15, 0.0)

        # Stochastic variance (expanded to ± 0.4 s to induce organic position swapping)
        # Optimized: Generate N(0, 1) and scale/shift to avoid slow array-based loc in np.random.normal
        lap_time += mean_offset + (np.random.randn(num_iterations, num_drivers) * 0.4)

        # ── Safety Car / VSC pace override ─────────────────────────────
        # Under caution, real racing pace/position battles stop entirely —
        # everyone laps at a common controlled pace instead of their own
        # compound curve. We fully overwrite lap_time for affected rows.
        is_sc_lap = sc_happens & (lap >= sc_start) & (lap < sc_end)          # (iters,)
        is_vsc_lap = vsc_happens & (lap >= vsc_start) & (lap < vsc_end) & (~is_sc_lap)  # (iters,)

        if np.any(is_sc_lap) or np.any(is_vsc_lap):
            sc_pace = field_flat_pace * SC_PACE_MULT + np.random.randn(num_iterations, num_drivers) * 0.12
            vsc_pace = field_flat_pace * VSC_PACE_MULT + np.random.randn(num_iterations, num_drivers) * 0.10
            lap_time = np.where(is_sc_lap[:, np.newaxis], sc_pace, lap_time)
            lap_time = np.where(is_vsc_lap[:, np.newaxis], vsc_pace, lap_time)

            # One-time field bunch-up at the exact lap the full SC is deployed:
            # rank order is preserved but absolute time gaps are crushed down
            # to ~0.8-1.6s per car, exactly like a real field forming up
            # behind the safety car. This is what lets a recovering car
            # (P8 -> P3 style) close for free instead of needing 15 laps of
            # green-flag overtaking to do it.
            just_deployed = is_sc_lap & (lap == sc_start) & (~bunching_done)
            if np.any(just_deployed):
                order = np.argsort(total_race_time, axis=1)
                gap_steps = np.take_along_axis(
                    np.random.uniform(0.8, 1.6, size=(num_iterations, num_drivers)), order, axis=1)
                gap_steps[:, 0] = 0.0
                cum_gap_by_rank = np.cumsum(gap_steps, axis=1)
                leader_time = np.take_along_axis(total_race_time, order, axis=1)[:, 0:1]
                new_time_by_rank = leader_time + cum_gap_by_rank
                new_time = np.empty_like(total_race_time)
                np.put_along_axis(new_time, order, new_time_by_rank, axis=1)
                apply_bunch = just_deployed[:, np.newaxis] & active_mask
                total_race_time = np.where(apply_bunch, new_time, total_race_time)
                bunching_done = bunching_done | just_deployed

        # ── Pit stop on this lap? ─────────────────────────────────────
        pitting = np.zeros((num_iterations, num_drivers), dtype=bool)
        for p_laps in pitstop_lap_arrays:
            pitting |= (lap == p_laps)
            
        # ── Unscheduled Pitstops (Minor Issues) ───────────────────────
        unscheduled_pit = (lap == unscheduled_lap) & active_mask
        pitting |= unscheduled_pit

        # Team-specific pit loss, discounted heavily if the stop falls under
        # a Safety Car (~30% of normal cost) or VSC (~60% of normal cost) —
        # this is the "free pit stop" dynamic that real strategists chase.
        pit_loss_effective = np.where(
            is_sc_lap[:, np.newaxis], pit_loss_per_driver[np.newaxis, :] * SC_PIT_DISCOUNT,
            np.where(is_vsc_lap[:, np.newaxis], pit_loss_per_driver[np.newaxis, :] * VSC_PIT_DISCOUNT,
                     pit_loss_per_driver[np.newaxis, :])
        )

        # Botched stop: team-specific probability of an extra slow stop
        # (wheel gun issue, unsafe release hold, cross-threaded nut, etc.)
        botch_occurs = pitting & (np.random.random(size=(num_iterations, num_drivers)) < botch_prob_per_driver[np.newaxis, :])
        botch_extra = np.where(botch_occurs, np.random.uniform(8.0, 22.0, size=(num_iterations, num_drivers)), 0.0)

        lap_time += np.where(pitting, pit_loss_effective + botch_extra, 0.0)

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
        # DRS is disabled on Lap 1 AND under Safety Car / VSC (no racing battles under caution)
        caution_now = (is_sc_lap | is_vsc_lap)
        drs_bonus = np.where((gaps < 1.0) & (lap > 1) & (~caution_now[:, np.newaxis]), -0.7 * drs_effectiveness, 0.0)

        # No dirty-air battles under caution either — the field is running nose-to-tail
        # at a fixed delta, not fighting for track position.
        dirty_pen = np.where(caution_now[:, np.newaxis], 0.0, dirty_pen)

        pen_sorted = np.zeros_like(total_race_time)
        
        # Combine dirty air and DRS. Add a tiny stochastic element so cars don't indefinitely swap
        stochastic_pass = np.random.normal(1.0, 0.1, size=gaps.shape)
        pen_sorted[:, 1:] = (dirty_pen + drs_bonus) * stochastic_pass

        # ── 2026 Battery Superclipping ──
        # High 'tow_factor' means long straights -> massive battery drain (superclipping).
        # Cars in clean air (leader, or gaps > 1.2s) suffer severe pace loss, while 
        # followers save battery in the slipstream. Suspended under caution — nobody
        # is pushing for lap time, so energy management stops being a differentiator.
        if year is not None and int(year) >= 2026:
            base_superclip = track_info.get('tow_factor', 0.15) if track_info else 0.15
            superclip_penalty = base_superclip * 0.8  # Up to +0.24s penalty per lap for clean air

            # Leader (index 0) is always in clean air
            pen_sorted[:, 0] += np.where(caution_now, 0.0, superclip_penalty)

            # Other cars are in clean air if gap > 1.2s
            clean_air_mask = gaps > 1.2
            pen_sorted[:, 1:] += np.where(clean_air_mask & (~caution_now[:, np.newaxis]), superclip_penalty, 0.0)

        # Scatter penalties back to original driver order
        penalties = np.zeros_like(total_race_time)
        np.put_along_axis(penalties, sort_idx, pen_sorted, axis=1)

        # Apply penalties only to drivers still actively racing
        total_race_time += np.where(active_mask, penalties, 0.0)

        # ── Pre-Computed DNF Check ────────────────────────────────────
        active_mask &= (lap < dnf_laps)

        # ── Restart chaos ──────────────────────────────────────────────
        # The lap immediately after a Safety Car period ends is when
        # bunched-up cars under braking/battling for position produce
        # disproportionately many incidents in real races (this is exactly
        # the kind of situation a recovering driver under pressure — e.g.
        # fighting through the pack — is more likely to have a mechanical
        # or contact-induced DNF in). Scaled by the circuit's own turn_1_chaos
        # proxy since technical/walled circuits punish restart mistakes harder.
        restart_lap_mask = sc_happens & (lap == sc_end)
        if np.any(restart_lap_mask):
            restart_incident_prob = (track_info.get('turn_1_chaos', 0.05) if track_info else 0.05) * 0.5
            restart_incident = (restart_lap_mask[:, np.newaxis] & active_mask &
                                 (np.random.random(size=(num_iterations, num_drivers)) < restart_incident_prob))
            active_mask &= ~restart_incident

            # Extra pace scatter on the restart lap itself (jostling for position)
            extra_restart_noise = np.where(restart_lap_mask[:, np.newaxis],
                                            np.random.randn(num_iterations, num_drivers) * 0.3, 0.0)
            total_race_time += np.where(active_mask, extra_restart_noise, 0.0)

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