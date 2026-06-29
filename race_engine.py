import numpy as np

def run_race_sim(race_stats, grid_positions, num_iterations=100000, num_laps=50, num_pitstops=1, pitstop_time_loss=22.0):
    """
    Runs a Monte Carlo simulation for the Race.
    
    Args:
        race_stats: Dict mapping driver to {compound: {'base_pace': float, 'deg_slope': float}}
        grid_positions: Dict mapping driver to their starting grid position (1 to N)
        num_iterations: Number of simulations to run
        num_laps: Total race laps
        num_pitstops: Number of planned pitstops (1, 2, or 3)
        pitstop_time_loss: Average seconds lost in the pitlane
        
    Returns:
        finishing_probabilities: Dict of driver to their probability of Win, Podium, Top10, Finish, DNF
        final_positions: Array of shape (num_iterations, num_drivers) with final ranks
        drivers: list of driver names
    """
    drivers = list(race_stats.keys())
    num_drivers = len(drivers)
    
    if num_drivers == 0:
        return {}, None, []

    # Initialize states
    # total_race_time: shape (num_iterations, num_drivers)
    # Start with staggered times based on grid position (e.g., 0.2s per grid slot)
    grid_offsets = np.array([grid_positions.get(d, 20) * 0.2 for d in drivers])
    total_race_time = np.tile(grid_offsets, (num_iterations, 1))
    
    # Active mask (True if still in race)
    active_mask = np.ones((num_iterations, num_drivers), dtype=bool)
    
    # Tire ages
    tire_ages = np.zeros((num_iterations, num_drivers))
    
    # Tire compounds: 0 for SOFT, 1 for HARD (simplified 1-stop strategy)
    current_compound = np.zeros((num_iterations, num_drivers), dtype=int)
    
    # Generate pitstop laps based on the number of pitstops
    pitstop_laps = []
    if num_pitstops > 0:
        interval = num_laps / (num_pitstops + 1)
        for p in range(1, num_pitstops + 1):
            start_window = max(1, int((interval * p) - (interval * 0.15)))
            end_window = min(num_laps - 1, int((interval * p) + (interval * 0.15)))
            p_laps = np.random.randint(start_window, end_window + 1, size=(num_iterations, num_drivers))
            pitstop_laps.append(p_laps)
    
    # Extract pace data into arrays for vectorization
    # Fallback to defaults if a driver is missing a compound in stats
    base_pace = np.zeros((num_drivers, 2)) # 0: Soft, 1: Hard
    deg_slope = np.zeros((num_drivers, 2))
    
    for i, d in enumerate(drivers):
        stats = race_stats.get(d, {})
        
        soft_stats = stats.get('SOFT', {'base_pace': 90.0, 'deg_slope': 0.1})
        hard_stats = stats.get('HARD', {'base_pace': 91.5, 'deg_slope': 0.04})
        
        base_pace[i, 0] = soft_stats['base_pace']
        deg_slope[i, 0] = soft_stats['deg_slope']
        
        base_pace[i, 1] = hard_stats['base_pace']
        deg_slope[i, 1] = hard_stats['deg_slope']

    # DNF Threshold (approx ~14% over 50 laps)
    DNF_THRESHOLD = 0.003
    
    for lap in range(1, num_laps + 1):
        # Determine pace based on current compound
        # current_compound is (100k, 20), base_pace is (20, 2)
        # We need to extract the pace for each driver in each iteration
        # using advanced indexing
        iter_indices = np.arange(num_iterations)[:, np.newaxis]
        driver_indices = np.arange(num_drivers)
        
        current_base_pace = base_pace[driver_indices, current_compound]
        current_deg_slope = deg_slope[driver_indices, current_compound]
        
        # Calculate lap time: Base + (Age * Deg)
        lap_time = current_base_pace + (tire_ages * current_deg_slope)
        
        # Add pitstop time if pitting this lap
        pitting_mask = np.zeros((num_iterations, num_drivers), dtype=bool)
        for p_laps in pitstop_laps:
            pitting_mask |= (lap == p_laps)
            
        lap_time += np.where(pitting_mask, pitstop_time_loss, 0.0)
        
        # Change compound and reset age if pitting
        current_compound = np.where(pitting_mask, 1, current_compound) # switch to HARD (1)
        tire_ages = np.where(pitting_mask, 0, tire_ages + 1)
        
        # Add random lap time variance (+- 0.3s)
        lap_time += np.random.normal(0, 0.3, size=(num_iterations, num_drivers))
        
        # Add lap time to total race time
        total_race_time += lap_time
        
        # --- Traffic Penalty Math ---
        # Sort current race times to find cars ahead
        # DNF cars will have infinity time, pushed to the back
        sort_indices = np.argsort(total_race_time, axis=1)
        sorted_times = np.take_along_axis(total_race_time, sort_indices, axis=1)
        
        # Calculate gaps to the car ahead
        gaps = np.diff(sorted_times, axis=1)
        
        # Dynamic traffic penalty: scale up to 0.6s penalty if gap is 0, fading to 0s at 2.0s gap
        # Apply only to cars with gap < 2.0s
        dirty_air_mask = gaps < 2.0
        penalties_sorted = np.zeros_like(total_race_time)
        penalties_sorted[:, 1:] = np.where(dirty_air_mask, (2.0 - gaps) * 0.3, 0.0)
        
        penalties = np.zeros_like(total_race_time)
        np.put_along_axis(penalties, sort_indices, penalties_sorted, axis=1)
        
        total_race_time += penalties
        
        # --- Stochastic DNF Math ---
        dnf_rolls = np.random.random(size=(num_iterations, num_drivers))
        new_dnfs = dnf_rolls < DNF_THRESHOLD
        
        # Only those currently active can DNF
        new_dnfs = new_dnfs & active_mask
        active_mask = active_mask & ~new_dnfs
        
        # If DNF, set total race time to infinity so they drop to the back of the classification
        total_race_time = np.where(active_mask, total_race_time, np.inf)

    # Calculate final results
    # Sort total_race_time to get final ranks. (Smallest time = P1)
    final_ranks = np.argsort(np.argsort(total_race_time, axis=1), axis=1) + 1
    
    # Calculate Probabilities
    finishing_probabilities = {}
    
    for i, d in enumerate(drivers):
        driver_ranks = final_ranks[:, i]
        driver_active = active_mask[:, i]
        
        wins = np.sum((driver_ranks == 1) & driver_active)
        podiums = np.sum((driver_ranks <= 3) & driver_active)
        top10s = np.sum((driver_ranks <= 10) & driver_active)
        finishes = np.sum(driver_active)
        dnfs = num_iterations - finishes
        
        finishing_probabilities[d] = {
            'Win': float(wins / num_iterations),
            'Podium': float(podiums / num_iterations),
            'Top10': float(top10s / num_iterations),
            'Finish': float(finishes / num_iterations),
            'DNF': float(dnfs / num_iterations)
        }
        
    return finishing_probabilities, final_ranks, drivers, active_mask
