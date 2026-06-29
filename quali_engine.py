import numpy as np

def run_quali_sim(quali_stats, num_iterations=100000):
    """
    Runs a Monte Carlo simulation for Qualifying.
    
    Args:
        quali_stats: Dict of driver -> {S1_mean, S1_std, S2_mean, S2_std, S3_mean, S3_std}
        num_iterations: Number of simulations to run
        
    Returns:
        expected_grid: Dict mapping driver to their expected (average) grid position
        pole_probabilities: Dict mapping driver to their probability of getting Pole Position
        full_results: numpy array of shape (num_iterations, num_drivers) with lap times
        drivers: list of driver names corresponding to the array indices
    """
    drivers = list(quali_stats.keys())
    num_drivers = len(drivers)
    
    if num_drivers == 0:
        return {}, {}, None, []

    # Prepare arrays for vectorized generation
    s1_means = np.array([quali_stats[d]['S1_mean'] for d in drivers])
    s1_stds = np.array([quali_stats[d]['S1_std'] for d in drivers])
    s2_means = np.array([quali_stats[d]['S2_mean'] for d in drivers])
    s2_stds = np.array([quali_stats[d]['S2_std'] for d in drivers])
    s3_means = np.array([quali_stats[d]['S3_mean'] for d in drivers])
    s3_stds = np.array([quali_stats[d]['S3_std'] for d in drivers])
    
    # Generate random sector times for all drivers and iterations simultaneously
    # Shape: (num_iterations, num_drivers)
    s1_sim = np.random.normal(loc=s1_means, scale=s1_stds, size=(num_iterations, num_drivers))
    s2_sim = np.random.normal(loc=s2_means, scale=s2_stds, size=(num_iterations, num_drivers))
    s3_sim = np.random.normal(loc=s3_means, scale=s3_stds, size=(num_iterations, num_drivers))
    
    # Sum sectors to get total lap time
    total_laps = s1_sim + s2_sim + s3_sim
    
    # Apply track evolution based on run order (drivers running later get a better track)
    # Generate a random run order for each iteration: 0 (first) to num_drivers-1 (last)
    run_order = np.argsort(np.random.random(size=(num_iterations, num_drivers)), axis=1)
    
    # Last driver gets max track evolution (e.g. 1.5% improvement -> 0.985 multiplier)
    # First driver gets no track evolution (1.0 multiplier)
    track_evo = 1.0 - (run_order / (num_drivers - 1)) * 0.015
    total_laps *= track_evo
    
    # Ranks (Grid positions): argsort twice gets the rank (0-indexed, so add 1)
    # The lowest time is P1.
    ranks = np.argsort(np.argsort(total_laps, axis=1), axis=1) + 1
    
    # Calculate Pole Probabilities
    # Pole is where rank == 1
    poles = (ranks == 1).sum(axis=0)
    pole_probs = poles / num_iterations
    
    # Calculate Expected Grid Position
    expected_ranks = ranks.mean(axis=0)
    
    pole_probabilities = {drivers[i]: float(pole_probs[i]) for i in range(num_drivers)}
    expected_grid = {drivers[i]: float(expected_ranks[i]) for i in range(num_drivers)}
    
    return expected_grid, pole_probabilities, total_laps, drivers
