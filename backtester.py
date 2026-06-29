import json
import time
from data_pipeline import get_session, extract_quali_stats, extract_race_pace_and_deg
from quali_engine import run_quali_sim
from race_engine import run_race_sim

def main(year=2023, race='Bahrain', num_iterations=100000):
    print(f"--- F1 Monte Carlo Simulation: {year} {race} ({num_iterations} iterations) ---")
    
    t0 = time.time()
    
    # 1. Load Data
    print("Loading telemetry data via FastF1...")
    try:
        session_fp3 = get_session(year, race, 'FP3')
        session_fp2 = get_session(year, race, 'FP2')
    except Exception as e:
        print(f"Failed to load sessions: {e}")
        print("Note: If FP3 or FP2 is unavailable (e.g., Sprint weekend), you must adapt the session loading.")
        return

    print("Extracting driver statistics...")
    quali_stats = extract_quali_stats(session_fp3)
    race_stats = extract_race_pace_and_deg(session_fp2)
    
    if not quali_stats or not race_stats:
        print("Error: Could not extract stats. Check if session data contains valid laps.")
        return
        
    t1 = time.time()
    print(f"Data extraction completed in {t1-t0:.2f} seconds.")
    
    # 2. Qualifying Simulation
    print(f"Running Qualifying Simulation ({num_iterations} iterations)...")
    expected_grid, pole_probs, quali_results, quali_drivers = run_quali_sim(quali_stats, num_iterations)
    
    t2 = time.time()
    print(f"Qualifying completed in {t2-t1:.2f} seconds.")
    
    # 3. Race Simulation
    print(f"Running Race Simulation ({num_iterations} iterations)...")
    # For a real track, we should query total laps. Hardcoded to 50 for now.
    num_laps = 50 
    
    finishing_probs, final_ranks, race_drivers, active_mask = run_race_sim(
        race_stats=race_stats,
        grid_positions=expected_grid,
        num_iterations=num_iterations,
        num_laps=num_laps,
        num_pitstops=2, # Defaulting to 2 stops for now, can be configured
        pitstop_time_loss=22.0
    )
    
    t3 = time.time()
    print(f"Race simulation completed in {t3-t2:.2f} seconds.")
    
    # 4. Aggregate & Output
    print("Formatting and saving results...")
    
    output_data = {
        'metadata': {
            'year': year,
            'race': race,
            'iterations': num_iterations,
            'laps': num_laps
        },
        'results': {}
    }
    
    for driver in race_drivers:
        output_data['results'][driver] = {
            'Qualifying': {
                'Expected_Grid': round(expected_grid.get(driver, 20), 2),
                'Pole_Probability': round(pole_probs.get(driver, 0.0), 4)
            },
            'Race': {
                'Win': round(finishing_probs[driver]['Win'], 4),
                'Podium': round(finishing_probs[driver]['Podium'], 4),
                'Top10': round(finishing_probs[driver]['Top10'], 4),
                'Finish': round(finishing_probs[driver]['Finish'], 4),
                'DNF': round(finishing_probs[driver]['DNF'], 4)
            }
        }
        
    output_file = f'results_{race.lower()}_{year}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    # Also save the raw ranks for visualization (KDE distribution)
    # We save a subset of the raw array so we don't create a massive file
    raw_results = {
        'drivers': race_drivers,
        # Only save first 5000 iterations for charting to keep file size reasonable
        'ranks': final_ranks[:5000].tolist(),
        'active_mask': active_mask[:5000].tolist()
    }
    with open(f'raw_ranks_{race.lower()}_{year}.json', 'w') as f:
        json.dump(raw_results, f)

    t4 = time.time()
    print(f"Saved results to {output_file}. Total time: {t4-t0:.2f} seconds.")

if __name__ == "__main__":
    # You can change the year and race string to backtest other events.
    main(year=2023, race='Bahrain', num_iterations=100000)
