import os
import fastf1
import pandas as pd
import numpy as np

# Configure FastF1 cache
CACHE_DIR = 'fastf1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def get_session(year, race, session_type):
    """Loads a fastf1 session."""
    session = fastf1.get_session(year, race, session_type)
    session.load()
    return session

def filter_laps(laps):
    """Filters out incomplete laps, out laps, in laps, safety car, and VSC laps."""
    # Filter for valid track conditions (no SC or VSC) and valid laps
    valid_laps = laps.pick_accurate().pick_track_status('1')
    return valid_laps

def extract_quali_stats(session):
    """
    Extracts Sector 1, 2, and 3 mean and standard deviation per driver from FP3/Q.
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    
    drivers = pd.unique(filtered_laps['Driver'])
    
    quali_stats = {}
    
    for driver in drivers:
        driver_laps = filtered_laps.pick_driver(driver)
        # Drop laps where any sector time is missing
        driver_laps = driver_laps.dropna(subset=['Sector1Time', 'Sector2Time', 'Sector3Time'])
        
        if len(driver_laps) < 3:
            # Not enough data, use field average or fallback (simplified here)
            continue
            
        s1 = driver_laps['Sector1Time'].dt.total_seconds().values
        s2 = driver_laps['Sector2Time'].dt.total_seconds().values
        s3 = driver_laps['Sector3Time'].dt.total_seconds().values
        
        quali_stats[driver] = {
            'S1_mean': np.mean(s1),
            'S1_std': np.std(s1),
            'S2_mean': np.mean(s2),
            'S2_std': np.std(s2),
            'S3_mean': np.mean(s3),
            'S3_std': np.std(s3)
        }
        
    return quali_stats

def extract_race_pace_and_deg(session):
    """
    Analyzes long runs in FP2 to determine base pace and tire degradation per driver/compound.
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    drivers = pd.unique(filtered_laps['Driver'])
    
    race_stats = {}
    
    for driver in drivers:
        driver_laps = filtered_laps.pick_driver(driver)
        driver_stats = {}
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_laps = driver_laps[driver_laps['Compound'] == compound]
            
            # Long run typically > 4 laps
            if len(compound_laps) > 4:
                # Basic linear regression to find slope (degradation)
                # X = Tyres life, Y = LapTime in seconds
                x = compound_laps['TyreLife'].values
                y = compound_laps['LapTime'].dt.total_seconds().values
                
                # Remove outliers (e.g., traffic) - basic IQR filter
                q1, q3 = np.percentile(y, [25, 75])
                iqr = q3 - q1
                mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
                
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 3:
                    # Fit a line: y = mx + c (where m is deg, c is base pace)
                    m, c = np.polyfit(x_clean, y_clean, 1)
                    
                    # Ensure degradation isn't heavily negative (which means track evo overpowered deg in practice)
                    m = max(m, 0.01) 
                    
                    driver_stats[compound] = {
                        'base_pace': c,
                        'deg_slope': m
                    }
                else:
                    # Fallback if filtering removed too much
                    driver_stats[compound] = {
                        'base_pace': np.median(y),
                        'deg_slope': 0.05
                    }
        
        if driver_stats:
            race_stats[driver] = driver_stats
            
    return race_stats

if __name__ == "__main__":
    pass
