"""
data_pipeline.py — FastF1 Telemetry Extractor
==============================================
Pulls FP2 (long-run race pace / tire degradation) and FP3 (low-fuel qualifying push laps).
All fallback paces are RELATIVE to the session field — no hardcoded lap times.

Qualifying stats use a "Theoretical Best" approach: the minimum (fastest) sector
time from each driver's top-10%-percentile flying laps, with a small σ derived
from that elite subset.

Race stats use IQR-filtered linear regression on long-run stints, and any
missing compound is filled via field-average compound deltas.
"""

import os
import fastf1
import pandas as pd
import numpy as np

# ── FastF1 cache ───────────────────────────────────────────────────────────
CACHE_DIR = 'fastf1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Session loader ─────────────────────────────────────────────────────────
def get_session(year, race, session_type):
    """Loads and returns a fully-loaded fastf1 session."""
    session = fastf1.get_session(year, race, session_type)
    session.load()
    return session


# ── Lap filter ─────────────────────────────────────────────────────────────
def filter_laps(laps):
    """
    Removes inaccurate laps (in/out laps with poor telemetry) and any lap
    run under Safety Car or Virtual Safety Car conditions.
    pick_accurate() strips out-/in-laps and laps with telemetry gaps.
    pick_track_status('1') keeps only green-flag running.
    """
    return laps.pick_accurate().pick_track_status('1')


# ── Qualifying extractor (Theoretical Best) ────────────────────────────────
def extract_quali_stats(session):
    """
    Builds qualifying sector statistics using a *Theoretical Best Lap* method.

    For each driver:
      1. Take all clean flying laps from the session.
      2. Keep only the top 10 % fastest laps (by total lap time) to isolate
         genuine push-lap pace and discard cool-down / aero-rake runs.
      3. From that elite subset, record the **minimum** sector time for S1,
         S2, S3 (the theoretical best sectors).
      4. σ is derived from the spread within that elite subset so the Monte
         Carlo draws stay tightly bounded around realistic qualifying pace.

    Returns:
        dict[driver] -> {S1_mean, S1_std, S2_mean, S2_std, S3_mean, S3_std}
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    drivers = pd.unique(filtered_laps['Driver'])

    quali_stats = {}

    for driver in drivers:
        driver_laps = filtered_laps.pick_driver(driver)
        driver_laps = driver_laps.dropna(
            subset=['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']
        )

        if len(driver_laps) < 3:
            continue

        # Convert to seconds
        s1 = driver_laps['Sector1Time'].dt.total_seconds().values
        s2 = driver_laps['Sector2Time'].dt.total_seconds().values
        s3 = driver_laps['Sector3Time'].dt.total_seconds().values
        total = driver_laps['LapTime'].dt.total_seconds().values

        # Keep only top-10 % fastest laps (at least 2 laps)
        cutoff = max(2, int(np.ceil(len(total) * 0.10)))
        elite_idx = np.argsort(total)[:cutoff]

        s1_elite = s1[elite_idx]
        s2_elite = s2[elite_idx]
        s3_elite = s3[elite_idx]

        # Theoretical best = minimum of each sector in the elite set
        # σ = std of the elite set (captures natural driver variance on push laps)
        # Floor σ at 0.05 s to avoid degenerate zero-variance draws
        quali_stats[driver] = {
            'S1_mean': float(np.min(s1_elite)),
            'S1_std':  float(max(np.std(s1_elite), 0.05)),
            'S2_mean': float(np.min(s2_elite)),
            'S2_std':  float(max(np.std(s2_elite), 0.05)),
            'S3_mean': float(np.min(s3_elite)),
            'S3_std':  float(max(np.std(s3_elite), 0.05)),
        }

    return quali_stats


# ── Race-pace / degradation extractor ──────────────────────────────────────
def extract_race_pace_and_deg(session):
    """
    Analyses FP2 continuous stints to isolate genuine heavy-fuel long runs.
    Filters out short, low-fuel qualifying simulations by enforcing a strict
    minimum of 7 consecutive laps per continuous stint.
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    drivers = pd.unique(filtered_laps['Driver'])

    COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
    raw_stats = {}          
    field_pace = {}         

    for driver in drivers:
        driver_laps = filtered_laps.pick_driver(driver)
        driver_stats = {}

        # Group by both Stint and Compound to isolate continuous runs out of the pits
        stints = driver_laps.groupby(['Stint', 'Compound'])

        for (stint_num, compound), stint_df in stints:
            if compound not in COMPOUNDS:
                continue

            # Strict threshold: Less than 7 laps means it's a qualifying sim or aborted run
            if len(stint_df) < 7:
                continue

            x = stint_df['TyreLife'].values.astype(float)
            y = stint_df['LapTime'].dt.total_seconds().values

            # IQR outlier removal (traffic, mistakes)
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
            x_clean, y_clean = x[mask], y[mask]

            if len(x_clean) < 5:
                continue

            # Fit line: y = mx + c
            m, c = np.polyfit(x_clean, y_clean, 1)
            m = max(m, 0.01)  

            # If a driver has multiple long runs on one compound, prioritize the more representative (lower base pace) run
            if compound not in driver_stats or c < driver_stats[compound]['base_pace']:
                driver_stats[compound] = {'base_pace': float(c), 'deg_slope': float(m)}
                field_pace.setdefault(compound, []).append(c)

        if driver_stats:
            raw_stats[driver] = driver_stats

    # Pass 2 & 3: Field-average compound delta fallbacks
    field_avg = {c: float(np.median(v)) for c, v in field_pace.items()}

    race_stats = {}
    for driver, stats in raw_stats.items():
        filled = dict(stats)
        known_compounds = list(stats.keys())
        for target in COMPOUNDS:
            if target in filled:
                continue
            for source in known_compounds:
                if source in field_avg and target in field_avg:
                    delta = field_avg[target] - field_avg[source]
                    filled[target] = {
                        'base_pace': stats[source]['base_pace'] + delta,
                        'deg_slope': stats[source]['deg_slope'],
                    }
                    break
        race_stats[driver] = filled

    return race_stats

# ── Real qualifying grid extractor ─────────────────────────────────────────
def extract_real_grid(quali_session):
    """
    Pulls the actual qualifying grid from a loaded Qualifying session.

    Uses session.results which contains 'Abbreviation' and 'GridPosition'
    (or 'Position') for every driver who participated.

    Returns:
        dict[driver_abbreviation] -> int grid position  (1 = pole)
        Returns None if the session has no results.
    """
    try:
        results = quali_session.results
        if results is None or results.empty:
            return None
    except Exception:
        return None

    grid = {}
    for _, row in results.iterrows():
        abbr = row.get('Abbreviation', '')
        pos = row.get('GridPosition')
        
        if pd.isna(pos) or pos == 0:
            pos = row.get('Position')
            
        if abbr and not pd.isna(pos):
            try:
                pos_int = int(pos)
                if pos_int > 0:
                    grid[abbr] = pos_int
            except (ValueError, TypeError):
                continue

    return grid if grid else None


# ── Reliability extractor ──────────────────────────────────────────────────
def extract_reliability_stats(year, current_race):
    """
    Extracts dynamic DNF (Did Not Finish) probabilities per driver.
    Looks at the 5 immediately preceding races. If early in the season,
    wraps around to the end of the previous year.
    
    Returns:
        dict[driver] -> float (per_lap_dnf_probability)
    """
    try:
        current_event = fastf1.get_event(year, current_race)
        current_round = current_event['RoundNumber']
    except Exception:
        # Fallback if event is not found
        return {}

    races_to_fetch = []
    # We want 5 races
    r = current_round - 1
    y = year
    
    while len(races_to_fetch) < 5:
        if r > 0:
            races_to_fetch.append((y, r))
            r -= 1
        else:
            y -= 1
            try:
                prev_schedule = fastf1.get_event_schedule(y)
                # Filter out pre-season testing (usually RoundNumber 0)
                max_round = prev_schedule['RoundNumber'].max()
                r = max_round
            except Exception:
                break # Cannot fetch previous year

    driver_races = {}
    driver_dnfs = {}

    for (fetch_year, fetch_round) in races_to_fetch:
        try:
            # Lightning-fast load without telemetry
            s = fastf1.get_session(fetch_year, fetch_round, 'R')
            s.load(telemetry=False, weather=False, laps=False)
            
            if s.results is None or s.results.empty:
                continue
                
            for _, row in s.results.iterrows():
                abbr = row.get('Abbreviation')
                if not abbr or pd.isna(abbr):
                    continue
                    
                status = str(row.get('Status', '')).strip().lower()
                
                # A driver entered the race
                driver_races[abbr] = driver_races.get(abbr, 0) + 1
                
                # Check if status is a DNF (not finished and not +Laps)
                if status != 'finished' and 'lap' not in status:
                    driver_dnfs[abbr] = driver_dnfs.get(abbr, 0) + 1
                    
        except Exception:
            continue
            
    reliability_stats = {}
    
    # Calculate per-lap probabilities (Assumed 50 laps per race)
    # Floor: 0.001, Cap: 0.008
    grid_total_dnfs = sum(driver_dnfs.values())
    grid_total_races = sum(driver_races.values())
    
    avg_dnf_rate = (grid_total_dnfs / (grid_total_races * 50)) if grid_total_races > 0 else 0.003
    rookie_penalty = min(avg_dnf_rate * 1.2, 0.008)
    
    # Populate the stats
    # We will compute the stats for drivers we found, but return a defaultdict-like behaviour
    # via rookie_penalty for any driver not in this dict during the simulation.
    for driver, races in driver_races.items():
        dnfs = driver_dnfs.get(driver, 0)
        rate = dnfs / (races * 50.0)
        smoothed_rate = np.clip(rate, 0.001, 0.008)
        reliability_stats[driver] = smoothed_rate
        
    # Store the rookie penalty in a special key so the engine can use it for unknown drivers
    reliability_stats['__ROOKIE_FALLBACK__'] = np.clip(rookie_penalty, 0.001, 0.008)

    return reliability_stats


if __name__ == "__main__":
    pass
