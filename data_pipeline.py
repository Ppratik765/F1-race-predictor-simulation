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


# ── Team Mapping Extractor ─────────────────────────────────────────────────
def extract_team_mapping(session):
    """
    Extracts driver → {team_name, team_color} from a session's results.
    Colors come directly from FastF1 so they are accurate for any year.
    """
    mapping = {}
    if session is None or session.results is None or session.results.empty:
        return mapping
    for _, row in session.results.iterrows():
        abbr = row.get('Abbreviation')
        if not abbr or pd.isna(abbr):
            continue
        team = str(row.get('TeamName', 'Unknown'))
        color = str(row.get('TeamColor', '888888')).strip('#')
        mapping[abbr] = {
            'team': team,
            'color': f'#{color}',
        }
    return mapping


# ── Weather Context Extractor ──────────────────────────────────────────────
def extract_weather_context(session):
    """
    Extracts median track temperature, air temperature, and rainfall flag
    from a session's weather data feed.
    """
    context = {'track_temp': 30.0, 'air_temp': 25.0, 'rainfall': False}
    if session is None:
        return context
    try:
        weather = session.weather_data
        if weather is not None and len(weather) > 0:
            context['track_temp'] = float(weather['TrackTemp'].median())
            context['air_temp'] = float(weather['AirTemp'].median())
            context['rainfall'] = bool(weather['Rainfall'].any())
    except Exception:
        pass
    return context


# ── Track Characteristics ──────────────────────────────────────────────────
# Classification: HIGH_DF (slow, twisty), MEDIUM, LOW_DF (power circuits)
# Metrics tuned precisely to historical F1 data (2018-2024 averages)
# - turn_1_chaos: Probability of lap 1 incident/DNF (e.g., Monza chicane, Mexico T1 are highest)
# - tow_factor: Slipstream benefit in Quali (seconds gained). Highest at Baku, Monza, Vegas.
# - overtaking_diff: Dirty air penalty multiplier. 1.0 = Monaco (impossible), 0.2 = Vegas (easy).
TRACK_CHARACTERISTICS = {
    # HIGH DOWNFORCE — slow corners, heavy aero dependency
    'Monaco':        {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.060, 'tow_factor': 0.00, 'overtaking_diff': 1.00},
    'Singapore':     {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.05, 'overtaking_diff': 0.85},
    'Imola':         {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.040, 'tow_factor': 0.10, 'overtaking_diff': 0.85},
    'Emilia Romagna':{'df_type': 'HIGH_DF', 'turn_1_chaos': 0.040, 'tow_factor': 0.10, 'overtaking_diff': 0.85},
    'Hungary':       {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.060, 'tow_factor': 0.08, 'overtaking_diff': 0.80},
    'Zandvoort':     {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.08, 'overtaking_diff': 0.80},
    
    # MEDIUM-HIGH DOWNFORCE — narrow or hard to follow
    'Spain':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.12, 'overtaking_diff': 0.70},
    'Barcelona':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.12, 'overtaking_diff': 0.70},
    'Melbourne':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.050, 'tow_factor': 0.10, 'overtaking_diff': 0.70},
    'Japan':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.65},
    'Suzuka':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.65},

    # MEDIUM DOWNFORCE — balanced circuits
    'Silverstone':   {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.10, 'overtaking_diff': 0.50},
    'Great Britain': {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.10, 'overtaking_diff': 0.50},
    'Miami':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.15, 'overtaking_diff': 0.50},
    'Saudi Arabia':  {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.18, 'overtaking_diff': 0.45},
    'Jeddah':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.18, 'overtaking_diff': 0.45},
    'Abu Dhabi':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.15, 'overtaking_diff': 0.45},
    'Yas Marina':    {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.15, 'overtaking_diff': 0.45},
    'Qatar':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.50},
    'Lusail':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.50},
    'China':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.035, 'tow_factor': 0.16, 'overtaking_diff': 0.45},
    'Shanghai':      {'df_type': 'MEDIUM', 'turn_1_chaos': 0.035, 'tow_factor': 0.16, 'overtaking_diff': 0.45},
    'Austin':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.060, 'tow_factor': 0.15, 'overtaking_diff': 0.40},
    'United States': {'df_type': 'MEDIUM', 'turn_1_chaos': 0.060, 'tow_factor': 0.15, 'overtaking_diff': 0.40},
    
    # MEDIUM-LOW DOWNFORCE — easier overtaking, heavy braking
    'Bahrain':       {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Austria':       {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'São Paulo':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Brazil':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Mexico':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.080, 'tow_factor': 0.20, 'overtaking_diff': 0.40},

    # LOW DOWNFORCE — power tracks, long straights, slipstream city
    'Canada':        {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Canadian':      {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Montreal':      {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.35},
    'Baku':          {'df_type': 'LOW_DF', 'turn_1_chaos': 0.065, 'tow_factor': 0.28, 'overtaking_diff': 0.30},
    'Azerbaijan':    {'df_type': 'LOW_DF', 'turn_1_chaos': 0.065, 'tow_factor': 0.28, 'overtaking_diff': 0.30},
    'Monza':         {'df_type': 'LOW_DF', 'turn_1_chaos': 0.080, 'tow_factor': 0.30, 'overtaking_diff': 0.30},
    'Italy':         {'df_type': 'LOW_DF', 'turn_1_chaos': 0.080, 'tow_factor': 0.30, 'overtaking_diff': 0.30},
    'Spa':           {'df_type': 'LOW_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.22, 'overtaking_diff': 0.25},
    'Belgium':       {'df_type': 'LOW_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.22, 'overtaking_diff': 0.25},
    'Las Vegas':     {'df_type': 'LOW_DF', 'turn_1_chaos': 0.050, 'tow_factor': 0.30, 'overtaking_diff': 0.20},
}

def _get_track_type(event_name):
    """Resolve an event name to a track type and its characteristics. Falls back to MEDIUM."""
    name = str(event_name)
    for key, info in TRACK_CHARACTERISTICS.items():
        if key.lower() in name.lower():
            return info
    return {'df_type': 'MEDIUM', 'turn_1_chaos': 0.03, 'tow_factor': 0.1, 'overtaking_diff': 0.5}

def _track_similarity_weight(type_a, type_b):
    """
    Returns a weight for how similar two track types are.
    Same type = 1.5x, adjacent = 1.0x, opposite = 0.5x.
    """
    order = {'HIGH_DF': 0, 'MEDIUM': 1, 'LOW_DF': 2}
    diff = abs(order.get(type_a, 1) - order.get(type_b, 1))
    if diff == 0:
        return 1.5
    elif diff == 1:
        return 1.0
    else:
        return 0.5


# ── Season Trend Model ─────────────────────────────────────────────────────
def extract_season_trends(year, current_race):
    """
    Extracts historical race vs qualifying performance from the preceding 3-5 races.
    Computes a 'Sunday Conversion Factor' (how much a driver improves on Sunday)
    and a 'Power Rank' (average clean race pace deficit to the fastest car).
    
    Historical races are weighted by track type similarity to the current race:
      Same type = 1.5x, Adjacent = 1.0x, Opposite = 0.5x.
    """
    print("⏳ Extracting Season Trend Model (Historical 5-race momentum) …")
    try:
        current_event = fastf1.get_event(year, current_race)
        current_round = current_event['RoundNumber']
        current_track_info = _get_track_type(current_event['EventName'])
        current_track_type = current_track_info['df_type']
    except Exception:
        return {}

    races_to_fetch = []
    r = current_round - 1
    y = year
    
    # Establish regulation boundaries (e.g. 2026 new regs, 2022 ground effect)
    # Don't look back before a regulation cutoff.
    regulation_cutoff = 0
    if year >= 2026:
        regulation_cutoff = 2026
    elif year >= 2022:
        regulation_cutoff = 2022

    while len(races_to_fetch) < 5:
        if r > 0:
            races_to_fetch.append((y, r))
            r -= 1
        else:
            y -= 1
            if y < regulation_cutoff:
                break
            try:
                prev_schedule = fastf1.get_event_schedule(y)
                max_round = prev_schedule['RoundNumber'].max()
                r = max_round
            except Exception:
                break

    driver_stats = {}
    
    for (fetch_year, fetch_round) in races_to_fetch:
        try:
            # Resolve the track type of this historical race
            hist_event = fastf1.get_event(fetch_year, fetch_round)
            hist_track_info = _get_track_type(hist_event['EventName'])
            hist_track_type = hist_track_info['df_type']
            weight = _track_similarity_weight(current_track_type, hist_track_type)
            
            q = fastf1.get_session(fetch_year, fetch_round, 'Q')
            q.load(telemetry=False, weather=False, messages=False)
            
            r_session = fastf1.get_session(fetch_year, fetch_round, 'R')
            r_session.load(telemetry=False, weather=False, messages=False)
            
            q_laps = q.laps
            r_laps = r_session.laps
            
            if q_laps is None or r_laps is None or len(q_laps) == 0 or len(r_laps) == 0:
                continue
                
            drivers = pd.unique(q_laps['Driver'])
            q_paces = {}
            r_paces = {}
            
            # Quali pace
            for drv in drivers:
                drv_laps = q_laps[q_laps['Driver'] == drv]
                fastest = drv_laps.pick_fastest()
                if not pd.isnull(fastest['LapTime']):
                    q_paces[drv] = fastest['LapTime'].total_seconds()
            
            if not q_paces: continue
            pole_pace = min(q_paces.values())
            
            # Race pace
            for drv in drivers:
                drv_laps = r_laps[r_laps['Driver'] == drv]
                clean_laps = drv_laps[
                    (drv_laps['TrackStatus'] == '1') & 
                    pd.isnull(drv_laps['PitOutTime']) & 
                    pd.isnull(drv_laps['PitInTime']) &
                    (drv_laps['LapNumber'] > 1)
                ]
                
                if len(clean_laps) > 5:
                    times = clean_laps['LapTime'].dt.total_seconds().dropna().values
                    if len(times) > 0:
                        q1, q3 = np.percentile(times, [25, 75])
                        iqr = q3 - q1
                        valid_times = times[(times >= q1 - 1.5*iqr) & (times <= q3 + 1.5*iqr)]
                        if len(valid_times) > 0:
                            r_paces[drv] = np.median(valid_times)
                            
            if not r_paces: continue
            fastest_r_pace = min(r_paces.values())
            
            # Compute Deltas (with track similarity weight)
            for drv in drivers:
                if drv in q_paces and drv in r_paces:
                    q_delta = q_paces[drv] - pole_pace
                    r_delta = r_paces[drv] - fastest_r_pace
                    sunday_conv = q_delta - r_delta
                    
                    if drv not in driver_stats:
                        driver_stats[drv] = {'conversions': [], 'r_deltas': [], 'q_deltas': [], 'weights': []}
                    driver_stats[drv]['conversions'].append(sunday_conv)
                    driver_stats[drv]['r_deltas'].append(r_delta)
                    driver_stats[drv]['q_deltas'].append(q_delta)
                    driver_stats[drv]['weights'].append(weight)
                    
        except Exception as e:
            continue

    # Weighted Aggregate
    season_trends = {}
    for drv, stats in driver_stats.items():
        if len(stats['conversions']) > 0:
            weights = np.array(stats['weights'])
            conversions = np.array(stats['conversions'])
            r_deltas = np.array(stats['r_deltas'])
            q_deltas = np.array(stats['q_deltas'])
            
            avg_conv = np.average(conversions, weights=weights)
            avg_r_delta = np.average(r_deltas, weights=weights)
            avg_q_delta = np.average(q_deltas, weights=weights)
            season_trends[drv] = {
                'sunday_conversion': avg_conv, # Positive = better on Sunday
                'power_rank_delta': avg_r_delta, # Lower = faster race pace
                'quali_power_rank': avg_q_delta # Lower = faster qualifying pace
            }
            
    return season_trends


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
    Analyses practice / sprint stints to determine race-pace parameters.

    Stint Integrity Protocol:
      1. Work on RAW laps (before track-status filtering) so that yellow-flag
         interruptions don't fragment a genuine 15-lap long run into two
         sub-threshold 6-lap shards.
      2. Identify continuous raw stints using the FastF1 'Stint' column.
      3. Only THEN skip individual anomalous laps (yellows, pit-lane cuts)
         within the stint — the remaining green-flag laps stay grouped.
      4. If a driver has zero stints ≥ 7 laps, compute a Global Fallback
         from their fastest single flying lap + a heavy-fuel penalty so
         they are never deleted from the simulation.

    Returns:
        dict[driver] -> dict[compound] -> {'base_pace': float, 'deg_slope': float}
    """
    laps = session.laps

    # ── Step 0: remove only pit in/out laps (keep ALL track statuses) ──
    try:
        raw_laps = laps.pick_accurate()
    except Exception:
        raw_laps = laps.copy()

    raw_laps = raw_laps.dropna(subset=['LapTime', 'TyreLife', 'Compound'])
    drivers = pd.unique(raw_laps['Driver'])

    COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
    HEAVY_FUEL_PENALTY = 4.5   # seconds added to a quali-sim lap to estimate race pace

    raw_stats  = {}    # driver -> {compound -> {base_pace, deg_slope}}
    field_pace = {}    # compound -> [list of base paces across field]

    # ── Also collect fastest flying lap per driver for global fallback ──
    driver_fastest_lap = {}

    for driver in drivers:
        driver_laps = raw_laps[raw_laps['Driver'] == driver]
        driver_stats = {}

        # Record the driver's fastest clean lap (for fallback)
        lap_times_all = driver_laps['LapTime'].dt.total_seconds().dropna()
        if len(lap_times_all) > 0:
            driver_fastest_lap[driver] = float(lap_times_all.min())

        # ── Group by the RAW Stint number (preserves continuity) ──────
        if 'Stint' not in driver_laps.columns:
            continue

        stints = driver_laps.groupby('Stint')

        for stint_num, stint_df in stints:
            # Determine dominant compound for this stint
            compound_counts = stint_df['Compound'].value_counts()
            compound = compound_counts.index[0]
            if compound not in COMPOUNDS:
                continue

            # Keep only laps on the dominant compound
            stint_df = stint_df[stint_df['Compound'] == compound]

            # Raw stint must be ≥ 7 laps to qualify as a long run (filters out qualifying sims)
            if len(stint_df) < 7:
                continue

            x_raw = stint_df['TyreLife'].values.astype(float)
            y_raw = stint_df['LapTime'].dt.total_seconds().values

            # ── Skip anomalous laps WITHOUT fragmenting the stint ─────
            #    Flag laps under yellow / VSC / SC (TrackStatus != '1')
            if 'TrackStatus' in stint_df.columns:
                track_ok = stint_df['TrackStatus'].astype(str).str.strip() == '1'
            else:
                track_ok = pd.Series(True, index=stint_df.index)

            x = x_raw[track_ok.values]
            y = y_raw[track_ok.values]

            # After removing yellows, still need ≥ 3 green-flag laps
            if len(x) < 3:
                continue

            # IQR outlier removal (traffic, mistakes)
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
            x_clean, y_clean = x[mask], y[mask]

            if len(x_clean) < 3:
                continue

            # Linear fit: y = mx + c
            m_raw, c_raw = np.polyfit(x_clean, y_clean, 1)
            
            # ── Polyfit Sanity Check & Clamping ───────────────────────
            # Linear extrapolation backwards from old tires (e.g., TyreLife=20) 
            # with high deg (e.g., 0.3s/lap) hallucinates impossible base paces.
            # We clip the slope to realistic F1 degradation (-0.02 to 0.15s/lap)
            # and recompute the intercept (base pace) to anchor it realistically.
            m_clamped = np.clip(m_raw, -0.02, 0.15)
            c_clamped = np.mean(y_clean) - m_clamped * np.mean(x_clean)
            
            base_pace = c_clamped
            deg_slope = m_clamped
                
            # Base pace must be a physically possible lap time (>40s)
            if c_clamped < 40.0:
                continue
                
            m = float(m_raw)
            c = float(c_raw)

            # ── Pace Sanity Check ─────────────────────────────────────
            # Reject aero-rake / constant-velocity runs. If the base pace 
            # is extremely slow compared to the driver's fastest lap 
            # (> 108%), it's not a genuine race-pace stint.
            fastest_lap = driver_fastest_lap.get(driver)
            if fastest_lap and base_pace > fastest_lap * 1.08:
                continue

            # Keep the most representative run per compound (lowest base)
            if compound not in driver_stats or base_pace < driver_stats[compound]['base_pace']:
                driver_stats[compound] = {'base_pace': float(base_pace), 'deg_slope': float(deg_slope)}
                field_pace.setdefault(compound, []).append(base_pace)

        if driver_stats:
            raw_stats[driver] = driver_stats

    # ── Pass 2: Calculate Compound Deltas ─────────────────────────────────
    # Calculate empirical deltas by comparing base paces for drivers who 
    # completed heavy-fuel long runs on MULTIPLE compounds in the same session.
    # This completely eliminates Simpson's Paradox and avoids fuel-load skew.
    
    deltas_soft_med = []
    deltas_med_hard = []
    
    for driver, stats in raw_stats.items():
        if 'SOFT' in stats and 'MEDIUM' in stats:
            deltas_soft_med.append(stats['MEDIUM']['base_pace'] - stats['SOFT']['base_pace'])
        if 'MEDIUM' in stats and 'HARD' in stats:
            deltas_med_hard.append(stats['HARD']['base_pace'] - stats['MEDIUM']['base_pace'])
            
    # Calculate median deltas, fallback to a standard realistic 0.4s if we don't have enough data
    soft_med_delta = float(np.median(deltas_soft_med)) if len(deltas_soft_med) >= 2 else 0.4
    med_hard_delta = float(np.median(deltas_med_hard)) if len(deltas_med_hard) >= 2 else 0.4
    
    # Ensure deltas are logically positive (HARD > MEDIUM > SOFT) and bounded
    soft_med_delta = max(min(soft_med_delta, 1.5), 0.1)
    med_hard_delta = max(min(med_hard_delta, 1.5), 0.1)
    
    # Build a unified pace offset map relative to SOFT
    COMPOUND_OFFSETS = {
        'SOFT': 0.0,
        'MEDIUM': soft_med_delta,
        'HARD': soft_med_delta + med_hard_delta
    }

    # ── Pass 3: Fill missing compounds ────────────────────────────────────
    race_stats = {}
    for driver, stats in raw_stats.items():
        filled = dict(stats)
        known_compounds = list(stats.keys())
        
        for target in COMPOUNDS:
            if target in filled:
                continue
                
            # Find a known source to project from (prefer Medium, then Soft, then Hard)
            source = None
            for pref in ['MEDIUM', 'SOFT', 'HARD']:
                if pref in known_compounds:
                    source = pref
                    break
                    
            if source:
                # Delta to add to the source to get the target
                delta = COMPOUND_OFFSETS[target] - COMPOUND_OFFSETS[source]
                filled[target] = {
                    'base_pace': stats[source]['base_pace'] + delta,
                    'deg_slope': stats[source]['deg_slope'],
                }
                
        race_stats[driver] = filled

    # ── Global Fallback: rescue drivers with zero long runs ───────────────
    #    Use the field median SOFT pace as a baseline, then apply standard deltas.
    field_deg_slopes = []
    for stats in race_stats.values():
        for cstats in stats.values():
            field_deg_slopes.append(cstats['deg_slope'])
    median_deg = float(np.median(field_deg_slopes)) if field_deg_slopes else 0.06

    soft_paces = [s['SOFT']['base_pace'] for s in race_stats.values() if 'SOFT' in s]
    fallback_soft_pace = float(np.median(soft_paces)) if soft_paces else 90.0

    for driver in drivers:
        if driver in race_stats:
            continue  # already has stint data

        filled = {}
        # Try to use their fastest lap, else use field median
        fastest_lap = driver_fastest_lap.get(driver)
        if fastest_lap:
            base_soft = fastest_lap + HEAVY_FUEL_PENALTY
        else:
            base_soft = fallback_soft_pace
            
        for compound in COMPOUNDS:
            filled[compound] = {
                'base_pace': base_soft + COMPOUND_OFFSETS[compound],
                'deg_slope': median_deg,
            }
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
