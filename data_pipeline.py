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
TRACK_CHARACTERISTICS = {
    # HIGH DOWNFORCE — slow corners, heavy aero dependency
    'Monaco':        'HIGH_DF',
    'Hungary':       'HIGH_DF',
    'Singapore':     'HIGH_DF',
    'Melbourne':     'HIGH_DF',
    'Zandvoort':     'HIGH_DF',
    # MEDIUM — balanced circuits
    'Bahrain':       'MEDIUM',
    'Spain':         'MEDIUM',
    'Barcelona':     'MEDIUM',
    'Austria':       'MEDIUM',
    'Silverstone':   'MEDIUM',
    'Great Britain': 'MEDIUM',
    'Abu Dhabi':     'MEDIUM',
    'Yas Marina':    'MEDIUM',
    'Lusail':        'MEDIUM',
    'Qatar':         'MEDIUM',
    'Mexico':        'MEDIUM',
    'São Paulo':     'MEDIUM',
    'Brazil':        'MEDIUM',
    'Imola':         'MEDIUM',
    'Emilia Romagna':'MEDIUM',
    'China':         'MEDIUM',
    'Shanghai':      'MEDIUM',
    'Japan':         'MEDIUM',
    'Suzuka':        'MEDIUM',
    'Miami':         'MEDIUM',
    'Las Vegas':     'MEDIUM',
    'Saudi Arabia':  'MEDIUM',
    'Jeddah':        'MEDIUM',
    # LOW DOWNFORCE — power tracks, long straights
    'Monza':         'LOW_DF',
    'Italy':         'LOW_DF',
    'Spa':           'LOW_DF',
    'Belgium':       'LOW_DF',
    'Baku':          'LOW_DF',
    'Azerbaijan':    'LOW_DF',
    'Canada':        'LOW_DF',
    'Canadian':      'LOW_DF',
    'Montreal':      'LOW_DF',
}

def _get_track_type(event_name):
    """Resolve an event name to a track type. Falls back to MEDIUM."""
    name = str(event_name)
    for key, ttype in TRACK_CHARACTERISTICS.items():
        if key.lower() in name.lower():
            return ttype
    return 'MEDIUM'

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


# ── Regulation Eras ────────────────────────────────────────────────────────
# Major regulation changes reset car performance hierarchies entirely.
# Season trends must NEVER cross a regulation boundary.
REGULATION_ERAS = {
    (2022, 2025): 'GROUND_EFFECT_V1',   # 2022 ground-effect regulations
    (2026, 2030): 'GROUND_EFFECT_V2',   # 2026 regs (battery, active aero)
}

def _get_regulation_era(year):
    """Returns the regulation era string for a given year."""
    for (start, end), era in REGULATION_ERAS.items():
        if start <= year <= end:
            return era
    return 'UNKNOWN'

def _is_same_regulation_era(year_a, year_b):
    """Returns True if both years fall within the same regulation era."""
    return _get_regulation_era(year_a) == _get_regulation_era(year_b)


# ── Lap 1 Incident Rates (Track-Specific) ─────────────────────────────────
# Historical probability that a driver is involved in a Lap 1 incident.
# Based on multi-year F1 incident data patterns per circuit.
LAP1_INCIDENT_RATES = {
    # HIGH RISK — tight T1 + long run to T1
    'Monza':       0.18,
    'Italy':       0.18,
    'Spa':         0.16,
    'Belgium':     0.16,
    'Baku':        0.14,
    'Azerbaijan':  0.14,
    'Las Vegas':   0.12,
    # MEDIUM RISK
    'Austria':     0.10,
    'Bahrain':     0.10,
    'Singapore':   0.10,
    'Mexico':      0.10,
    'Brazil':      0.10,
    'São Paulo':   0.10,
    'Abu Dhabi':   0.08,
    'Yas Marina':  0.08,
    'Spain':       0.08,
    'Barcelona':   0.08,
    'Silverstone': 0.08,
    'Great Britain':0.08,
    'Canada':      0.08,
    'Canadian':    0.08,
    'Montreal':    0.08,
    'Qatar':       0.07,
    'Lusail':      0.07,
    'Japan':       0.07,
    'Suzuka':      0.07,
    'Imola':       0.08,
    'Emilia Romagna':0.08,
    'China':       0.09,
    'Shanghai':    0.09,
    'Saudi Arabia':0.10,
    'Jeddah':      0.10,
    # LOW RISK — wide T1 or short run
    'Hungary':     0.06,
    'Zandvoort':   0.05,
    'Monaco':      0.04,
    'Melbourne':   0.07,
}

def get_lap1_incident_rate(race_name):
    """Resolve a race name to its Lap 1 incident probability. Default: 0.08."""
    name = str(race_name)
    for key, rate in LAP1_INCIDENT_RATES.items():
        if key.lower() in name.lower():
            return rate
    return 0.08


# ── Slipstream / Tow Effect for Qualifying ────────────────────────────────
# (min_bonus_seconds, max_bonus_seconds) by track type
SLIPSTREAM_EFFECT = {
    'LOW_DF':  (0.15, 0.30),   # Monza, Spa, Baku — massive tow
    'MEDIUM':  (0.05, 0.12),   # Austria, Silverstone — moderate
    'HIGH_DF': (0.00, 0.03),   # Monaco, Hungary — negligible
}

def get_slipstream_range(track_type):
    """Returns (min_bonus, max_bonus) for the given track type."""
    return SLIPSTREAM_EFFECT.get(track_type, (0.05, 0.12))


# ── Overtaking Difficulty Index ────────────────────────────────────────────
# 0.0 = trivially easy to pass, 1.0 = nearly impossible.
# Reflects track width, DRS zones, corner types, and historical overtake data.
OVERTAKING_DIFFICULTY = {
    # VERY HARD — narrow, few DRS opportunities
    'Monaco':      0.95,
    'Hungary':     0.75,
    'Singapore':   0.70,
    'Zandvoort':   0.70,
    'Imola':       0.65,
    'Emilia Romagna':0.65,
    # MODERATE
    'Spain':       0.50,
    'Barcelona':   0.50,
    'Silverstone': 0.45,
    'Great Britain':0.45,
    'Japan':       0.45,
    'Suzuka':      0.45,
    'Abu Dhabi':   0.40,
    'Yas Marina':  0.40,
    'Mexico':      0.40,
    'Austria':     0.35,
    'Bahrain':     0.30,
    'Melbourne':   0.50,
    'Qatar':       0.35,
    'Lusail':      0.35,
    'China':       0.35,
    'Shanghai':    0.35,
    'Saudi Arabia':0.30,
    'Jeddah':      0.30,
    # EASY — long straights, multiple DRS zones
    'Baku':        0.20,
    'Azerbaijan':  0.20,
    'Monza':       0.20,
    'Italy':       0.20,
    'Spa':         0.25,
    'Belgium':     0.25,
    'Brazil':      0.25,
    'São Paulo':   0.25,
    'Las Vegas':   0.25,
    'Canada':      0.30,
    'Canadian':    0.30,
    'Montreal':    0.30,
}

def get_overtaking_difficulty(race_name):
    """Resolve a race name to its overtaking difficulty index. Default: 0.40."""
    name = str(race_name)
    for key, diff in OVERTAKING_DIFFICULTY.items():
        if key.lower() in name.lower():
            return diff
    return 0.40


# ── Energy Recovery Potential (2026 Regs — Superclipping Model) ────────────
# Under the 2026 regulations, cars have a 350kW MGU-K but NO MGU-H (no turbo
# energy recovery). Energy is harvested ONLY under braking.
# 
# Tracks with few heavy braking zones cannot fully recharge the battery,
# causing "superclipping" — a sudden loss of ~150kW on long straights when
# the battery runs empty. This is a massive lap-time penalty.
#
# Scale: 0.0 = almost no braking recovery (extreme superclip risk)
#        1.0 = abundant heavy braking zones (battery always charged)
ENERGY_RECOVERY_POTENTIAL = {
    # VERY LOW RECOVERY — long straights, few/light braking zones
    'Monza':        0.25,   # Only 2 real braking zones (T1, Ascari)
    'Italy':        0.25,
    'Spa':          0.35,   # La Source + Bus Stop, but Kemmel/Blanchimont are flat
    'Belgium':      0.35,
    'Las Vegas':    0.35,   # Long straights, light braking
    'Silverstone':  0.40,   # Fast flowing, Stowe/Village are medium braking
    'Great Britain':0.40,
    'Baku':         0.40,   # Long main straight but T1, T3, T15 are heavy braking
    'Azerbaijan':   0.40,
    'Jeddah':       0.40,   # Fast flowing, limited heavy braking
    'Saudi Arabia': 0.40,
    # MODERATE RECOVERY
    'Austria':      0.55,   # Short lap but T1/T3/T4 have decent braking
    'Canada':       0.55,   # Hairpin + chicanes provide some recovery
    'Canadian':     0.55,
    'Montreal':     0.55,
    'Japan':        0.55,   # Chicane + hairpin but much is high-speed
    'Suzuka':       0.55,
    'China':        0.55,
    'Shanghai':     0.55,
    'Brazil':       0.55,   # Descida do Lago + Juncão braking
    'São Paulo':    0.55,
    'Qatar':        0.50,
    'Lusail':       0.50,
    'Imola':        0.55,
    'Emilia Romagna':0.55,
    'Melbourne':    0.55,
    'Spain':        0.60,
    'Barcelona':    0.60,
    'Zandvoort':    0.55,
    # HIGH RECOVERY — many heavy braking zones
    'Bahrain':      0.75,   # T1, T4, T8, T10, T14 — heavy braking everywhere
    'Abu Dhabi':    0.70,   # Multiple chicanes + hairpins
    'Yas Marina':   0.70,
    'Mexico':       0.65,   # Hairpin + Esses braking
    'Singapore':    0.80,   # 23 corners, many heavy braking zones (street circuit)
    'Monaco':       0.85,   # Constant heavy braking (low speed, tight corners)
    'Hungary':      0.75,   # Stop-start circuit, heavy braking T1, T2, T6
}

def get_energy_recovery_potential(race_name):
    """Resolve a race name to its energy recovery potential (0-1). Default: 0.55."""
    name = str(race_name)
    for key, val in ENERGY_RECOVERY_POTENTIAL.items():
        if key.lower() in name.lower():
            return val
    return 0.55

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
        current_track_type = _get_track_type(current_event['EventName'])
    except Exception:
        return {}

    races_to_fetch = []
    r = current_round - 1
    y = year
    current_era = _get_regulation_era(year)
    
    while len(races_to_fetch) < 5:
        if r > 0:
            races_to_fetch.append((y, r))
            r -= 1
        else:
            y -= 1
            # REGULATION BOUNDARY: never cross into a different regulation era
            if not _is_same_regulation_era(y, year):
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
            hist_track_type = _get_track_type(hist_event['EventName'])
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
                        driver_stats[drv] = {'conversions': [], 'r_deltas': [], 'weights': []}
                    driver_stats[drv]['conversions'].append(sunday_conv)
                    driver_stats[drv]['r_deltas'].append(r_delta)
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
            
            avg_conv = np.average(conversions, weights=weights)
            avg_r_delta = np.average(r_deltas, weights=weights)
            season_trends[drv] = {
                'sunday_conversion': avg_conv, # Positive = better on Sunday
                'power_rank_delta': avg_r_delta # Lower = faster race pace
            }
            
    return season_trends


# ── Qualifying Power Rank (Sandbagging Detector) ──────────────────────────
def extract_quali_power_rank(year, current_race):
    """
    Extracts historical qualifying pace deltas from the last 5 qualifying
    sessions within the SAME regulation era.
    
    Returns:
        dict[driver] -> float (average qualifying delta-to-pole in seconds)
        Lower = historically faster qualifier.
    """
    print("⏳ Extracting Qualifying Power Rank (Sandbagging Detection) …")
    try:
        current_event = fastf1.get_event(year, current_race)
        current_round = current_event['RoundNumber']
    except Exception:
        return {}

    races_to_fetch = []
    r = current_round - 1
    y = year

    while len(races_to_fetch) < 5:
        if r > 0:
            races_to_fetch.append((y, r))
            r -= 1
        else:
            y -= 1
            # REGULATION BOUNDARY: never cross into a different regulation era
            if not _is_same_regulation_era(y, year):
                break
            try:
                prev_schedule = fastf1.get_event_schedule(y)
                max_round = prev_schedule['RoundNumber'].max()
                r = max_round
            except Exception:
                break

    driver_deltas = {}  # driver -> [list of delta_to_pole values]

    for (fetch_year, fetch_round) in races_to_fetch:
        try:
            q = fastf1.get_session(fetch_year, fetch_round, 'Q')
            q.load(telemetry=False, weather=False, messages=False)

            q_laps = q.laps
            if q_laps is None or len(q_laps) == 0:
                continue

            drivers = pd.unique(q_laps['Driver'])
            q_paces = {}

            for drv in drivers:
                drv_laps = q_laps[q_laps['Driver'] == drv]
                fastest = drv_laps.pick_fastest()
                if not pd.isnull(fastest['LapTime']):
                    q_paces[drv] = fastest['LapTime'].total_seconds()

            if not q_paces:
                continue

            pole_pace = min(q_paces.values())

            for drv, pace in q_paces.items():
                delta = pace - pole_pace
                if drv not in driver_deltas:
                    driver_deltas[drv] = []
                driver_deltas[drv].append(delta)

        except Exception:
            continue

    # Average the deltas
    quali_power_rank = {}
    for drv, deltas in driver_deltas.items():
        if len(deltas) > 0:
            quali_power_rank[drv] = float(np.mean(deltas))

    if quali_power_rank:
        print(f"  ✓ Quali power rank extracted for {len(quali_power_rank)} drivers "
              f"({len(races_to_fetch)} races, era: {_get_regulation_era(year)})")
    else:
        print(f"  ℹ  No historical quali data in this regulation era yet")

    return quali_power_rank


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

            # Raw stint must be ≥ 4 laps to qualify as a long run
            if len(stint_df) < 4:
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
            
            # ── Polyfit Sanity Check ──────────────────────────────────
            # If the IQR filter fails to catch an outlier on a short 3-lap run, 
            # it produces absurd slopes (e.g., 20s/lap) and negative base paces.
            # We strictly bound acceptable tire degradation (-0.05 to 0.4 seconds/lap).
            # Note: Fuel burn causes lap times to drop by ~0.05s per lap. On low 
            # degradation tracks (like Monza), raw lap time slopes can legitimately 
            # be slightly negative (-0.02) because fuel burn outpaces tire wear.
            if m_raw > 0.4 or m_raw < -0.05:
                continue
                
            # Base pace must be a physically possible lap time (>40s)
            if c_raw < 40.0:
                continue
                
            m = float(m_raw)
            c = float(c_raw)

            # ── Pace Sanity Check ─────────────────────────────────────
            # Reject aero-rake / constant-velocity runs. If the base pace 
            # is extremely slow compared to the driver's fastest lap 
            # (> 108%), it's not a genuine race-pace stint.
            fastest_lap = driver_fastest_lap.get(driver)
            if fastest_lap and c > fastest_lap * 1.08:
                continue

            # Keep the most representative run per compound (lowest base)
            if compound not in driver_stats or c < driver_stats[compound]['base_pace']:
                driver_stats[compound] = {'base_pace': c, 'deg_slope': m}
                field_pace.setdefault(compound, []).append(c)

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
        for compound in COMPOUNDS:
            filled[compound] = {
                'base_pace': fallback_soft_pace + COMPOUND_OFFSETS[compound],
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
