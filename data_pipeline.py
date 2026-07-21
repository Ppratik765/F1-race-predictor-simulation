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
import threading
import itertools
import sys
import time

class Spinner:
    def __init__(self, message="Fetching data..."):
        self.message = message
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{self.message} {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write(f"\r✓ {self.message} Done!{' ' * 20}\n")
        sys.stdout.flush()

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
# - sc_probability / vsc_probability: P(at least one SC/VSC deployment during the race),
#   sourced from F1.com's own "Need to Know" per-race statistics (trailing ~5-10 race window)
#   where available, otherwise estimated from circuit geometry (wall-lined street circuits =
#   high, modern run-off-heavy circuits = low). THESE ARE APPROXIMATIONS — update them each
#   season from F1.com's pre-race "Need to Know" articles for the sharpest calibration.
# - pit_loss_base: typical total pit-lane time loss in seconds (entry + stationary + exit),
#   again sourced from F1.com where available.
TRACK_CHARACTERISTICS = {
    # HIGH DOWNFORCE — slow corners, heavy aero dependency
    'Monaco':        {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.060, 'tow_factor': 0.00, 'overtaking_diff': 0.95, 'sc_probability': 0.55, 'vsc_probability': 0.20, 'pit_loss_base': 21.5},
    'Singapore':     {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.05, 'overtaking_diff': 0.88, 'sc_probability': 0.95, 'vsc_probability': 0.35, 'pit_loss_base': 29.0},
    'Imola':         {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.040, 'tow_factor': 0.10, 'overtaking_diff': 0.75, 'sc_probability': 0.45, 'vsc_probability': 0.25, 'pit_loss_base': 24.0},
    'Emilia Romagna':{'df_type': 'HIGH_DF', 'turn_1_chaos': 0.040, 'tow_factor': 0.10, 'overtaking_diff': 0.75, 'sc_probability': 0.45, 'vsc_probability': 0.25, 'pit_loss_base': 24.0},
    'Hungary':       {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.060, 'tow_factor': 0.08, 'overtaking_diff': 0.85, 'sc_probability': 0.25, 'vsc_probability': 0.25, 'pit_loss_base': 20.6},
    'Zandvoort':     {'df_type': 'HIGH_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.08, 'overtaking_diff': 0.82, 'sc_probability': 0.50, 'vsc_probability': 0.30, 'pit_loss_base': 22.0},

    # MEDIUM-HIGH DOWNFORCE — narrow or hard to follow
    'Spain':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.12, 'overtaking_diff': 0.70, 'sc_probability': 0.25, 'vsc_probability': 0.20, 'pit_loss_base': 21.0},
    'Barcelona':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.12, 'overtaking_diff': 0.70, 'sc_probability': 0.25, 'vsc_probability': 0.20, 'pit_loss_base': 21.0},
    'Melbourne':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.050, 'tow_factor': 0.10, 'overtaking_diff': 0.65, 'sc_probability': 0.67, 'vsc_probability': 0.50, 'pit_loss_base': 20.1},
    'Japan':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.60, 'sc_probability': 0.30, 'vsc_probability': 0.25, 'pit_loss_base': 21.5},
    'Suzuka':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.60, 'sc_probability': 0.30, 'vsc_probability': 0.25, 'pit_loss_base': 21.5},

    # MEDIUM DOWNFORCE — balanced circuits
    'Silverstone':   {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.10, 'overtaking_diff': 0.55, 'sc_probability': 0.40, 'vsc_probability': 0.30, 'pit_loss_base': 21.0},
    'Great Britain': {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.10, 'overtaking_diff': 0.55, 'sc_probability': 0.40, 'vsc_probability': 0.30, 'pit_loss_base': 21.0},
    'Miami':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.040, 'tow_factor': 0.15, 'overtaking_diff': 0.58, 'sc_probability': 0.60, 'vsc_probability': 0.35, 'pit_loss_base': 21.0},
    'Saudi Arabia':  {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.18, 'overtaking_diff': 0.60, 'sc_probability': 0.55, 'vsc_probability': 0.30, 'pit_loss_base': 22.5},
    'Jeddah':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.18, 'overtaking_diff': 0.60, 'sc_probability': 0.55, 'vsc_probability': 0.30, 'pit_loss_base': 22.5},
    'Abu Dhabi':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.15, 'overtaking_diff': 0.65, 'sc_probability': 0.15, 'vsc_probability': 0.15, 'pit_loss_base': 21.5},
    'Yas Marina':    {'df_type': 'MEDIUM', 'turn_1_chaos': 0.030, 'tow_factor': 0.15, 'overtaking_diff': 0.65, 'sc_probability': 0.15, 'vsc_probability': 0.15, 'pit_loss_base': 21.5},
    'Qatar':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.70, 'sc_probability': 0.35, 'vsc_probability': 0.25, 'pit_loss_base': 24.5},
    'Lusail':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.12, 'overtaking_diff': 0.70, 'sc_probability': 0.35, 'vsc_probability': 0.25, 'pit_loss_base': 24.5},
    'China':         {'df_type': 'MEDIUM', 'turn_1_chaos': 0.035, 'tow_factor': 0.16, 'overtaking_diff': 0.58, 'sc_probability': 0.30, 'vsc_probability': 0.20, 'pit_loss_base': 22.0},
    'Shanghai':      {'df_type': 'MEDIUM', 'turn_1_chaos': 0.035, 'tow_factor': 0.16, 'overtaking_diff': 0.58, 'sc_probability': 0.30, 'vsc_probability': 0.20, 'pit_loss_base': 22.0},
    'Austin':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.060, 'tow_factor': 0.15, 'overtaking_diff': 0.55, 'sc_probability': 0.40, 'vsc_probability': 0.25, 'pit_loss_base': 19.5},
    'United States': {'df_type': 'MEDIUM', 'turn_1_chaos': 0.060, 'tow_factor': 0.15, 'overtaking_diff': 0.55, 'sc_probability': 0.40, 'vsc_probability': 0.25, 'pit_loss_base': 19.5},

    # MEDIUM-LOW DOWNFORCE — easier overtaking, heavy braking
    'Bahrain':       {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.55, 'sc_probability': 0.20, 'vsc_probability': 0.20, 'pit_loss_base': 22.0},
    'Austria':       {'df_type': 'MEDIUM', 'turn_1_chaos': 0.055, 'tow_factor': 0.15, 'overtaking_diff': 0.35, 'sc_probability': 0.35, 'vsc_probability': 0.25, 'pit_loss_base': 19.5},
    'São Paulo':     {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.45, 'sc_probability': 0.50, 'vsc_probability': 0.30, 'pit_loss_base': 20.5},
    'Brazil':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.045, 'tow_factor': 0.15, 'overtaking_diff': 0.45, 'sc_probability': 0.50, 'vsc_probability': 0.30, 'pit_loss_base': 20.5},
    'Mexico':        {'df_type': 'MEDIUM', 'turn_1_chaos': 0.080, 'tow_factor': 0.20, 'overtaking_diff': 0.40, 'sc_probability': 0.45, 'vsc_probability': 0.25, 'pit_loss_base': 22.0},

    # LOW DOWNFORCE — power tracks, long straights, slipstream city
    'Canada':        {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.40, 'sc_probability': 0.65, 'vsc_probability': 0.30, 'pit_loss_base': 20.5},
    'Canadian':      {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.40, 'sc_probability': 0.65, 'vsc_probability': 0.30, 'pit_loss_base': 20.5},
    'Montreal':      {'df_type': 'LOW_DF', 'turn_1_chaos': 0.035, 'tow_factor': 0.15, 'overtaking_diff': 0.40, 'sc_probability': 0.65, 'vsc_probability': 0.30, 'pit_loss_base': 20.5},
    'Baku':          {'df_type': 'LOW_DF', 'turn_1_chaos': 0.065, 'tow_factor': 0.28, 'overtaking_diff': 0.68, 'sc_probability': 0.65, 'vsc_probability': 0.30, 'pit_loss_base': 20.0},
    'Azerbaijan':    {'df_type': 'LOW_DF', 'turn_1_chaos': 0.065, 'tow_factor': 0.28, 'overtaking_diff': 0.68, 'sc_probability': 0.65, 'vsc_probability': 0.30, 'pit_loss_base': 20.0},
    'Monza':         {'df_type': 'LOW_DF', 'turn_1_chaos': 0.080, 'tow_factor': 0.30, 'overtaking_diff': 0.75, 'sc_probability': 0.50, 'vsc_probability': 0.38, 'pit_loss_base': 23.7},
    'Italy':         {'df_type': 'LOW_DF', 'turn_1_chaos': 0.080, 'tow_factor': 0.30, 'overtaking_diff': 0.75, 'sc_probability': 0.50, 'vsc_probability': 0.38, 'pit_loss_base': 23.7},
    'Spa':           {'df_type': 'LOW_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.22, 'overtaking_diff': 0.70, 'sc_probability': 0.55, 'vsc_probability': 0.30, 'pit_loss_base': 21.0},
    'Belgium':       {'df_type': 'LOW_DF', 'turn_1_chaos': 0.070, 'tow_factor': 0.22, 'overtaking_diff': 0.70, 'sc_probability': 0.55, 'vsc_probability': 0.30, 'pit_loss_base': 21.0},
    'Las Vegas':     {'df_type': 'LOW_DF', 'turn_1_chaos': 0.050, 'tow_factor': 0.30, 'overtaking_diff': 0.72, 'sc_probability': 0.40, 'vsc_probability': 0.25, 'pit_loss_base': 20.0},
}

_DEFAULT_TRACK_INFO = {
    'df_type': 'MEDIUM', 'turn_1_chaos': 0.03, 'tow_factor': 0.1, 'overtaking_diff': 0.5,
    'sc_probability': 0.35, 'vsc_probability': 0.25, 'pit_loss_base': 22.0,
}

def _get_track_type(event_name):
    """Resolve an event name to a track type and its characteristics. Falls back to MEDIUM."""
    name = str(event_name).lower()
    
    # Map common Grand Prix adjectives/names to keys in TRACK_CHARACTERISTICS
    name_mappings = {
        'belgian': 'belgium',
        'italian': 'italy',
        'brazilian': 'brazil',
        'japanese': 'japan',
        'spanish': 'spain',
        'mexican': 'mexico',
        'canadian': 'canada',
        'hungarian': 'hungary',
        'austrian': 'austria',
        'british': 'silverstone',
        'emilia': 'imola',
        'monégasque': 'monaco',
        'dutch': 'zandvoort',
        'saudi arabian': 'saudi arabia',
    }
    for adj, country in name_mappings.items():
        if adj in name:
            name += " " + country

    for key, info in TRACK_CHARACTERISTICS.items():
        if key.lower() in name:
            return info
    return dict(_DEFAULT_TRACK_INFO)

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

    Returns:
        (season_trends, team_trends) — season_trends is dict[driver] -> {...},
        team_trends is dict[team_name] -> {...} (aggregated across that team's
        drivers), used as the "known current form" prior for sandbag detection.
    """
    try:
        current_event = fastf1.get_event(year, current_race)
        current_round = current_event['RoundNumber']
        current_track_info = _get_track_type(current_event['EventName'])
        current_track_type = current_track_info['df_type']
    except Exception:
        return {}, {}

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
            drv_team = {}

            # Quali pace
            for drv in drivers:
                drv_laps = q_laps[q_laps['Driver'] == drv]
                fastest = drv_laps.pick_fastest()
                if not pd.isnull(fastest['LapTime']):
                    q_paces[drv] = fastest['LapTime'].total_seconds()
                if 'Team' in drv_laps.columns and len(drv_laps) > 0:
                    team_val = drv_laps['Team'].iloc[0]
                    if pd.notna(team_val):
                        drv_team[drv] = str(team_val)
            
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
                        driver_stats[drv] = {'conversions': [], 'r_deltas': [], 'q_deltas': [], 'weights': [], 'team': drv_team.get(drv)}
                    driver_stats[drv]['conversions'].append(sunday_conv)
                    driver_stats[drv]['r_deltas'].append(r_delta)
                    driver_stats[drv]['q_deltas'].append(q_delta)
                    driver_stats[drv]['weights'].append(weight)
                    if driver_stats[drv].get('team') is None:
                        driver_stats[drv]['team'] = drv_team.get(drv)

        except Exception as e:
            continue

    # Weighted Aggregate (per driver)
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
                'quali_power_rank': avg_q_delta, # Lower = faster qualifying pace
                'team': stats.get('team'),
            }

    # ── Team-level aggregate ───────────────────────────────────────────
    # This is the "known current form" prior used for sandbagging detection:
    # if a team's FP/Quali pace this weekend is wildly worse than what both
    # of their cars have actually been doing on-track in recent races, that's
    # a signal the session pace isn't representative (fuel loads, programme,
    # sandbagging) rather than a genuine drop in competitiveness.
    team_groups = {}
    for drv, stats in season_trends.items():
        team = stats.get('team')
        if not team:
            continue
        team_groups.setdefault(team, []).append(stats)

    team_trends = {}
    for team, entries in team_groups.items():
        team_trends[team] = {
            'power_rank_delta': float(np.mean([e['power_rank_delta'] for e in entries])),
            'quali_power_rank': float(np.mean([e['quali_power_rank'] for e in entries])),
            'sunday_conversion': float(np.mean([e['sunday_conversion'] for e in entries])),
            'num_drivers': len(entries),
        }

    return season_trends, team_trends


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
# ── Tow / Slipstream Detection ──────────────────────────────────────────────
def detect_tow_assisted_laps(session, track_info=None):
    """
    Flags a driver's fastest lap as tow-assisted if its speed-trap reading is
    an outlier relative to THAT DRIVER'S OWN other laps in the same session
    (using intra-driver z-score) or extremely high relative to the field.
    """
    tow_flags = {}
    try:
        filtered_laps = session.laps.pick_quicklaps(1.07)
    except Exception:
        filtered_laps = session.laps

    if track_info is None or track_info.get('tow_factor', 0.10) < 0.16:
        return tow_flags

    trap_cols = [c for c in ['SpeedST', 'SpeedFL', 'SpeedI1', 'SpeedI2'] if c in filtered_laps.columns]
    if not trap_cols:
        return tow_flags

    import numpy as np

    for driver in pd.unique(filtered_laps['Driver']):
        driver_laps = filtered_laps.pick_drivers(driver).dropna(subset=['LapTime'])
        if len(driver_laps) < 2:
            continue

        times = driver_laps['LapTime'].dt.total_seconds().values
        fastest_idx = int(np.argmin(times))
        fastest_time = times[fastest_idx]
        
        baseline_mask = (times <= fastest_time * 1.04)
        baseline_mask[fastest_idx] = False
        
        if not np.any(baseline_mask):
            continue
            
        max_tow_penalty = 0.0
        
        # 1. Multi-Trap Delta Evaluation (Speed Traps)
        for col in trap_cols:
            if col not in driver_laps.columns:
                continue
            
            speeds = driver_laps[col].values.astype(float)
            fast_lap_speed = speeds[fastest_idx]
            
            if np.isnan(fast_lap_speed):
                continue
                
            baseline_speeds = speeds[baseline_mask]
            baseline_speeds = baseline_speeds[~np.isnan(baseline_speeds)]
            
            if len(baseline_speeds) == 0:
                continue
                
            baseline_median = np.median(baseline_speeds)
            speed_spike = fast_lap_speed - baseline_median
            
            if speed_spike > 4.0:
                penalty = min(speed_spike * 0.03, 0.45)
                max_tow_penalty = max(max_tow_penalty, penalty)
                
        # 2. Sector-Time Delta Evaluation (S1 & S3 Fallback to bypass speed-trap blindness)
        s2_gain = 0.0
        if 'Sector2Time' in driver_laps.columns:
            s2_times = driver_laps['Sector2Time'].dt.total_seconds().values.astype(float)
            fast_s2 = s2_times[fastest_idx]
            baseline_s2 = s2_times[baseline_mask]
            baseline_s2 = baseline_s2[~np.isnan(baseline_s2)]
            if len(baseline_s2) > 0:
                s2_gain = np.median(baseline_s2) - fast_s2

        for sec_col in ['Sector1Time', 'Sector3Time']:
            if sec_col not in driver_laps.columns:
                continue
                
            sec_times = driver_laps[sec_col].dt.total_seconds().values.astype(float)
            fast_sec_time = sec_times[fastest_idx]
            
            if np.isnan(fast_sec_time):
                continue
                
            baseline_sec_times = sec_times[baseline_mask]
            baseline_sec_times = baseline_sec_times[~np.isnan(baseline_sec_times)]
            
            if len(baseline_sec_times) == 0:
                continue
                
            baseline_sec_median = np.median(baseline_sec_times)
            sec_delta = baseline_sec_median - fast_sec_time  # Gained time (lower is faster/better)
            
            # Gaining >0.35s in a straight-line sector is a strong physics-based tow indicator,
            # but only if it isn't simply track evolution (which would improve the twisty Sector 2 even more).
            if sec_delta > 0.35 and sec_delta > (s2_gain - 0.10):
                penalty = min(sec_delta, 0.45)
                max_tow_penalty = max(max_tow_penalty, penalty)
                
        if max_tow_penalty > 0.0:
            tow_flags[driver] = float(max_tow_penalty)

    return tow_flags


def extract_quali_stats(session, track_info=None):
    """
    Builds qualifying sector statistics using a *Theoretical Best Lap* method.

    For each driver:
      1. Take all clean flying laps from the session.
      2. Keep only the top 10 % fastest laps (by total lap time) to isolate
         genuine push-lap pace and discard cool-down / aero-rake runs.
      3. From that elite subset, record the **minimum** sector time for S1,
         S2, S3 (the theoretical best sectors) — EXCLUDING any lap flagged
         as tow-assisted (see detect_tow_assisted_laps) wherever an
         un-flagged alternative exists in the elite set.
      4. σ is derived from the spread within that elite subset so the Monte
         Carlo draws stay tightly bounded around realistic qualifying pace.
      5. If the driver's overall fastest lap was tow-flagged, partially
         discount the theoretical best pace by the estimated tow inflation
         (spread evenly across the three sectors, since we can't attribute
         the tow to a specific sector from speed-trap data alone).

    Returns:
        (quali_stats, tow_flags) — quali_stats is dict[driver] -> {S1_mean,
        S1_std, S2_mean, S2_std, S3_mean, S3_std}; tow_flags is dict[driver]
        -> estimated seconds of lap-time inflation from a detected tow.
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    drivers = pd.unique(filtered_laps['Driver'])

    tow_flags = detect_tow_assisted_laps(session, track_info=track_info)

    speed_col = next((c for c in ['SpeedST', 'SpeedFL', 'SpeedI2', 'SpeedI1']
                       if c in filtered_laps.columns), None)

    quali_stats = {}

    for driver in drivers:
        driver_laps = filtered_laps.pick_drivers(driver)
        driver_laps = driver_laps.dropna(
            subset=['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']
        )

        if len(driver_laps) < 2:
            continue

        # Convert to seconds
        s1 = driver_laps['Sector1Time'].dt.total_seconds().values
        s2 = driver_laps['Sector2Time'].dt.total_seconds().values
        s3 = driver_laps['Sector3Time'].dt.total_seconds().values
        total = driver_laps['LapTime'].dt.total_seconds().values
        speeds = driver_laps[speed_col].values.astype(float) if speed_col else None

        # Keep only top-10 % fastest laps (at least 2 laps)
        cutoff = max(2, int(np.ceil(len(total) * 0.10)))
        elite_idx = np.argsort(total)[:cutoff]

        s1_elite = s1[elite_idx]
        s2_elite = s2[elite_idx]
        s3_elite = s3[elite_idx]

        # ── Exclude tow-outlier laps from the elite pool where possible ──
        # A big draft shows up as an outlier-high speed-trap reading versus
        # this SAME driver's other elite-set laps. If we can drop it and
        # still have laps left, do so rather than let it set the "theoretical
        # best" for a sector it didn't earn on pure pace.
        if speeds is not None:
            elite_speeds = speeds[elite_idx]
            valid = ~np.isnan(elite_speeds)
            if valid.sum() >= 2:
                e_mean = np.mean(elite_speeds[valid])
                e_std = max(np.std(elite_speeds[valid]), 1.0)
                z_scores = np.where(valid, (elite_speeds - e_mean) / e_std, 0.0)
                is_outlier = z_scores > 1.75
                if np.any(is_outlier) and np.any(~is_outlier):
                    keep = ~is_outlier
                    s1_elite, s2_elite, s3_elite = s1_elite[keep], s2_elite[keep], s3_elite[keep]

        # Theoretical best = minimum of each sector in the (tow-filtered) elite set
        # σ = std of the elite set (captures natural driver variance on push laps)
        # Floor σ at 0.05 s to avoid degenerate zero-variance draws
        s1_mean = float(np.min(s1_elite))
        s2_mean = float(np.min(s2_elite))
        s3_mean = float(np.min(s3_elite))

        # If the driver's single fastest lap overall was tow-flagged, that
        # inflation likely still leaked into whichever sector holds the
        # straight — partially discount it since we can't isolate the exact
        # sector from speed-trap data alone.
        if driver in tow_flags:
            per_sector = tow_flags[driver] / 3.0
            s1_mean += per_sector
            s2_mean += per_sector
            s3_mean += per_sector

        quali_stats[driver] = {
            'S1_mean': s1_mean,
            'S1_std':  float(max(np.std(s1_elite), 0.05)),
            'S2_mean': s2_mean,
            'S2_std':  float(max(np.std(s2_elite), 0.05)),
            'S3_mean': s3_mean,
            'S3_std':  float(max(np.std(s3_elite), 0.05)),
        }

    return quali_stats, tow_flags


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
        driver_stint_pool = {}  # compound -> list of {base_pace, deg_slope, n_laps}

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

            # ── Accumulate this stint (don't cherry-pick yet) ──────────
            # NOTE: we deliberately do NOT keep "whichever stint is fastest"
            # here. That was the previous behaviour, and it's exactly the
            # bug that lets a backmarker team's short, deliberately
            # light-fuel long-run make their race pace look better than it
            # really is: if you always keep the fastest of several stints,
            # you're systematically biased toward whichever stint happened
            # to be on the least fuel. Instead we keep every valid stint and
            # combine them below, weighted toward longer runs — a 15+ lap
            # stint is much harder to fake on light fuel than a 7-9 lap one,
            # so it should count for more.
            n_laps = len(x_clean)
            driver_stint_pool.setdefault(compound, []).append(
                {'base_pace': float(base_pace), 'deg_slope': float(deg_slope), 'n_laps': n_laps}
            )

        for compound, entries in driver_stint_pool.items():
            n_arr = np.array([e['n_laps'] for e in entries], dtype=float)
            pace_arr = np.array([e['base_pace'] for e in entries])
            deg_arr = np.array([e['deg_slope'] for e in entries])
            # Credibility weighting: a stint of 12+ laps is treated as a
            # confirmed race-fuel-equivalent run and counts double per lap;
            # shorter stints (7-11 laps) still count, just proportionally less.
            weights = np.where(n_arr >= 12, n_arr * 2.0, n_arr)

            combined_pace = float(np.average(pace_arr, weights=weights))
            combined_deg = float(np.average(deg_arr, weights=weights))
            total_laps = int(n_arr.sum())
            longest_stint = int(n_arr.max())

            driver_stats[compound] = {
                'base_pace': combined_pace,
                'deg_slope': combined_deg,
                'sample_laps': total_laps,
                'longest_stint': longest_stint,
                'num_stints': len(entries),
            }
            field_pace.setdefault(compound, []).append(combined_pace)

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
                    'sample_laps': 0,  # projected from another compound, not directly observed
                    'longest_stint': 0,
                    'num_stints': 0,
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
                'sample_laps': 0,
                'longest_stint': 0,
                'num_stints': 0,
            }
        race_stats[driver] = filled

    return race_stats

# ── Real qualifying grid extractor ─────────────────────────────────────────
def extract_real_grid(quali_session):
    """
    Pulls the actual qualifying grid from a loaded Qualifying or Race session.

    Uses session.results which contains 'Abbreviation' and 'GridPosition'
    (or 'Position') for every driver who participated. FastF1 sets GridPosition=0 
    for Pit Lane starters.

    Returns:
        tuple(dict[driver_abbreviation] -> int, list[pitlane_starters])
        Returns (None, []) if the session has no results.
    """
    try:
        results = quali_session.results
        if results is None or results.empty:
            return None, []
    except Exception:
        return None, []

    grid = {}
    pitlane_starters = []
    
    for _, row in results.iterrows():
        abbr = row.get('Abbreviation', '')
        if not abbr:
            continue
            
        pos = row.get('GridPosition')
        
        # If GridPosition is 0 or 0.0, they are explicitly a Pit Lane starter in FastF1
        if pd.notna(pos) and float(pos) == 0.0:
            pitlane_starters.append(abbr)
            continue
            
        # Fallback to normal position if GridPosition is NaN
        if pd.isna(pos):
            pos = row.get('Position')
            
        if pd.notna(pos):
            try:
                pos_int = int(float(pos))
                if pos_int > 0:
                    grid[abbr] = pos_int
            except (ValueError, TypeError):
                continue
                
    # Assign pitlane starters to virtual slots at the back of the grid to keep NumPy arrays valid
    if grid and pitlane_starters:
        current_max = max(grid.values())
        for abbr in pitlane_starters:
            current_max += 1
            grid[abbr] = current_max

    return (grid if grid else None), pitlane_starters


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
    driver_mechanical = {}
    driver_collisions = {}

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
                    if any(term in status for term in ['accident', 'collision', 'spun', 'crash']):
                        driver_collisions[abbr] = driver_collisions.get(abbr, 0) + 1
                    else:
                        driver_mechanical[abbr] = driver_mechanical.get(abbr, 0) + 1
                    
        except Exception:
            continue
            
    reliability_stats = {}
    
    # Calculate continuous per-lap probabilities (Assumed 50 laps per race)
    grid_total_mechanical = sum(driver_mechanical.values())
    grid_total_collisions = sum(driver_collisions.values())
    grid_total_races = sum(driver_races.values())
    
    avg_dnf_rate = ((grid_total_mechanical + grid_total_collisions) / (grid_total_races * 50)) if grid_total_races > 0 else 0.003
    rookie_penalty = avg_dnf_rate * 1.2
    
    # Populate the stats with continuous float assignment logic
    for driver, races in driver_races.items():
        mech_rate = driver_mechanical.get(driver, 0) / (races * 50.0)
        coll_rate = driver_collisions.get(driver, 0) / (races * 50.0)
        
        continuous_weight = mech_rate + coll_rate
        # Remove tiered bucketing (no np.clip), use tiny fallback to prevent log(0) in PMF sampling
        reliability_stats[driver] = max(continuous_weight, 0.0001)
        
    # Store the rookie penalty in a special key so the engine can use it for unknown drivers
    reliability_stats['__ROOKIE_FALLBACK__'] = max(rookie_penalty, 0.0001)

    return reliability_stats


# ── Straight-line Speed / Power-Unit Deployment Index ──────────────────────
def extract_speed_metrics(session):
    """
    Extracts each driver's straight-line speed advantage (or deficit) relative
    to the field, using FastF1's built-in speed-trap columns (SpeedST, SpeedFL,
    SpeedI1, SpeedI2 — already present on session.laps, no telemetry pull needed).

    Under 2026 regs, energy deployment / battery management is a much bigger
    differentiator than outright downforce on power-sensitive circuits, so a
    car that is consistently faster in a straight line than its cornering pace
    would suggest (the "Kimi at Spa" case) shows up here as a positive index.

    Returns:
        dict[driver] -> float power_index (z-score vs field, positive = faster
        in a straight line than the field average). 0.0 if no data.
    """
    if session is None or session.laps is None or len(session.laps) == 0:
        return {}

    laps = session.laps
    speed_cols = [c for c in ['SpeedST', 'SpeedFL', 'SpeedI1', 'SpeedI2'] if c in laps.columns]
    if not speed_cols:
        return {}

    drivers = pd.unique(laps['Driver'])
    driver_top_speed = {}

    for drv in drivers:
        drv_laps = laps[laps['Driver'] == drv]
        vals = []
        for col in speed_cols:
            col_vals = drv_laps[col].dropna()
            if len(col_vals) > 0:
                # Use the 90th percentile rather than max to avoid tow-inflated outliers
                vals.append(float(np.percentile(col_vals, 90)))
        if vals:
            driver_top_speed[drv] = float(np.mean(vals))

    if len(driver_top_speed) < 3:
        return {}

    speeds = np.array(list(driver_top_speed.values()))
    field_mean = np.mean(speeds)
    field_std = max(np.std(speeds), 0.5)  # floor to avoid div-by-zero on freak-identical data

    power_index = {
        drv: float((spd - field_mean) / field_std)
        for drv, spd in driver_top_speed.items()
    }
    return power_index


# ── Team Pit Stop Performance Extractor ────────────────────────────────────
def extract_pitstop_stats(year, current_race, track_pit_loss_base=22.0):
    """
    Extracts team-specific pit-lane time loss from the 5 preceding races, using
    each stop's (PitOutTime - PitInTime) as the total pit-lane loss for that
    stop. Values are normalised against each historical race's own field median
    (since pit-lane length/speed-limit varies enormously by circuit) then
    re-anchored onto the CURRENT circuit's expected pit loss.

    Also estimates a per-team "botched stop" probability from how often a
    team's stop was a statistical outlier (>1.8x the IQR above their own
    median) in that window — a crude proxy for pit-crew reliability.

    Returns:
        dict[team_name] -> {'pit_loss': float seconds, 'botch_prob': float}
        Always includes a '__FIELD__' fallback key for unmapped teams.
    """
    try:
        current_event = fastf1.get_event(year, current_race)
        current_round = current_event['RoundNumber']
    except Exception:
        return {'__FIELD__': {'pit_loss': track_pit_loss_base, 'botch_prob': 0.06}}

    races_to_fetch = []
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
                r = prev_schedule['RoundNumber'].max()
            except Exception:
                break

    team_deltas = {}  # team -> list of (stop_time - race_median)

    for (fetch_year, fetch_round) in races_to_fetch:
        try:
            s = fastf1.get_session(fetch_year, fetch_round, 'R')
            s.load(telemetry=False, weather=False, messages=False)
            laps = s.laps
            if laps is None or len(laps) == 0 or 'PitInTime' not in laps.columns:
                continue

            race_stop_times = []
            stops_this_race = []  # (team, stop_time)

            for drv in pd.unique(laps['Driver']):
                drv_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
                team_val = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns and len(drv_laps) else None
                pit_in_rows = drv_laps[drv_laps['PitInTime'].notna()]

                for idx in pit_in_rows.index:
                    pit_in_t = drv_laps.loc[idx, 'PitInTime']
                    later = drv_laps[drv_laps.index > idx]
                    out_rows = later[later['PitOutTime'].notna()]
                    if len(out_rows) == 0:
                        continue
                    pit_out_t = out_rows.iloc[0]['PitOutTime']
                    try:
                        stop_time = (pit_out_t - pit_in_t).total_seconds()
                    except Exception:
                        continue
                    # Sanity bound: real stops are 15-45s; anything else is a
                    # red flag / long pit-lane closure artifact, not a normal stop
                    if 15.0 <= stop_time <= 45.0 and team_val:
                        race_stop_times.append(stop_time)
                        stops_this_race.append((str(team_val), stop_time))

            if len(race_stop_times) < 4:
                continue
            race_median = float(np.median(race_stop_times))
            race_q1, race_q3 = np.percentile(race_stop_times, [25, 75])
            race_iqr = max(race_q3 - race_q1, 0.3)

            for team, stop_time in stops_this_race:
                team_deltas.setdefault(team, []).append(stop_time - race_median)
                # Track outliers for botch-rate estimate
                team_deltas.setdefault(team + '__outlier_flags', [])
                is_outlier = stop_time > (race_q3 + 1.8 * race_iqr)
                team_deltas[team + '__outlier_flags'].append(is_outlier)

        except Exception:
            continue

    pitstop_stats = {}
    all_deltas = []
    for key, vals in team_deltas.items():
        if key.endswith('__outlier_flags'):
            continue
        if len(vals) >= 3:
            all_deltas.extend(vals)

    for team in [k for k in team_deltas.keys() if not k.endswith('__outlier_flags')]:
        deltas = team_deltas[team]
        flags = team_deltas.get(team + '__outlier_flags', [])
        if len(deltas) < 3:
            continue
        team_delta = float(np.median(deltas))
        # Clamp: no team should be modeled as more than +/-1.2s off the field
        team_delta = float(np.clip(team_delta, -1.2, 1.2))
        botch_prob = float(np.clip((sum(flags) / len(flags)) if flags else 0.05, 0.02, 0.25))
        pitstop_stats[team] = {
            'pit_loss': track_pit_loss_base + team_delta,
            'botch_prob': botch_prob,
        }

    pitstop_stats['__FIELD__'] = {'pit_loss': track_pit_loss_base, 'botch_prob': 0.06}
    return pitstop_stats


# ── Team-Level Sandbagging Correction ──────────────────────────────────────
def apply_team_sandbag_correction(quali_stats, race_stats, team_mapping, team_trends,
                                   session_label='FP3', blend=0.5, threshold=0.35):
    """
    Detects a systematic team-level anomaly: BOTH of a team's cars showing a
    session pace much worse than that team's actual recent-race form would
    predict. A single driver having a scruffy session is normal variance; both
    cars of a leading team being 6th/8th tenths off is much more likely to be
    fuel loads / programme work / deliberate sandbagging than a genuine form
    collapse — so we blend the extracted pace partway back toward the team's
    known recent form rather than trusting the raw session data at face value.

    This is a heuristic, not a certainty — it only fires when the anomaly is
    large AND consistent across both cars of the same team, and it only ever
    partially corrects (blend=0.5 by default), never fully overrides the data.

    Args:
        quali_stats: dict[driver] -> {S1_mean, ...} from extract_quali_stats (mutated copy returned)
        race_stats: dict[driver] -> {compound -> {...}} from extract_race_pace_and_deg (mutated copy returned)
        team_mapping: dict[driver] -> {'team': str, 'color': str}
        team_trends: dict[team] -> {'power_rank_delta': ..., 'quali_power_rank': ...} from extract_season_trends
        session_label: which session fed race_stats/quali_stats ('FP1' is noisiest, trusted least)
        blend: how strongly to pull toward the team's known form (0=ignore session, 1=trust session fully)
        threshold: minimum seconds of anomaly (vs field-best) before a correction fires

    Returns:
        (corrected_quali_stats, corrected_race_stats, flags) where flags is a
        list of human-readable strings describing any corrections applied.
    """
    flags = []
    if not team_mapping or not team_trends:
        return quali_stats, race_stats, flags

    # FP1 pace is the least trustworthy (setup work, fuel sims, new-part testing)
    # so we blend harder toward the prior there than for FP2/FP3/Quali data.
    session_trust = {'FP1': 0.35, 'FP2': 0.55, 'FP3': 0.65, 'Q': 0.8, 'SQ': 0.6, 'S': 0.55}
    effective_blend = blend * session_trust.get(session_label, 0.5) / 0.55

    # Group drivers by team
    teams = {}
    for drv, info in team_mapping.items():
        teams.setdefault(info.get('team', 'Unknown'), []).append(drv)

    # ── Quali sandbagging check ────────────────────────────────────────
    if quali_stats:
        totals = {d: (s['S1_mean'] + s['S2_mean'] + s['S3_mean']) for d, s in quali_stats.items()}
        if totals:
            best = min(totals.values())
            for team, drvs in teams.items():
                team_drvs = [d for d in drvs if d in totals]
                if len(team_drvs) < 2 or team not in team_trends:
                    continue
                actual_gaps = [totals[d] - best for d in team_drvs]
                expected_gap = team_trends[team]['quali_power_rank']
                # Both cars anomalously slow vs their known form?
                if min(actual_gaps) - expected_gap > threshold:
                    correction = (min(actual_gaps) - expected_gap) * effective_blend
                    per_sector = correction / 3.0
                    for d in team_drvs:
                        quali_stats[d]['S1_mean'] -= per_sector
                        quali_stats[d]['S2_mean'] -= per_sector
                        quali_stats[d]['S3_mean'] -= per_sector
                    flags.append(
                        f"Quali: {team} both cars ~{min(actual_gaps):.2f}s off pole vs. "
                        f"expected ~{expected_gap:.2f}s from recent form — pace blended "
                        f"{correction:.2f}s faster (possible sandbagging/fuel-load effect)."
                    )

    # ── Race-pace sandbagging check (SOFT compound as the common reference) ──
    if race_stats:
        soft_paces = {d: s['SOFT']['base_pace'] for d, s in race_stats.items() if 'SOFT' in s}
        if soft_paces:
            best_r = min(soft_paces.values())
            for team, drvs in teams.items():
                team_drvs = [d for d in drvs if d in soft_paces]
                if len(team_drvs) < 2 or team not in team_trends:
                    continue
                actual_gaps = [soft_paces[d] - best_r for d in team_drvs]
                expected_gap = team_trends[team]['power_rank_delta']
                if min(actual_gaps) - expected_gap > threshold:
                    correction = (min(actual_gaps) - expected_gap) * effective_blend
                    for d in team_drvs:
                        for compound in race_stats[d]:
                            race_stats[d][compound]['base_pace'] -= correction
                    flags.append(
                        f"Race pace: {team} both cars ~{min(actual_gaps):.2f}s off the fastest "
                        f"long-run vs. expected ~{expected_gap:.2f}s from recent form — pace "
                        f"blended {correction:.2f}s faster (possible sandbagging)."
                    )

    return quali_stats, race_stats, flags


if __name__ == "__main__":
    pass