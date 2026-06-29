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
    Analyses FP2 long runs to determine:
      • base_pace  (intercept of linear fit, i.e. lap time at TyreLife = 0)
      • deg_slope  (seconds lost per additional lap on the tyre)

    Missing-compound fallback logic:
      After computing every driver × compound pair that has data, we build a
      field-average compound-delta matrix.  Any driver who is missing a compound
      gets their pace estimated as:
          known_pace + field_delta(known → missing)
      This eliminates all hardcoded lap-time constants.

    Returns:
        dict[driver] -> dict[compound] -> {'base_pace': float, 'deg_slope': float}
    """
    laps = session.laps
    filtered_laps = filter_laps(laps)
    drivers = pd.unique(filtered_laps['Driver'])

    COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

    # ── Pass 1: compute raw stats for every driver × compound with data ──
    raw_stats = {}          # driver -> {compound -> {base_pace, deg_slope}}
    field_pace = {}         # compound -> [list of base paces across field]

    for driver in drivers:
        driver_laps = filtered_laps.pick_driver(driver)
        driver_stats = {}

        for compound in COMPOUNDS:
            compound_laps = driver_laps[driver_laps['Compound'] == compound]
            if len(compound_laps) < 5:
                continue

            x = compound_laps['TyreLife'].values.astype(float)
            y = compound_laps['LapTime'].dt.total_seconds().values

            # IQR outlier removal
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
            x_clean, y_clean = x[mask], y[mask]

            if len(x_clean) < 4:
                continue

            m, c = np.polyfit(x_clean, y_clean, 1)
            m = max(m, 0.01)  # floor: degradation can't be negative

            driver_stats[compound] = {'base_pace': float(c), 'deg_slope': float(m)}
            field_pace.setdefault(compound, []).append(c)

        if driver_stats:
            raw_stats[driver] = driver_stats

    # ── Pass 2: build field-average compound deltas ──────────────────────
    field_avg = {c: float(np.median(v)) for c, v in field_pace.items()}

    # ── Pass 3: fill missing compounds per driver using deltas ───────────
    race_stats = {}
    for driver, stats in raw_stats.items():
        filled = dict(stats)  # start with what we have

        known_compounds = list(stats.keys())
        for target in COMPOUNDS:
            if target in filled:
                continue
            # Find a known compound to derive from
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
        # GridPosition is the actual starting grid; Position is quali result
        pos = row.get('GridPosition', row.get('Position', None))
        if abbr and pos is not None:
            try:
                pos_int = int(pos)
                if pos_int > 0:
                    grid[abbr] = pos_int
            except (ValueError, TypeError):
                continue

    return grid if grid else None


if __name__ == "__main__":
    pass
