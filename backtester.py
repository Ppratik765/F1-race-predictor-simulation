"""
backtester.py — Smart Orchestrator & Human-Readable Reporter
==============================================================
Detects which sessions are available for a given race weekend and
adapts its behaviour accordingly:

  • Future race  → prints countdown to weekend / data availability
  • FP2+FP3 exist, but Qualifying hasn't happened → runs quali prediction
  • Qualifying results exist → uses REAL grid for race simulation
  • All sessions exist → full backtest with real grid
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import datetime
import argparse
import numpy as np
import fastf1

from data_pipeline import (
    get_session,
    extract_quali_stats,
    extract_race_pace_and_deg,
    extract_real_grid,
    extract_reliability_stats,
    extract_season_trends,
    extract_team_mapping,
    extract_weather_context,
    _get_track_type,
)
from quali_engine import run_quali_sim
from race_engine import run_race_sim


# ── Pretty-print helpers ──────────────────────────────────────────────────

def _pct(value):
    """Format a 0-1 float as a clean percentage string like '82.4%'."""
    return f"{value * 100:.1f}%"


def _print_table(title, headers, rows):
    """Renders a Markdown-style ASCII grid table to stdout."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    print(f"\n{'=' * len(sep)}")
    print(f"  {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(hdr)
    print(sep.replace('-', '='))
    for row in rows:
        line = "|" + "|".join(
            f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)
        ) + "|"
        print(line)
    print(sep)


def _fmt_delta(td):
    """Format a timedelta into a human-readable string like '4 days, 3 hours'."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "already passed"
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0 and days == 0:
        parts.append(f"{minutes} min")
    return ", ".join(parts) if parts else "< 1 min"


# ── Session availability checker ─────────────────────────────────────────

def _check_event_schedule(year, race):
    """
    Returns the event schedule row and session dates.
    Uses fastf1.get_event() to look up the race weekend.
    Returns (event, is_future, delta_to_weekend, delta_to_race) or None.
    """
    try:
        event = fastf1.get_event(year, race)
    except Exception:
        return None

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Find the earliest session date (FP1 / Session1)
    weekend_start = None
    race_time = None
    for col in ['Session1DateUtc', 'Session2DateUtc', 'Session3DateUtc',
                'Session4DateUtc', 'Session5DateUtc']:
        val = event.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                dt = pd.Timestamp(val)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                if weekend_start is None or dt < weekend_start:
                    weekend_start = dt
            except Exception:
                pass

    # Race is typically Session5
    for col in ['Session5DateUtc', 'Session4DateUtc']:
        val = event.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                dt = pd.Timestamp(val)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                race_time = dt
                break
            except Exception:
                pass

    if weekend_start is None:
        return None

    now_ts = pd.Timestamp(now)
    is_future = now_ts < weekend_start
    delta_weekend = weekend_start - now_ts if is_future else None
    delta_race = (race_time - now_ts) if race_time and now_ts < race_time else None

    return event, is_future, delta_weekend, delta_race


def _try_load_session(year, race, session_type):
    """Attempts to load a session. Returns the session or None."""
    try:
        s = get_session(year, race, session_type)
        # Check if the session actually has lap data
        if s.laps is not None and len(s.laps) > 0:
            return s
    except Exception:
        pass
    return None


# -- Main ------------------------------------------------------------------

def main(year=2025, race='Monza', num_iterations=200_000):
    import pandas as pd  # local import for Timestamp usage

    print(f"\n{'-' * 60}")
    print(f"  F1 Monte Carlo Simulation  ·  {year} {race}")
    print(f"  Iterations: {num_iterations:,}")
    print(f"{'-' * 60}")

    t0 = time.time()

    # ── Step 0: Check if this is a future race ────────────────────────
    schedule_info = _check_event_schedule(year, race)
    if schedule_info is not None:
        event, is_future, delta_weekend, delta_race = schedule_info
        if is_future:
            event_name = event.get('EventName', race)
            print(f"\n🏁 {event_name} {year}")
            print(f"   Race weekend commences in: {_fmt_delta(delta_weekend)}")
            if delta_race:
                print(f"   Race prediction available in: {_fmt_delta(delta_race)}")
                print(f"   (Requires FP2/FP3 or Sprint telemetry to be available)")
            print(f"\n   ⏳ No telemetry data available yet. Check back after")
            print(f"      practice sessions have been completed.")
            return

    # ── Step 1: Load practice data (FP2 for race pace, FP3 for quali) ─
    print("\n⏳ Loading telemetry via FastF1 …")

    session_fp2 = _try_load_session(year, race, 'FP2')
    session_fp3 = _try_load_session(year, race, 'FP3')

    # Fallback: if FP3 doesn't exist (sprint weekend), try Sprint Qualifying
    if session_fp3 is None:
        print("   ℹ  FP3 not found — trying Sprint Qualifying …")
        session_fp3 = _try_load_session(year, race, 'SQ')
    if session_fp3 is None:
        print("   ℹ  Sprint Qualifying not found — trying FP1 …")
        session_fp3 = _try_load_session(year, race, 'FP1')

    if session_fp2 is None:
        print("   ℹ  FP2 not found — trying Sprint Race for long-run pace …")
        session_fp2 = _try_load_session(year, race, 'S')
    if session_fp2 is None:
        print("   ℹ  Sprint Race not found — trying FP1 …")
        session_fp2 = _try_load_session(year, race, 'FP1')

    if session_fp2 is None:
        print("✗ No race-pace representative session found (FP2/S/FP1).")
        print("  Hint: Data may not yet be uploaded for this weekend.")
        print("  Try again closer to the race weekend (typically Friday afternoon onwards).")
        return

    if session_fp3 is None:
        print("✗ No qualifying-representative session found (FP3/SQ/FP1).")
        print("  Hint: Data may not yet be uploaded for this weekend.")
        print("  Try again closer to the race weekend (typically Saturday afternoon onwards).")
        return

    print("⏳ Extracting driver statistics …")
    quali_stats = extract_quali_stats(session_fp3)
    race_stats  = extract_race_pace_and_deg(session_fp2)
    reliability_stats = extract_reliability_stats(year, race)
    season_trends = extract_season_trends(year, race)
    team_mapping = extract_team_mapping(session_fp2)
    weather_context = extract_weather_context(session_fp2)
    
    # Get track specifics (df_type, turn_1_chaos, tow_factor, overtaking_diff)
    try:
        event_info = fastf1.get_event(year, race)
        track_info = _get_track_type(event_info['EventName'])
    except Exception:
        track_info = None

    if not quali_stats or not race_stats:
        print("✗ Could not extract stats. Check session data.")
        return

    t1 = time.time()
    print(f"✓ Data ready in {t1 - t0:.1f} s")
    
    # Print weather conditions
    if weather_context:
        rain_str = "WET" if weather_context.get('rainfall', False) else "DRY"
        print(f"\n   Weather: Track {weather_context['track_temp']:.0f} C | Air {weather_context['air_temp']:.0f} C | {rain_str}")

    if season_trends:
        trend_rows = sorted(season_trends.items(), key=lambda x: x[1]['power_rank_delta'])
        _print_table(
            "SEASON POWER RANKINGS (Last 5 Races)",
            ["Rank", "Driver", "Avg Race Pace Deficit", "Sunday Conversion Factor"],
            [
                [str(idx + 1), drv, f"+{stats['power_rank_delta']:.3f}s", f"{stats['sunday_conversion']:+.3f}s"]
                for idx, (drv, stats) in enumerate(trend_rows)
            ],
        )

    # ── Step 2: Determine grid — real or simulated ────────────────────
    real_grid = None
    pole_probs = {}
    grid_source = "SIMULATED"

    # Try loading actual Qualifying results
    print("\n⏳ Checking for real Qualifying results …")
    quali_session = _try_load_session(year, race, 'Q')
    if quali_session is not None:
        real_grid = extract_real_grid(quali_session)

    if real_grid is not None:
        grid_source = "ACTUAL (from Qualifying)"
        grid_positions = {d: float(p) for d, p in real_grid.items()}
        print(f"✓ Using REAL qualifying grid ({len(real_grid)} drivers)")

        # Print actual grid
        grid_rows = sorted(real_grid.items(), key=lambda x: x[1])
        _print_table(
            "QUALIFYING GRID (ACTUAL RESULTS)",
            ["Pos", "Driver"],
            [
                [str(pos), drv]
                for drv, pos in grid_rows
            ],
        )
    else:
        # No real qualifying → run simulation
        print("   ℹ  No qualifying results found — running Monte Carlo prediction")
        print(f"\n⏳ Qualifying sim ({num_iterations:,} iter) …")
        expected_grid, pole_probs, _, quali_drivers = run_quali_sim(
            quali_stats, track_info, season_trends, num_iterations
        )
        grid_positions = expected_grid
        grid_source = "PREDICTED (Monte Carlo)"

        t_q = time.time()
        print(f"✓ Qualifying sim done in {t_q - t1:.1f} s")

        grid_rows = sorted(expected_grid.items(), key=lambda x: x[1])
        _print_table(
            "QUALIFYING PREDICTIONS",
            ["Pos", "Driver", "Exp. Grid", "Pole %"],
            [
                [str(idx + 1), drv, f"{pos:.1f}", _pct(pole_probs.get(drv, 0))]
                for idx, (drv, pos) in enumerate(grid_rows)
            ],
        )

    print(f"\n   Grid source: {grid_source}")

    # ── Step 3: Race simulation ───────────────────────────────────────
    num_laps = 57
    t2 = time.time()
    print(f"\n⏳ Race sim ({num_iterations:,} iter, {num_laps} laps) …")

    # Ensure grid_positions only includes drivers we have race pace for
    sim_grid = {d: grid_positions.get(d, 20) for d in race_stats.keys()}

    finishing_probs, final_ranks, race_drivers, active_mask = run_race_sim(
        race_stats=race_stats,
        reliability_stats=reliability_stats,
        grid_positions=sim_grid,
        num_iterations=num_iterations,
        num_laps=num_laps,
        num_pitstops=2,
        pitstop_time_loss=22.0,
        season_trends=season_trends,
        weather_context=weather_context,
        track_info=track_info,
    )
    t3 = time.time()
    print(f"✓ Race done in {t3 - t2:.1f} s")

    # Print race table
    race_rows = sorted(
        finishing_probs.items(),
        key=lambda x: (x[1]['Win'], x[1]['Podium'], x[1]['Top10'], x[1]['Finish']),
        reverse=True,
    )
    _print_table(
        "RACE PREDICTIONS",
        ["#", "Driver", "Win", "Podium", "Top 10", "Finish", "DNF"],
        [
            [
                str(idx + 1),
                drv,
                _pct(probs['Win']),
                _pct(probs['Podium']),
                _pct(probs['Top10']),
                _pct(probs['Finish']),
                _pct(probs['DNF']),
            ]
            for idx, (drv, probs) in enumerate(race_rows)
        ],
    )

    # ── Step 4: Save JSON ─────────────────────────────────────────────
    output_data = {
        'metadata': {
            'year': year,
            'race': race,
            'iterations': num_iterations,
            'laps': num_laps,
            'grid_source': grid_source,
        },
        'results': {},
    }

    for driver in race_drivers:
        driver_data = {
            'Grid_Position': sim_grid.get(driver, 20),
            'Race': {
                'Win':    round(finishing_probs[driver]['Win'],    4),
                'Podium': round(finishing_probs[driver]['Podium'], 4),
                'Top10':  round(finishing_probs[driver]['Top10'],  4),
                'Finish': round(finishing_probs[driver]['Finish'], 4),
                'DNF':    round(finishing_probs[driver]['DNF'],    4),
            },
        }
        # Include pole probability only if we ran a quali simulation
        if pole_probs:
            driver_data['Pole_Probability'] = round(pole_probs.get(driver, 0.0), 4)

        output_data['results'][driver] = driver_data

    # Include team mapping for dynamic visualizer colors
    output_data['team_mapping'] = team_mapping
    output_data['weather'] = weather_context

    output_file = f'results_{race.lower()}_{year}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Save raw ranks subset for visualizer KDE
    raw_results = {
        'drivers': race_drivers,
        'ranks': final_ranks[:5000].tolist(),
        'active_mask': active_mask[:5000].tolist(),
    }
    with open(f'raw_ranks_{race.lower()}_{year}.json', 'w') as f:
        json.dump(raw_results, f)

    t4 = time.time()
    print(f"\n💾 JSON saved → {output_file}")
    print(f"⏱  Total wall time: {t4 - t0:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Monte Carlo Backtester")
    parser.add_argument('--year', type=int, default=2026, help='Year of the race')
    parser.add_argument('--race', type=str, default='Austria', help='Name of the race')
    parser.add_argument('--iterations', type=int, default=50000, help='Number of Monte Carlo iterations')
    args = parser.parse_args()
    
    main(year=args.year, race=args.race, num_iterations=args.iterations)
